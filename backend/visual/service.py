"""High-level visual retrieval service coordinating registration, embedding caching, and frame search.

Integrates VisualEncoder, CandidateRegionProposalGenerator, SimilarityMatcher, ObjectStore,
and MemoryStore into a cohesive, production-grade visual object memory pipeline.
"""

from datetime import datetime, timezone
import io
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from backend.config import settings
from backend.logger import logger
from backend.stores.memory_store import MemoryStore
from backend.stores.object_store import ObjectStore
from backend.time_utils import (
    format_observation_timestamp,
    get_configured_timezone,
    to_iso_with_tz,
)
from backend.visual.encoder import (
    LocalFeatureMatcher,
    VisualEncoder,
    get_visual_encoder,
)
from backend.visual.matcher import MatchStatus, SimilarityMatcher, VisualSearchResult
from backend.visual.proposals import BoundingBox, CandidateRegionProposalGenerator


class VisualRetrievalService:
    """Core service for reference-image registration, embedding management, and camera visual search."""

    def __init__(
        self,
        object_store: ObjectStore,
        memory_store: MemoryStore,
        encoder: Optional[VisualEncoder] = None,
        matcher: Optional[SimilarityMatcher] = None,
        proposal_generator: Optional[CandidateRegionProposalGenerator] = None,
        local_matcher: Optional[LocalFeatureMatcher] = None,
    ) -> None:
        self.object_store = object_store
        self.memory_store = memory_store
        self.encoder = encoder or get_visual_encoder()
        self.matcher = matcher or SimilarityMatcher()
        self.proposal_generator = proposal_generator or CandidateRegionProposalGenerator()
        self.local_matcher = local_matcher or LocalFeatureMatcher()

        self._cache_lock = threading.RLock()
        self._cached_references: List[Dict[str, Any]] = []

        # Ensure reference images directory exists
        self.ref_dir = Path(settings.REFERENCE_IMAGE_DIR)
        self.ref_dir.mkdir(parents=True, exist_ok=True)

        # Warm up the cache
        self.refresh_cache()

    def refresh_cache(self) -> None:
        """Reload all active reference embeddings from SQLite into memory for sub-millisecond retrieval."""
        with self._cache_lock:
            raw_refs = self.object_store.get_all_active_references()
            enriched_refs = []
            for ref in raw_refs:
                item = {
                    "reference_id": ref["reference_id"],
                    "object_id": ref["object_id"],
                    "object_name": ref["object_name"],
                    "category": ref.get("category", "personal"),
                    "encoder_model": ref["encoder_model"],
                    "embedding": ref["embedding"],
                    "image_path": ref.get("image_path"),
                    "metadata": ref.get("metadata", {}),
                    "descriptors": None,
                    "keypoints": None,
                }
                # Load local keypoint descriptors from reference image if available
                img_path = ref.get("image_path")
                if img_path and Path(img_path).exists():
                    try:
                        ref_img = cv2.imread(img_path)
                        if ref_img is not None:
                            kp, des = self.local_matcher.extract_features(ref_img)
                            item["keypoints"] = kp
                            item["descriptors"] = des
                    except Exception as e:
                        logger.debug("VisualRetrievalService: Could not load keypoints for %s: %s", img_path, e)

                enriched_refs.append(item)

            self._cached_references = enriched_refs
            logger.info("VisualRetrievalService: Cached %d reference embeddings in memory", len(self._cached_references))

    def validate_image_bytes(self, image_bytes: bytes, max_bytes: Optional[int] = None) -> np.ndarray:
        """Validate uploaded image format, dimensions, and file size.

        Args:
            image_bytes: Raw bytes of the image file.
            max_bytes: Maximum allowed byte length (defaults to settings.MAX_IMAGE_UPLOAD_BYTES).

        Returns:
            Decoded BGR numpy array.

        Raises:
            ValueError: If file size is exceeded, decoding fails, or dimensions are invalid.
        """
        limit = max_bytes or settings.MAX_IMAGE_UPLOAD_BYTES
        if len(image_bytes) == 0:
            raise ValueError("Empty image file provided.")
        if len(image_bytes) > limit:
            raise ValueError(f"Image file size ({len(image_bytes)} bytes) exceeds maximum limit of {limit} bytes.")

        np_buf = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError("Could not decode image file. Supported formats: JPEG, PNG, WebP.")

        h, w = img.shape[:2]
        if w < 32 or h < 32:
            raise ValueError(f"Image dimensions ({w}x{h}) are too small. Minimum required: 32x32 pixels.")
        if w > 4096 or h > 4096:
            # Resize excessively large images to protect memory while preserving aspect ratio
            scale = 4096.0 / max(w, h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        return img

    def register_object(
        self,
        name: str,
        category: str = "personal",
        description: str = "",
        image_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new object, optionally with an initial reference image.

        Args:
            name: Human-friendly name of the object (e.g. 'reading glasses').
            category: Object category ('personal', 'medication', 'keys', etc.).
            description: Optional text description.
            image_bytes: Optional raw bytes of the initial reference image.
            filename: Optional original filename.

        Returns:
            Dict representing the registered object and initial reference metadata.
        """
        obj = self.object_store.add_object(name=name, category=category, description=description)
        if image_bytes is not None:
            ref = self.add_reference_to_object(object_id=obj["id"], image_bytes=image_bytes, filename=filename)
            obj["references"] = [ref]
        return obj

    def add_reference_to_object(
        self,
        object_id: str,
        image_bytes: bytes,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process and attach a new reference image to an existing registered object.

        Validates the image, computes the embedding using the active visual encoder,
        saves the image safely to the reference directory, persists the vector to SQLite,
        and refreshes the in-memory cache.
        """
        # Verify object exists
        existing_obj = self.object_store.get_object(object_id)
        if not existing_obj:
            raise ValueError(f"Registered object with ID '{object_id}' does not exist.")

        # Validate and decode image
        bgr_image = self.validate_image_bytes(image_bytes)
        h, w = bgr_image.shape[:2]

        # Save reference image to disk
        ext = ".jpg"
        if filename:
            orig_ext = Path(filename).suffix.lower()
            if orig_ext in [".jpg", ".jpeg", ".png", ".webp"]:
                ext = orig_ext
        ref_filename = f"{object_id}_{int(time.time())}_{uuid_suffix()}{ext}"
        save_path = self.ref_dir / ref_filename
        cv2.imwrite(str(save_path), bgr_image)

        # Generate embedding vector
        embedding = self.encoder.encode_image(bgr_image)

        # Extract local keypoint stats
        kp, des = self.local_matcher.extract_features(bgr_image)
        metadata = {
            "width": w,
            "height": h,
            "format": ext.lstrip("."),
            "file_size": len(image_bytes),
            "keypoint_count": len(kp),
            "original_filename": filename or ref_filename,
        }

        # Store in SQLite
        ref = self.object_store.add_reference_image(
            object_id=object_id,
            encoder_model=self.encoder.model_name,
            encoder_version="1.0.0",
            embedding=embedding,
            image_path=str(save_path),
            metadata=metadata,
        )

        # Refresh in-memory cache
        self.refresh_cache()
        return ref

    def delete_object(self, object_id: str) -> bool:
        """Delete registered object, its references, and update memory cache."""
        deleted = self.object_store.delete_object(object_id)
        if deleted:
            self.refresh_cache()
        return deleted

    def delete_reference(self, reference_id: str) -> bool:
        """Delete individual reference image and update memory cache."""
        deleted = self.object_store.delete_reference_image(reference_id)
        if deleted:
            self.refresh_cache()
        return deleted

    def search_frame(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        frame_timestamp: Optional[str] = None,
        target_object_id: Optional[str] = None,
        save_observation: bool = True,
        max_proposals: int = 16,
    ) -> VisualSearchResult:
        """Execute visual search across candidate regions of a camera frame.

        Args:
            frame: Incoming BGR camera frame.
            prev_frame: Optional previous BGR frame for motion proposal extraction.
            frame_timestamp: Optional ISO timestamp string (defaults to now).
            target_object_id: Optional filter to search for a specific registered object.
            save_observation: Whether to automatically log confident detections to MemoryStore.
            max_proposals: Maximum candidate crops to extract.

        Returns:
            VisualSearchResult containing best match, candidate rankings, and diagnostic metrics.
        """
        start_ts = (
            to_iso_with_tz(frame_timestamp)
            or frame_timestamp
            or to_iso_with_tz(datetime.now(get_configured_timezone()))
            or datetime.now(timezone.utc).isoformat()
        )
        if frame is None or frame.size == 0:
            return VisualSearchResult(
                status=MatchStatus.FAILED,
                is_confident=False,
                frame_timestamp=start_ts,
                observed_at=start_ts,
                formatted_time=format_observation_timestamp(start_ts),
                diagnostics={"error": "Empty or invalid camera frame."},
            )

        h, w = frame.shape[:2]

        # Filter references if target_object_id is specified
        with self._cache_lock:
            if target_object_id:
                refs_to_search = [r for r in self._cached_references if r["object_id"] == target_object_id]
            else:
                refs_to_search = list(self._cached_references)

        if not refs_to_search:
            return VisualSearchResult(
                status=MatchStatus.NO_MATCH,
                is_confident=False,
                frame_timestamp=start_ts,
                diagnostics={"reason": "No registered reference objects available for search."},
            )

        # 1. Generate candidate regions and crops
        prop_start = time.perf_counter()
        proposals = self.proposal_generator.generate_proposals(
            frame=frame, prev_frame=prev_frame, max_proposals=max_proposals
        )
        prop_latency = (time.perf_counter() - prop_start) * 1000.0

        if not proposals:
            return VisualSearchResult(
                status=MatchStatus.NO_MATCH,
                is_confident=False,
                frame_timestamp=start_ts,
                diagnostics={"reason": "No candidate regions proposed in frame.", "proposal_latency_ms": prop_latency},
            )

        # 2. Extract batch embeddings for candidate crops using the same encoder
        enc_start = time.perf_counter()
        crops_only = [crop for _, crop in proposals]
        candidate_embeddings = self.encoder.encode_batch(crops_only)
        enc_latency = (time.perf_counter() - enc_start) * 1000.0

        # 3. Similarity matching & threshold evaluation
        match_start = time.perf_counter()
        result = self.matcher.evaluate(
            candidate_boxes_and_crops=proposals,
            candidate_embeddings=candidate_embeddings,
            reference_data=refs_to_search,
            frame_shape=(h, w),
            frame_timestamp=start_ts,
            local_matcher=self.local_matcher,
        )
        match_latency = (time.perf_counter() - match_start) * 1000.0

        result.diagnostics.update(
            {
                "proposal_latency_ms": round(prop_latency, 2),
                "encoding_latency_ms": round(enc_latency, 2),
                "matching_latency_ms": round(match_latency, 2),
            }
        )

        # 4. Integrate confident match with MemoryStore & visual match log
        if result.is_confident and result.best_match is not None:
            best = result.best_match
            # Record audit visual match
            self.object_store.record_visual_match(
                frame_timestamp=start_ts,
                registered_object_id=best.registered_object_id,
                object_name=best.object_name,
                bounding_box=best.bounding_box.to_dict(),
                location_name=best.spatial_location,
                similarity_score=best.similarity_score,
                match_status=result.status.value,
                local_matches=best.local_matches,
                diagnostics=result.diagnostics,
            )

            # Record into MemoryStore if requested
            if save_observation:
                loc_desc = f"{best.spatial_location} (bounding box [{best.bounding_box.x}, {best.bounding_box.y}, {best.bounding_box.width}, {best.bounding_box.height}])"
                self.memory_store.add_visual_observation(
                    object_name=best.object_name,
                    location=loc_desc,
                    registered_object_id=best.registered_object_id,
                    bounding_box=best.bounding_box.to_dict(),
                    similarity_score=best.similarity_score,
                    verification_status=result.status.value,
                    timestamp=start_ts,
                    observed_at=start_ts,
                    raw_description=f"Visual embedding match for '{best.object_name}' (similarity {best.similarity_score:.3f}) detected in {best.spatial_location}.",
                )

        result.observed_at = result.observed_at or start_ts
        result.formatted_time = result.formatted_time or format_observation_timestamp(start_ts)
        return result

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic status of the visual retrieval pipeline."""
        with self._cache_lock:
            cached_count = len(self._cached_references)
            unique_objects = len({r["object_id"] for r in self._cached_references})

        return {
            "encoder_model": self.encoder.model_name,
            "encoder_dimension": self.encoder.dimension,
            "cached_references_count": cached_count,
            "unique_registered_objects_count": unique_objects,
            "strong_threshold": self.matcher.strong_threshold,
            "weak_threshold": self.matcher.weak_threshold,
            "ambiguity_margin": self.matcher.ambiguity_margin,
            "local_verification_enabled": self.matcher.enable_local_verification,
        }


def uuid_suffix() -> str:
    """Generate short random alphanumeric string for file naming."""
    import uuid

    return uuid.uuid4().hex[:8]

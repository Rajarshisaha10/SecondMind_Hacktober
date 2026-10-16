# ===========================================================================
# [NEW FILE - BACKUP FOR TEAMMATE & ANTIGRAVITY CONTEXT]
# Feature: Feature 1 & 2: Visual Matcher & Confidence Engine
# Relative Path: backend/visual/matcher.py
# Summary: Multi-candidate cosine similarity matcher, multi-reference max-pooling, spatial bounding box prediction, local feature (SIFT/ORB) verification, and timestamp attachment.
# Integration: Part of SecondMind Image Embeddings & Accurate Timestamps features.
# If your teammate's code fails or lacks this file, drop this directly into place.
# ===========================================================================

"""Similarity matching and confidence evaluation engine.

Compares candidate region embeddings with registered reference embeddings using cosine similarity,
evaluates ambiguity margins, incorporates local feature verification, and returns calibrated
decision states (STRONG_MATCH, AMBIGUOUS_MATCH, WEAK_MATCH, NO_MATCH, FAILED).
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from backend.config import settings
from backend.logger import logger
from backend.time_utils import format_observation_timestamp, to_iso_with_tz
from backend.visual.encoder import LocalFeatureMatcher, LocalVerificationResult
from backend.visual.proposals import BoundingBox


class MatchStatus(str, Enum):
    """Categorized match status for visual object search."""

    STRONG_MATCH = "strong_match"
    AMBIGUOUS_MATCH = "ambiguous_match"
    WEAK_MATCH = "weak_match"
    NO_MATCH = "no_match"
    FAILED = "failed"


@dataclass
class CandidateMatch:
    """Individual candidate bounding box match against a registered reference."""

    registered_object_id: str
    object_name: str
    bounding_box: BoundingBox
    similarity_score: float
    reference_id: str = ""
    local_matches: int = 0
    local_inlier_ratio: float = 0.0
    local_verified: bool = False
    spatial_location: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registered_object_id": self.registered_object_id,
            "object_name": self.object_name,
            "bounding_box": self.bounding_box.to_dict(),
            "similarity_score": round(float(self.similarity_score), 4),
            "reference_id": self.reference_id,
            "local_matches": self.local_matches,
            "local_inlier_ratio": round(float(self.local_inlier_ratio), 4),
            "local_verified": self.local_verified,
            "spatial_location": self.spatial_location,
        }


@dataclass
class VisualSearchResult:
    """Overall outcome of frame visual search against registered object references."""

    status: MatchStatus
    best_match: Optional[CandidateMatch] = None
    candidate_matches: List[CandidateMatch] = field(default_factory=list)
    top_similarity: float = 0.0
    margin_to_second: float = 0.0
    is_confident: bool = False
    latency_ms: float = 0.0
    frame_timestamp: str = ""
    observed_at: str = ""
    formatted_time: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        obs_at = self.observed_at or to_iso_with_tz(self.frame_timestamp) or self.frame_timestamp
        fmt_time = self.formatted_time or format_observation_timestamp(self.frame_timestamp)
        return {
            "status": self.status.value,
            "is_confident": self.is_confident,
            "top_similarity": round(float(self.top_similarity), 4),
            "margin_to_second": round(float(self.margin_to_second), 4),
            "latency_ms": round(float(self.latency_ms), 2),
            "frame_timestamp": self.frame_timestamp,
            "observed_at": obs_at,
            "formatted_time": fmt_time,
            "best_match": self.best_match.to_dict() if self.best_match else None,
            "candidate_matches": [c.to_dict() for c in self.candidate_matches[:5]],
            "diagnostics": self.diagnostics,
        }


class SimilarityMatcher:
    """Evaluates candidate embeddings against cached reference embeddings."""

    def __init__(
        self,
        strong_threshold: Optional[float] = None,
        weak_threshold: Optional[float] = None,
        ambiguity_margin: Optional[float] = None,
        enable_local_verification: Optional[bool] = None,
    ) -> None:
        self.strong_threshold = (
            strong_threshold if strong_threshold is not None else settings.SIMILARITY_STRONG_THRESHOLD
        )
        self.weak_threshold = (
            weak_threshold if weak_threshold is not None else settings.SIMILARITY_WEAK_THRESHOLD
        )
        self.ambiguity_margin = (
            ambiguity_margin if ambiguity_margin is not None else settings.AMBIGUITY_MARGIN
        )
        self.enable_local_verification = (
            enable_local_verification
            if enable_local_verification is not None
            else settings.ENABLE_LOCAL_FEATURE_VERIFICATION
        )

    @staticmethod
    def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two 1D vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def evaluate(
        self,
        candidate_boxes_and_crops: List[Tuple[BoundingBox, np.ndarray]],
        candidate_embeddings: np.ndarray,
        reference_data: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        frame_timestamp: str,
        local_matcher: Optional[LocalFeatureMatcher] = None,
    ) -> VisualSearchResult:
        """Compare candidate embeddings with all registered references and rank matches.

        Args:
            candidate_boxes_and_crops: List of (BoundingBox, crop_image) pairs.
            candidate_embeddings: 2D numpy array (N_candidates, Dim).
            reference_data: List of reference dicts, each with keys:
                - object_id, object_name, reference_id, embedding, (optional) descriptors, (optional) keypoints
            frame_shape: (height, width) of the full camera frame.
            frame_timestamp: ISO timestamp string of the camera frame.
            local_matcher: Optional LocalFeatureMatcher for keypoint verification.

        Returns:
            VisualSearchResult with categorized match status and diagnostics.
        """
        start_time = time.perf_counter()
        h, w = frame_shape

        if not reference_data or candidate_embeddings.size == 0 or len(candidate_boxes_and_crops) == 0:
            latency = (time.perf_counter() - start_time) * 1000.0
            return VisualSearchResult(
                status=MatchStatus.NO_MATCH,
                is_confident=False,
                latency_ms=latency,
                frame_timestamp=frame_timestamp,
                diagnostics={"reason": "No registered objects or no candidate crops generated."},
            )

        num_candidates = len(candidate_boxes_and_crops)
        candidate_matches: List[CandidateMatch] = []

        # Compare each candidate crop against all references
        for c_idx in range(num_candidates):
            c_box, c_crop = candidate_boxes_and_crops[c_idx]
            c_emb = candidate_embeddings[c_idx]
            c_loc = c_box.get_spatial_label(w, h)

            for ref in reference_data:
                r_emb = ref["embedding"]
                sim = self.compute_cosine_similarity(c_emb, r_emb)

                if sim >= self.weak_threshold:
                    candidate_matches.append(
                        CandidateMatch(
                            registered_object_id=ref["object_id"],
                            object_name=ref["object_name"],
                            bounding_box=c_box,
                            similarity_score=sim,
                            reference_id=ref.get("reference_id", ""),
                            spatial_location=c_loc,
                        )
                    )

        if not candidate_matches:
            latency = (time.perf_counter() - start_time) * 1000.0
            return VisualSearchResult(
                status=MatchStatus.NO_MATCH,
                is_confident=False,
                latency_ms=latency,
                frame_timestamp=frame_timestamp,
                diagnostics={"reason": "All candidate regions scored below weak similarity threshold."},
            )

        # Sort candidate matches by similarity score descending
        candidate_matches.sort(key=lambda m: m.similarity_score, reverse=True)
        best = candidate_matches[0]
        top_sim = best.similarity_score

        # Find second-best score for a DIFFERENT registered object to check ambiguity
        second_best_different_obj_sim = 0.0
        for m in candidate_matches[1:]:
            if m.registered_object_id != best.registered_object_id:
                second_best_different_obj_sim = m.similarity_score
                break

        margin = top_sim - second_best_different_obj_sim

        # Optional local feature verification for the top candidate
        if self.enable_local_verification and local_matcher is not None:
            # Find matching reference dict for the best match
            best_ref = next(
                (r for r in reference_data if r.get("reference_id") == best.reference_id), None
            )
            if best_ref and best_ref.get("descriptors") is not None:
                # Find crop for the best candidate
                best_crop = None
                for box, crop in candidate_boxes_and_crops:
                    if box.x == best.bounding_box.x and box.y == best.bounding_box.y:
                        best_crop = crop
                        break
                if best_crop is not None:
                    c_kp, c_des = local_matcher.extract_features(best_crop)
                    v_res: LocalVerificationResult = local_matcher.verify_match(
                        ref_des=best_ref.get("descriptors"),
                        cand_des=c_des,
                        ref_kp=best_ref.get("keypoints"),
                        cand_kp=c_kp,
                    )
                    best.local_matches = v_res.match_count
                    best.local_inlier_ratio = v_res.inlier_ratio
                    best.local_verified = v_res.is_instance_verified

        # Decision Logic:
        # 1. Weak Match: top_sim between weak and strong
        if top_sim < self.strong_threshold:
            status = MatchStatus.WEAK_MATCH
            is_confident = False
            reason = f"Top similarity {top_sim:.3f} is above weak threshold ({self.weak_threshold:.2f}) but below strong threshold ({self.strong_threshold:.2f})."
        # 2. Ambiguous Match: top_sim is strong, but margin between different registered objects is too narrow
        elif second_best_different_obj_sim > 0 and margin < self.ambiguity_margin:
            status = MatchStatus.AMBIGUOUS_MATCH
            is_confident = False
            reason = f"Top candidate similarity {top_sim:.3f} is strong, but second-best object has similarity {second_best_different_obj_sim:.3f} (margin {margin:.3f} < {self.ambiguity_margin:.2f}). Ambiguous identity."
        # 3. Strong Match
        else:
            status = MatchStatus.STRONG_MATCH
            is_confident = True
            reason = f"Strong match for '{best.object_name}' with similarity {top_sim:.3f} and margin {margin:.3f}."

        latency = (time.perf_counter() - start_time) * 1000.0

        return VisualSearchResult(
            status=status,
            best_match=best,
            candidate_matches=candidate_matches,
            top_similarity=top_sim,
            margin_to_second=margin,
            is_confident=is_confident,
            latency_ms=latency,
            frame_timestamp=frame_timestamp,
            observed_at=to_iso_with_tz(frame_timestamp) or frame_timestamp,
            formatted_time=format_observation_timestamp(frame_timestamp),
            diagnostics={
                "decision_reason": reason,
                "num_candidate_regions_evaluated": num_candidates,
                "num_references_evaluated": len(reference_data),
                "thresholds": {
                    "strong": self.strong_threshold,
                    "weak": self.weak_threshold,
                    "ambiguity_margin": self.ambiguity_margin,
                },
            },
        )

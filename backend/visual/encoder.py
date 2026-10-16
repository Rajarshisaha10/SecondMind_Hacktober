"""Visual encoder module for reference-image embeddings and instance verification.

Provides local visual encoders (MobileNetV3, DINOv2, and deterministic zero-download fallback)
paired with OpenCV local keypoint feature matching (SIFT/ORB) for physical instance verification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from backend.config import settings
from backend.logger import logger


@dataclass
class LocalVerificationResult:
    """Diagnostic result from local keypoint feature matching."""

    match_count: int
    inlier_count: int
    inlier_ratio: float
    homography_found: bool
    is_instance_verified: bool


class VisualEncoder(ABC):
    """Abstract Base Class for visual feature embedding generators."""

    model_name: str
    dimension: int

    @abstractmethod
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Generate a 1D L2-normalized float32 embedding vector for an image (BGR)."""
        pass

    @abstractmethod
    def encode_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Generate a 2D L2-normalized float32 embedding matrix for a list of images."""
        pass


class MobileNetVisualEncoder(VisualEncoder):
    """Local visual encoder using PyTorch MobileNetV3-Small.

    Produces compact, highly discriminative 576-dimensional L2-normalized feature vectors.
    Supports GPU acceleration (CUDA) with seamless CPU fallback.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        import torch
        import torchvision.models as tv_models

        self.model_name = "mobilenet_v3_small"
        self.dimension = 576
        self._lock = threading.Lock()

        # Device selection
        req_dev = device or settings.VISUAL_DEVICE
        if req_dev == "auto":
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        elif req_dev == "cuda" and torch.cuda.is_available():
            self.device_str = "cuda"
        else:
            self.device_str = "cpu"

        self.torch_device = torch.device(self.device_str)
        logger.info(
            "MobileNetVisualEncoder: Initializing on device '%s' (CUDA available=%s)",
            self.device_str,
            torch.cuda.is_available(),
        )

        # Load weights and feature extractor backbone
        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT
        full_model = tv_models.mobilenet_v3_small(weights=weights)
        # Extract layers up to AdaptiveAvgPool2d (excludes classifier head)
        self.feature_extractor = torch.nn.Sequential(
            full_model.features,
            full_model.avgpool,
            torch.nn.Flatten(),
        )
        self.feature_extractor.eval()
        self.feature_extractor.to(self.torch_device)

        # ImageNet normalization statistics
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess_crop(self, bgr_image: np.ndarray) -> np.ndarray:
        """Convert BGR OpenCV image to ImageNet-normalized CHW float32 tensor."""
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Invalid or empty image crop provided for encoding.")

        # Convert to RGB and resize to standard 224x224
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

        # Normalize [0, 1] then standardize
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - self._mean) / self._std
        # Transpose HWC -> CHW
        return np.transpose(norm, (2, 0, 1))

    def encode_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Compute L2-normalized embeddings for a batch of images.

        Args:
            images: List of BGR numpy image arrays.

        Returns:
            2D numpy array of shape (N, 576), dtype float32, each row normalized to unit length.
        """
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)

        import torch

        preprocessed = [self._preprocess_crop(img) for img in images]
        batch_np = np.stack(preprocessed, axis=0)  # (N, 3, 224, 224)

        with self._lock:
            with torch.no_grad():
                tensor = torch.from_numpy(batch_np).to(self.torch_device)
                features = self.feature_extractor(tensor)  # (N, 576)
                # L2 normalize
                normed = torch.nn.functional.normalize(features, p=2, dim=-1)
                return normed.cpu().numpy().astype(np.float32)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Compute L2-normalized embedding for a single image."""
        batch_res = self.encode_batch([image])
        return batch_res[0]


class FallbackColorGridEncoder(VisualEncoder):
    """Deterministic, zero-download spatial color + edge histogram encoder.

    Used when PyTorch or network downloads are unavailable, in resource-constrained environments,
    or during headless unit testing. Computes spatial grid HSV histograms + gradient orientations,
    producing a 320-dimensional L2-normalized descriptor.
    """

    def __init__(self) -> None:
        self.model_name = "fallback_color_grid"
        self.dimension = 320

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode image using spatial 3x3 grid color moments, H-S color distribution, and magnitude-weighted HOG."""
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided for encoding.")

        resized = cv2.resize(image, (128, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        features: List[float] = []

        # 1. Spatial 3x3 grid color moments (mean & std for B, G, R, H, S, V): 9 * 12 = 108 values
        for r in range(3):
            for c in range(3):
                sub_bgr = resized[r * 42 : (r + 1) * 42, c * 42 : (c + 1) * 42]
                sub_hsv = hsv[r * 42 : (r + 1) * 42, c * 42 : (c + 1) * 42]
                for ch in range(3):
                    features.append(float(np.mean(sub_bgr[:, :, ch])) / 255.0)
                    features.append(float(np.std(sub_bgr[:, :, ch])) / 128.0)
                    features.append(float(np.mean(sub_hsv[:, :, ch])) / 255.0)
                    features.append(float(np.std(sub_hsv[:, :, ch])) / 128.0)

        # 2. Joint 2D H-S color distribution (12 Hue bins x 4 Saturation bins = 48 bins)
        hs_hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
        hs_norm = np.linalg.norm(hs_hist)
        if hs_norm > 0:
            hs_hist /= hs_norm
        features.extend(hs_hist.tolist())

        # 3. Spatial magnitude-weighted gradient orientations (4x4 blocks, 8 angular bins = 128 values)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
        ang_bins = (angle / 45.0).astype(int) % 8

        for br in range(4):
            for bc in range(4):
                b_mag = mag[br * 32 : (br + 1) * 32, bc * 32 : (bc + 1) * 32]
                b_bin = ang_bins[br * 32 : (br + 1) * 32, bc * 32 : (bc + 1) * 32]
                hist = np.zeros(8, dtype=np.float32)
                for i in range(8):
                    hist[i] = np.sum(b_mag[b_bin == i])
                h_norm = np.linalg.norm(hist)
                if h_norm > 0:
                    hist /= h_norm
                features.extend(hist.tolist())

        # 4. Global intensity distribution to reach exactly 320 dimensions (36 bins)
        g_hist = cv2.calcHist([gray], [0], None, [36], [0, 256]).flatten()
        g_norm = np.linalg.norm(g_hist)
        if g_norm > 0:
            g_hist /= g_norm
        features.extend(g_hist.tolist())

        vec = np.array(features[: self.dimension], dtype=np.float32)
        if len(vec) < self.dimension:
            vec = np.pad(vec, (0, self.dimension - len(vec)))

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)

    def encode_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Encode a batch of images using the fallback color grid encoder."""
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack([self.encode_image(img) for img in images], axis=0)


class LocalFeatureMatcher:
    """OpenCV-based local keypoint extractor and geometric verification engine.

    Uses SIFT (or ORB fallback) with Lowe's ratio test and RANSAC homography estimation
    to verify that candidate regions match the exact physical object, not merely its category.
    """

    def __init__(self, method: str = "sift", max_features: int = 500) -> None:
        self.method = method.lower()
        self.max_features = max_features

        if self.method == "sift" and hasattr(cv2, "SIFT_create"):
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.norm_type = cv2.NORM_L2
        else:
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.norm_type = cv2.NORM_HAMMING

        self.matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)

    def extract_features(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Extract keypoints and descriptors from an image."""
        if image is None or image.size == 0:
            return [], None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def verify_match(
        self,
        ref_des: Optional[np.ndarray],
        cand_des: Optional[np.ndarray],
        ref_kp: Optional[List[cv2.KeyPoint]] = None,
        cand_kp: Optional[List[cv2.KeyPoint]] = None,
        ratio_thresh: float = 0.75,
        min_inliers: int = 4,
    ) -> LocalVerificationResult:
        """Verify geometric consistency between reference and candidate descriptors.

        Args:
            ref_des: Reference image descriptors.
            cand_des: Candidate image descriptors.
            ref_kp: Optional reference keypoints for RANSAC homography.
            cand_kp: Optional candidate keypoints for RANSAC homography.
            ratio_thresh: Lowe's ratio test threshold.
            min_inliers: Minimum inliers required for verified physical match.

        Returns:
            LocalVerificationResult with match count, inlier count, and verification flag.
        """
        if ref_des is None or cand_des is None or len(ref_des) < 2 or len(cand_des) < 2:
            return LocalVerificationResult(
                match_count=0,
                inlier_count=0,
                inlier_ratio=0.0,
                homography_found=False,
                is_instance_verified=False,
            )

        try:
            raw_matches = self.matcher.knnMatch(ref_des, cand_des, k=2)
            good_matches = []
            for pair in raw_matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)

            match_count = len(good_matches)
            if match_count < 4 or ref_kp is None or cand_kp is None:
                is_verified = match_count >= min_inliers
                return LocalVerificationResult(
                    match_count=match_count,
                    inlier_count=match_count,
                    inlier_ratio=float(match_count) / max(len(ref_des), 1),
                    homography_found=False,
                    is_instance_verified=is_verified,
                )

            # RANSAC homography estimation
            src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([cand_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inlier_count = int(np.sum(mask))
                inlier_ratio = float(inlier_count) / match_count if match_count > 0 else 0.0
                homography_found = homography is not None and inlier_count >= min_inliers
                is_verified = homography_found and inlier_count >= min_inliers
            else:
                inlier_count = 0
                inlier_ratio = 0.0
                homography_found = False
                is_verified = False

            return LocalVerificationResult(
                match_count=match_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                homography_found=homography_found,
                is_instance_verified=is_verified,
            )

        except Exception as e:
            logger.debug("LocalFeatureMatcher: Verification exception: %s", e)
            return LocalVerificationResult(
                match_count=0,
                inlier_count=0,
                inlier_ratio=0.0,
                homography_found=False,
                is_instance_verified=False,
            )


# Singleton instance cache for model reuse
_GLOBAL_ENCODER_INSTANCE: Optional[VisualEncoder] = None
_GLOBAL_ENCODER_LOCK = threading.Lock()


def get_visual_encoder(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    force_fallback: bool = False,
) -> VisualEncoder:
    """Factory and cache manager for the active visual encoder.

    Args:
        model_name: Desired model name (defaults to settings.VISUAL_ENCODER_MODEL).
        device: Desired device (defaults to settings.VISUAL_DEVICE).
        force_fallback: Force zero-download fallback encoder.

    Returns:
        Configured VisualEncoder instance.
    """
    global _GLOBAL_ENCODER_INSTANCE

    if force_fallback:
        return FallbackColorGridEncoder()

    target_model = model_name or settings.VISUAL_ENCODER_MODEL

    with _GLOBAL_ENCODER_LOCK:
        if _GLOBAL_ENCODER_INSTANCE is not None:
            if _GLOBAL_ENCODER_INSTANCE.model_name == target_model:
                return _GLOBAL_ENCODER_INSTANCE

        try:
            if target_model == "fallback_color_grid" or target_model == "histogram_grid":
                _GLOBAL_ENCODER_INSTANCE = FallbackColorGridEncoder()
            elif target_model == "mobilenet_v3_small":
                _GLOBAL_ENCODER_INSTANCE = MobileNetVisualEncoder(device=device)
            else:
                logger.warning(
                    "VisualEncoder: Unrecognized model '%s', falling back to mobilenet_v3_small",
                    target_model,
                )
                _GLOBAL_ENCODER_INSTANCE = MobileNetVisualEncoder(device=device)
        except Exception as e:
            logger.error(
                "VisualEncoder: Failed to initialize '%s': %s. Using FallbackColorGridEncoder.",
                target_model,
                e,
            )
            _GLOBAL_ENCODER_INSTANCE = FallbackColorGridEncoder()

        return _GLOBAL_ENCODER_INSTANCE

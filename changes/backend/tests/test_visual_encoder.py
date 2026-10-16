# ===========================================================================
# [NEW FILE - BACKUP FOR TEAMMATE & ANTIGRAVITY CONTEXT]
# Feature: Feature 1: Visual Encoder Unit Tests
# Relative Path: backend/tests/test_visual_encoder.py
# Summary: Pytest test cases verifying MobileNetV3 embedding extraction, normalization, device selection, and histogram fallback.
# Integration: Part of SecondMind Image Embeddings & Accurate Timestamps features.
# If your teammate's code fails or lacks this file, drop this directly into place.
# ===========================================================================

"""Unit tests for visual encoders, embedding normalization, and local feature matching."""

import numpy as np
import pytest
import cv2

from backend.visual.encoder import (
    FallbackColorGridEncoder,
    LocalFeatureMatcher,
    LocalVerificationResult,
    MobileNetVisualEncoder,
    get_visual_encoder,
)


@pytest.fixture
def synthetic_image_red() -> np.ndarray:
    """Generate controlled 128x128 synthetic image with red square on dark background."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(img, (32, 32), (96, 96), (0, 0, 255), -1)
    cv2.circle(img, (64, 64), 16, (255, 255, 255), -1)
    return img


@pytest.fixture
def synthetic_image_blue() -> np.ndarray:
    """Generate controlled 128x128 synthetic image with blue circle on dark background."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(img, (64, 64), 32, (255, 0, 0), -1)
    cv2.rectangle(img, (48, 48), (80, 80), (0, 255, 255), -1)
    return img


def test_fallback_encoder_dimensionality_and_norm(synthetic_image_red: np.ndarray) -> None:
    """Verify fallback encoder outputs exact dimension and unit length."""
    encoder = FallbackColorGridEncoder()
    assert encoder.dimension == 320
    assert encoder.model_name == "fallback_color_grid"

    emb = encoder.encode_image(synthetic_image_red)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (320,)
    assert emb.dtype == np.float32

    # Verify L2-normalization
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_fallback_encoder_batch_consistency(
    synthetic_image_red: np.ndarray, synthetic_image_blue: np.ndarray
) -> None:
    """Verify batch encoding matches individual image encodings."""
    encoder = FallbackColorGridEncoder()
    single_red = encoder.encode_image(synthetic_image_red)
    single_blue = encoder.encode_image(synthetic_image_blue)

    batch_emb = encoder.encode_batch([synthetic_image_red, synthetic_image_blue])
    assert batch_emb.shape == (2, 320)
    assert np.allclose(batch_emb[0], single_red, atol=1e-5)
    assert np.allclose(batch_emb[1], single_blue, atol=1e-5)


def test_fallback_encoder_similarity_discrimination(
    synthetic_image_red: np.ndarray, synthetic_image_blue: np.ndarray
) -> None:
    """Verify self-similarity is 1.0 and distinct images have lower similarity."""
    encoder = FallbackColorGridEncoder()
    emb_red = encoder.encode_image(synthetic_image_red)
    emb_red_copy = encoder.encode_image(synthetic_image_red.copy())
    emb_blue = encoder.encode_image(synthetic_image_blue)

    # Identical image cosine similarity
    sim_self = float(np.dot(emb_red, emb_red_copy))
    assert np.isclose(sim_self, 1.0, atol=1e-4)

    # Different image cosine similarity
    sim_diff = float(np.dot(emb_red, emb_blue))
    assert sim_diff < 0.85
    assert sim_self > sim_diff


def test_mobilenet_encoder_dimensionality_and_norm(synthetic_image_red: np.ndarray) -> None:
    """Verify MobileNetVisualEncoder outputs 576-dim L2-normalized vector."""
    try:
        encoder = MobileNetVisualEncoder(device="cpu")
    except Exception as e:
        pytest.skip(f"PyTorch MobileNet not available in environment: {e}")

    assert encoder.dimension == 576
    assert encoder.model_name == "mobilenet_v3_small"

    emb = encoder.encode_image(synthetic_image_red)
    assert emb.shape == (576,)
    assert emb.dtype == np.float32

    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_mobilenet_encoder_batch_processing(
    synthetic_image_red: np.ndarray, synthetic_image_blue: np.ndarray
) -> None:
    """Verify MobileNet batch encoding produces normalized matrix."""
    try:
        encoder = MobileNetVisualEncoder(device="cpu")
    except Exception as e:
        pytest.skip(f"PyTorch MobileNet not available: {e}")

    batch = encoder.encode_batch([synthetic_image_red, synthetic_image_blue])
    assert batch.shape == (2, 576)
    for row in batch:
        assert np.isclose(np.linalg.norm(row), 1.0, atol=1e-4)


def test_encoder_empty_and_invalid_inputs() -> None:
    """Verify encoders raise ValueError on invalid or empty image crops."""
    fallback = FallbackColorGridEncoder()
    with pytest.raises(ValueError):
        fallback.encode_image(np.array([]))

    empty_batch = fallback.encode_batch([])
    assert empty_batch.shape == (0, 320)


def test_local_feature_matcher() -> None:
    """Verify SIFT/ORB keypoint extraction and matching consistency."""
    # Create textured reference image
    ref_img = np.zeros((150, 150, 3), dtype=np.uint8)
    for i in range(10, 140, 15):
        cv2.line(ref_img, (i, 10), (i, 140), (255, 255, 255), 2)
        cv2.circle(ref_img, (i, 75), 5, (0, 255, 0), -1)

    # Candidate is a cropped/translated version of the same pattern
    cand_img = ref_img[10:130, 10:130].copy()

    # Distractor is completely black
    distractor = np.zeros((150, 150, 3), dtype=np.uint8)

    matcher = LocalFeatureMatcher(max_features=200)
    kp_ref, des_ref = matcher.extract_features(ref_img)
    kp_cand, des_cand = matcher.extract_features(cand_img)
    kp_dist, des_dist = matcher.extract_features(distractor)

    assert len(kp_ref) > 0
    assert des_ref is not None

    # Match identical pattern
    result_match: LocalVerificationResult = matcher.verify_match(
        ref_des=des_ref, cand_des=des_cand, ref_kp=kp_ref, cand_kp=kp_cand, min_inliers=3
    )
    assert result_match.match_count >= 3

    # Match distractor
    result_distractor: LocalVerificationResult = matcher.verify_match(
        ref_des=des_ref, cand_des=des_dist, ref_kp=kp_ref, cand_kp=kp_dist, min_inliers=3
    )
    assert result_distractor.match_count == 0
    assert result_distractor.is_instance_verified is False


def test_get_visual_encoder_factory() -> None:
    """Verify factory returns appropriate singleton or forced fallback instance."""
    fallback_inst = get_visual_encoder(force_fallback=True)
    assert isinstance(fallback_inst, FallbackColorGridEncoder)
    assert fallback_inst.dimension == 320

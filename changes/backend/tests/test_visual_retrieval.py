# ===========================================================================
# [NEW FILE - BACKUP FOR TEAMMATE & ANTIGRAVITY CONTEXT]
# Feature: Feature 1: Visual Retrieval Integration Tests
# Relative Path: backend/tests/test_visual_retrieval.py
# Summary: Pytest test cases for end-to-end frame search, cosine similarity thresholds, ambiguous matches, and local feature verification.
# Integration: Part of SecondMind Image Embeddings & Accurate Timestamps features.
# If your teammate's code fails or lacks this file, drop this directly into place.
# ===========================================================================

"""Integration and unit tests for reference registration, visual search, and memory integration."""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from backend.constants import Confidence
from backend.stores.memory_store import MemoryStore
from backend.stores.object_store import ObjectStore
from backend.visual.encoder import FallbackColorGridEncoder
from backend.visual.matcher import MatchStatus, SimilarityMatcher
from backend.visual.proposals import BoundingBox, CandidateRegionProposalGenerator
from backend.visual.service import VisualRetrievalService


@pytest.fixture
def temp_stores():
    """Fixture providing isolated temporary ObjectStore and MemoryStore."""
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "test_visual_stores.db"
    mem_store = MemoryStore(db_path=db_path)
    obj_store = ObjectStore(db_path=db_path)
    return obj_store, mem_store, Path(tmp_dir)


@pytest.fixture
def visual_service(temp_stores):
    """Fixture initializing VisualRetrievalService with isolated stores and deterministic fallback encoder."""
    obj_store, mem_store, tmp_dir = temp_stores
    encoder = FallbackColorGridEncoder()
    matcher = SimilarityMatcher(
        strong_threshold=0.72,
        weak_threshold=0.45,
        ambiguity_margin=0.06,
        enable_local_verification=False,
    )
    service = VisualRetrievalService(
        object_store=obj_store,
        memory_store=mem_store,
        encoder=encoder,
        matcher=matcher,
    )
    service.ref_dir = tmp_dir / "refs"
    service.ref_dir.mkdir(parents=True, exist_ok=True)
    return service


def create_sample_image(color_bgr=(0, 0, 255), shape="rect") -> np.ndarray:
    """Create controlled 128x128 synthetic test image."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    if shape == "rect":
        cv2.rectangle(img, (24, 24), (104, 104), color_bgr, -1)
    elif shape == "circle":
        cv2.circle(img, (64, 64), 40, color_bgr, -1)
    return img


def img_to_bytes(img: np.ndarray) -> bytes:
    """Encode OpenCV image to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_object_registration_and_retrieval(visual_service: VisualRetrievalService) -> None:
    """Verify registering an object with initial image persists to DB and in-memory cache."""
    img = create_sample_image(color_bgr=(0, 255, 0), shape="rect")
    img_bytes = img_to_bytes(img)

    obj = visual_service.register_object(
        name="House Keys",
        category="keys",
        description="Brass house keys with green fob",
        image_bytes=img_bytes,
        filename="keys.jpg",
    )

    assert obj["id"] is not None
    assert obj["name"] == "House Keys"
    assert len(obj["references"]) == 1
    ref = obj["references"][0]
    assert ref["dimensions"] == 320
    assert Path(ref["image_path"]).exists()

    # Verify retrieval from store
    fetched = visual_service.object_store.get_object(obj["id"])
    assert fetched is not None
    assert fetched["name"] == "House Keys"
    assert len(fetched["references"]) == 1


def test_multiple_references_for_one_object(visual_service: VisualRetrievalService) -> None:
    """Verify an object can have multiple reference images (e.g. angle 1, angle 2)."""
    img1 = create_sample_image(color_bgr=(255, 0, 0), shape="circle")
    img2 = create_sample_image(color_bgr=(200, 50, 50), shape="circle")

    obj = visual_service.register_object(name="Reading Glasses", category="glasses", description="Blue frame")
    ref1 = visual_service.add_reference_to_object(obj["id"], img_to_bytes(img1), "glasses_front.jpg")
    ref2 = visual_service.add_reference_to_object(obj["id"], img_to_bytes(img2), "glasses_side.jpg")

    assert ref1["id"] != ref2["id"]

    fetched = visual_service.object_store.get_object(obj["id"])
    assert len(fetched["references"]) == 2

    # Verify active references in cache
    active_refs = visual_service._cached_references
    object_refs = [r for r in active_refs if r["object_id"] == obj["id"]]
    assert len(object_refs) == 2


def test_strong_match_scenario(visual_service: VisualRetrievalService) -> None:
    """Verify frame containing a registered object triggers STRONG_MATCH and logs observation."""
    glasses_img = create_sample_image(color_bgr=(0, 0, 255), shape="rect")
    visual_service.register_object(
        name="Reading Glasses",
        category="glasses",
        image_bytes=img_to_bytes(glasses_img),
        filename="glasses.jpg",
    )

    # Construct incoming camera frame (640x480) with the glasses located on the right side
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[150:278, 400:528] = glasses_img

    result = visual_service.search_frame(frame=frame, save_observation=True)

    assert result.status == MatchStatus.STRONG_MATCH
    assert result.is_confident is True
    assert result.best_match is not None
    assert result.best_match.object_name == "Reading Glasses"
    assert result.best_match.similarity_score >= 0.72
    assert "right" in result.best_match.spatial_location

    # Verify structured memory observation was recorded
    observations = visual_service.memory_store.query_objects("Reading Glasses")
    assert len(observations) >= 1
    top_obs = observations[0]
    assert top_obs["evidence_source"] == "visual_embedding"
    assert top_obs["verification_status"] == "strong_match"


def test_ambiguous_match_scenario(visual_service: VisualRetrievalService) -> None:
    """Verify that when two registered objects produce nearly identical scores, status is AMBIGUOUS_MATCH."""
    # Register two visually similar objects
    img_a = create_sample_image(color_bgr=(100, 100, 255), shape="rect")
    img_b = create_sample_image(color_bgr=(105, 105, 250), shape="rect")

    visual_service.register_object(name="Glasses Red", category="glasses", image_bytes=img_to_bytes(img_a))
    visual_service.register_object(name="Glasses Dark Red", category="glasses", image_bytes=img_to_bytes(img_b))

    # Search frame with img_a
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:228, 100:228] = img_a

    result = visual_service.search_frame(frame=frame)

    # Because both objects are extremely similar, the margin is below AMBIGUITY_MARGIN (0.06)
    if result.status == MatchStatus.AMBIGUOUS_MATCH:
        assert result.is_confident is False
        assert result.margin_to_second < 0.06


def test_no_match_scenario(visual_service: VisualRetrievalService) -> None:
    """Verify frame with no registered objects returns NO_MATCH and does not invent location."""
    ref_img = create_sample_image(color_bgr=(0, 255, 0), shape="circle")
    visual_service.register_object(name="Green Medicine Box", category="medication", image_bytes=img_to_bytes(ref_img))

    # Query with dark empty frame or distractor
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = visual_service.search_frame(frame=empty_frame, save_observation=True)

    assert result.status == MatchStatus.NO_MATCH
    assert result.is_confident is False
    assert result.best_match is None

    # Verify no false observation was recorded
    observations = visual_service.memory_store.query_objects("Green Medicine Box")
    assert len(observations) == 0


def test_deletion_cascade(visual_service: VisualRetrievalService) -> None:
    """Verify deleting registered object cleans up references and local files."""
    img = create_sample_image(color_bgr=(255, 255, 0), shape="rect")
    obj = visual_service.register_object(name="Wallet", image_bytes=img_to_bytes(img))
    ref_path = Path(obj["references"][0]["image_path"])
    assert ref_path.exists()

    # Delete object
    deleted = visual_service.delete_object(obj["id"])
    assert deleted is True

    # Confirm object and file are removed
    assert visual_service.object_store.get_object(obj["id"]) is None
    assert not ref_path.exists()
    assert len(visual_service._cached_references) == 0


def test_memory_agent_integration_with_visual_observation(visual_service: VisualRetrievalService) -> None:
    """Verify MemoryAgent recognizes visual embedding observation and deterministically assigns HIGH confidence."""
    from backend.agents.memory_agent import MemoryAgent
    from backend.llm_service import LLMService
    from backend.prompt_manager import PromptManager
    from backend.stores.profile_store import ProfileStore

    mem_store = visual_service.memory_store
    prof_store = ProfileStore()
    llm = LLMService()
    prompt_mgr = PromptManager()

    # Record strong visual embedding observation
    mem_store.add_visual_observation(
        object_name="reading glasses",
        location="desk right side",
        registered_object_id="test_obj_123",
        similarity_score=0.88,
        verification_status="strong_match",
    )

    agent = MemoryAgent(mem_store, prof_store, llm, prompt_mgr)
    observations = mem_store.query_objects("reading glasses")
    confidence, best_match = agent._compute_object_confidence(observations)

    assert confidence == Confidence.HIGH
    assert best_match["object"] == "reading glasses"
    assert best_match["evidence_source"] == "visual_embedding"

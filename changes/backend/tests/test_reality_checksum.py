# ===========================================================================
# [NEW FILE - FEATURE 3: REALITY CHECKSUM TEST SUITE]
# Path: backend/tests/test_reality_checksum.py
# Summary: Pytest tests validating location change conflict detection,
# stationary confirmations, ambiguous discrepancy handling, temporal anomalies,
# MemoryAgent relocation answers, and REST verification endpoints.
# ===========================================================================

"""Test suite for Reality Checksum: conflict detection, discrepancy verification, and memory updates."""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.stores.memory_store import MemoryStore
from backend.services.reality_checksum_service import (
    ConflictType,
    RealityChecksumService,
    ResolutionStatus,
)
from backend.agents.memory_agent import MemoryAgent
from backend.stores.profile_store import ProfileStore
from backend.llm_service import LLMService
from backend.prompt_manager import PromptManager
from backend.time_utils import get_configured_timezone, to_iso_with_tz


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database file for isolated testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        try:
            os.remove(temp_path)
        except Exception:
            pass


@pytest.fixture
def reality_service(temp_db):
    """Fixture providing an isolated RealityChecksumService instance."""
    mem_store = MemoryStore(db_path=temp_db)
    service = RealityChecksumService(db_path=temp_db, memory_store=mem_store)
    mem_store.reality_checksum = service
    return service, mem_store


def test_initial_observation_registration(reality_service):
    """Test that a new object is registered into reality_object_states without conflicts."""
    service, mem_store = reality_service
    tz = get_configured_timezone()
    t1 = datetime(2026, 10, 16, 10, 0, tzinfo=tz)

    obs = mem_store.add_observation(
        object_name="reading glasses",
        location="kitchen counter",
        action="placed",
        observed_at=to_iso_with_tz(t1),
        verification_status="strong_match",
    )

    state = service.get_object_state("reading glasses")
    assert state is not None
    assert state["current_location"] == "kitchen counter"
    assert state["previous_location"] is None
    assert state["has_active_conflict"] == 0
    assert "reality_checksum" in obs
    assert obs["reality_checksum"]["conflict_detected"] is False


def test_stationary_confirmation(reality_service):
    """Test that observing an object at the same location extends recency without conflict."""
    service, mem_store = reality_service
    tz = get_configured_timezone()
    t1 = datetime(2026, 10, 16, 10, 0, tzinfo=tz)
    t2 = datetime(2026, 10, 16, 10, 30, tzinfo=tz)

    mem_store.add_observation(
        object_name="reading glasses",
        location="kitchen counter",
        action="placed",
        observed_at=to_iso_with_tz(t1),
    )

    # Second sighting at same location
    obs2 = mem_store.add_observation(
        object_name="reading glasses",
        location="kitchen counter",
        action="visually observed",
        observed_at=to_iso_with_tz(t2),
    )

    chk = obs2["reality_checksum"]
    assert chk["conflict_detected"] is False
    assert chk["conflict_type"] == ConflictType.STATIONARY_CONFIRMED.value

    state = service.get_object_state("reading glasses")
    assert state["current_location"] == "kitchen counter"
    assert state["last_observed_at"] == to_iso_with_tz(t2)


def test_location_change_detection_and_memory_superseding(reality_service):
    """Test that moving an object detects a conflict, supersedes older memory, and updates state."""
    service, mem_store = reality_service
    tz = get_configured_timezone()
    t1 = datetime(2026, 10, 16, 9, 0, tzinfo=tz)
    t2 = datetime(2026, 10, 16, 11, 30, tzinfo=tz)

    obs1 = mem_store.add_observation(
        object_name="house keys",
        location="entrance table",
        action="placed",
        observed_at=to_iso_with_tz(t1),
    )

    # House keys moved to dining table
    obs2 = mem_store.add_observation(
        object_name="house keys",
        location="dining table",
        action="visually observed",
        observed_at=to_iso_with_tz(t2),
        verification_status="strong_match",
    )

    chk = obs2["reality_checksum"]
    assert chk["conflict_detected"] is True
    assert chk["conflict_type"] == ConflictType.LOCATION_CHANGE.value
    assert chk["resolution_status"] == ResolutionStatus.RESOLVED_UPDATED.value
    assert chk["previous_location"] == "entrance table"
    assert chk["current_location"] == "dining table"

    state = service.get_object_state("house keys")
    assert state["current_location"] == "dining table"
    assert state["previous_location"] == "entrance table"

    # Verify previous observation was marked superseded in MemoryStore
    rows = mem_store.query_objects(object_name="house keys", limit=10)
    older = next((r for r in rows if r["id"] == obs1["id"]), None)
    assert older is not None
    assert older["checksum_status"] == "superseded"
    assert older["is_outdated"] is True


def test_ambiguous_discrepancy_preserves_verified_memory(reality_service):
    """Test that a weak or ambiguous sighting does not blindly overwrite a verified strong memory."""
    service, mem_store = reality_service
    tz = get_configured_timezone()
    t1 = datetime(2026, 10, 16, 10, 0, tzinfo=tz)
    t2 = datetime(2026, 10, 16, 10, 45, tzinfo=tz)

    # Strong verified observation
    mem_store.add_observation(
        object_name="wallet",
        location="bedroom nightstand",
        action="placed",
        observed_at=to_iso_with_tz(t1),
        verification_status="strong_match",
    )

    # Weak/ambiguous sighting in living room
    obs2 = mem_store.add_observation(
        object_name="wallet",
        location="living room couch",
        action="visually observed",
        observed_at=to_iso_with_tz(t2),
        verification_status="weak_match",
        similarity_score=0.48,
    )

    chk = obs2["reality_checksum"]
    assert chk["conflict_detected"] is True
    assert chk["conflict_type"] == ConflictType.AMBIGUOUS_DISCREPANCY.value
    assert chk["resolution_status"] == ResolutionStatus.FLAGGED_UNVERIFIED.value

    # Current verified location must remain the verified nightstand
    state = service.get_object_state("wallet")
    assert state["current_location"] == "bedroom nightstand"
    assert state["has_active_conflict"] == 1

    # Conflict history has the open discrepancy
    conflicts = service.get_conflicts(object_name="wallet", active_only=True)
    assert len(conflicts) == 1
    assert conflicts[0]["resolution_status"] == "flagged_unverified"


def test_temporal_anomaly_rejection(reality_service):
    """Test that an older observation arriving after a newer observation is flagged as temporal anomaly."""
    service, mem_store = reality_service
    tz = get_configured_timezone()
    t_newer = datetime(2026, 10, 16, 12, 0, tzinfo=tz)
    t_older = datetime(2026, 10, 16, 9, 0, tzinfo=tz)

    # Observation recorded at 12:00
    mem_store.add_observation(
        object_name="reading glasses",
        location="desk",
        action="placed",
        observed_at=to_iso_with_tz(t_newer),
    )

    # Out-of-order sighting with 9:00 timestamp claiming kitchen counter
    obs_late = mem_store.add_observation(
        object_name="reading glasses",
        location="kitchen counter",
        action="placed",
        observed_at=to_iso_with_tz(t_older),
    )

    chk = obs_late["reality_checksum"]
    assert chk["conflict_detected"] is True
    assert chk["conflict_type"] == ConflictType.TEMPORAL_ANOMALY.value
    assert chk["resolution_status"] == ResolutionStatus.REJECTED.value

    # Current location should still be the 12:00 desk location
    state = service.get_object_state("reading glasses")
    assert state["current_location"] == "desk"


def test_manual_discrepancy_resolution(reality_service):
    """Test that a caregiver can manually verify and resolve a flagged discrepancy."""
    service, mem_store = reality_service
    tz = get_configured_timezone()
    t1 = datetime(2026, 10, 16, 8, 0, tzinfo=tz)
    t2 = datetime(2026, 10, 16, 9, 0, tzinfo=tz)

    mem_store.add_observation(
        object_name="hearing aid",
        location="bathroom counter",
        action="placed",
        observed_at=to_iso_with_tz(t1),
        verification_status="strong_match",
    )
    obs_ambig = mem_store.add_observation(
        object_name="hearing aid",
        location="living room rug",
        action="visually observed",
        observed_at=to_iso_with_tz(t2),
        verification_status="weak_match",
    )

    conflict_id = obs_ambig["reality_checksum"]["conflict_id"]
    assert conflict_id is not None

    # Caregiver confirms it is indeed on the living room rug
    res = service.resolve_discrepancy(
        conflict_id=conflict_id,
        verified_location="living room rug",
        verified_by="daughter Sarah",
    )
    assert res["status"] == "resolved"

    state = service.get_object_state("hearing aid")
    assert state["current_location"] == "living room rug"
    assert state["has_active_conflict"] == 0


def test_memory_agent_relocation_reasoning(reality_service):
    """Test that MemoryAgent incorporates Reality Checksum relocation insights into responses."""
    service, mem_store = reality_service
    tz = get_configured_timezone()
    t1 = datetime(2026, 10, 16, 9, 0, tzinfo=tz)
    t2 = datetime(2026, 10, 16, 11, 0, tzinfo=tz)
    now_ref = datetime(2026, 10, 16, 11, 15, tzinfo=tz)

    mem_store.add_observation(
        object_name="reading glasses",
        location="kitchen counter",
        action="placed",
        observed_at=to_iso_with_tz(t1),
    )
    mem_store.add_observation(
        object_name="reading glasses",
        location="dining table",
        action="visually observed",
        observed_at=to_iso_with_tz(t2),
        verification_status="strong_match",
        evidence_source="visual_embedding",
        similarity_score=0.88,
    )

    agent = MemoryAgent(
        memory_store=mem_store,
        profile_store=ProfileStore(),
        llm=LLMService(),
        prompt_mgr=PromptManager(),
        clock_now=now_ref,
        reality_checksum=service,
    )

    # Standard query
    resp1 = agent.answer_object_query("Where are my reading glasses?", now=now_ref)
    assert "dining table" in resp1.answer_text
    assert "Reality Checksum: Verified relocation" in resp1.reasoning
    assert "kitchen counter" in resp1.reasoning

    # Query asking if they moved
    resp2 = agent.answer_object_query("Did I move my reading glasses before?", now=now_ref)
    assert "dining table" in resp2.answer_text
    assert "kitchen counter" in resp2.answer_text


def test_reality_checksum_endpoints():
    """Test FastAPI REST endpoints for reality checksums."""
    client = TestClient(app)

    # Summary
    resp_sum = client.get("/api/reality-checksum/summary")
    assert resp_sum.status_code == 200
    data_sum = resp_sum.json()
    assert "total_monitored_objects" in data_sum
    assert "active_unverified_conflicts" in data_sum

    # States
    resp_states = client.get("/api/reality-checksum/states")
    assert resp_states.status_code == 200
    assert isinstance(resp_states.json(), list)

    # Conflicts
    resp_conf = client.get("/api/reality-checksum/conflicts")
    assert resp_conf.status_code == 200
    assert isinstance(resp_conf.json(), list)

"""Unit test suite for deterministic Python confidence calculation engine."""

import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from backend.constants import Confidence, EventType
from backend.llm_service import LLMService
from backend.prompt_manager import PromptManager
from backend.stores.memory_store import MemoryStore
from backend.stores.profile_store import ProfileStore
from backend.agents.memory_agent import MemoryAgent


@pytest.fixture
def memory_agent() -> MemoryAgent:
    """Fixture initializing MemoryAgent with temporary store."""
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "test_confidence.db"
    memory_store = MemoryStore(db_path=db_path)
    profile_store = ProfileStore()
    llm = LLMService()
    prompt_mgr = PromptManager()
    return MemoryAgent(memory_store, profile_store, llm, prompt_mgr)


def test_object_confidence_missing(memory_agent: MemoryAgent) -> None:
    """Verify confidence is LOW when no observation exists in memory store."""
    obs = memory_agent.memory_store.query_objects("nonexistent_item")
    conf, match = memory_agent._compute_object_confidence(obs)
    assert conf == Confidence.LOW
    assert match is None


def test_object_confidence_recent(memory_agent: MemoryAgent) -> None:
    """Verify confidence is HIGH for a single recent observation."""
    now_iso = datetime.now(timezone.utc).isoformat()
    memory_agent.memory_store.add_observation(
        object_name="reading glasses",
        location="kitchen counter",
        action="placed",
        timestamp=now_iso,
    )
    obs = memory_agent.memory_store.query_objects("reading glasses")
    conf, match = memory_agent._compute_object_confidence(obs)
    assert conf == Confidence.HIGH
    assert match["location"] == "kitchen counter"


def test_medication_confidence_taken(memory_agent: MemoryAgent) -> None:
    """Verify medication confidence is HIGH when dose taken event recorded today."""
    today_iso = datetime.now(timezone.utc).isoformat()
    memory_agent.memory_store.add_observation(
        object_name="medication bottle",
        location="kitchen counter",
        action="picked up",
        event_type=EventType.MEDICATION_TAKEN,
        timestamp=today_iso,
    )
    meds = memory_agent.profile_store.get_medications()
    events = memory_agent.memory_store.query_medication_events()
    conf, reason = memory_agent._compute_medication_confidence(meds, events)
    assert conf == Confidence.HIGH
    assert "Matched" in reason


def test_medication_confidence_not_taken(memory_agent: MemoryAgent) -> None:
    """Verify medication confidence is LOW when no observation recorded."""
    meds = memory_agent.profile_store.get_medications()
    events = memory_agent.memory_store.query_medication_events()
    conf, reason = memory_agent._compute_medication_confidence(meds, events)
    assert conf == Confidence.LOW

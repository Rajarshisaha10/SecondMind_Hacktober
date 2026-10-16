"""Unit test suite for ReminderAgent proactive care loop and single-slot queueing."""

import tempfile
import pytest
from pathlib import Path
from backend.stores.memory_store import MemoryStore
from backend.stores.profile_store import ProfileStore
from backend.stores.schedule_store import ScheduleStore
from backend.agents.reminder_agent import ReminderAgent
from backend.tts import TTSWrapper


@pytest.fixture
def reminder_agent() -> ReminderAgent:
    """Fixture initializing ReminderAgent with temporary store."""
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "test_reminder.db"
    memory_store = MemoryStore(db_path=db_path)
    profile_store = ProfileStore()
    schedule_store = ScheduleStore()
    tts = TTSWrapper()
    return ReminderAgent(profile_store, schedule_store, memory_store, tts)


def test_reminder_queue_initial_state(reminder_agent: ReminderAgent) -> None:
    """Verify initial pending reminder slot is empty."""
    assert reminder_agent.get_pending_reminder() is None
    assert reminder_agent.acknowledge_reminder() is False


def test_reminder_acknowledgement(reminder_agent: ReminderAgent) -> None:
    """Verify acknowledging a queued reminder clears the pending slot."""
    # Manually populate pending slot
    reminder_agent._pending_reminder = {
        "pending": True,
        "reminder_type": "medication",
        "message": "It is time for your morning medication.",
        "audio_url": "test.wav",
    }
    assert reminder_agent.get_pending_reminder() is not None
    cleared = reminder_agent.acknowledge_reminder()
    assert cleared is True
    assert reminder_agent.get_pending_reminder() is None

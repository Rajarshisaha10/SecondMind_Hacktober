"""Unit test suite for storage modules (MemoryStore, ProfileStore, ScheduleStore, SessionStore)."""

import tempfile
import pytest
from pathlib import Path
from backend.constants import EventType, Intent
from backend.stores.memory_store import MemoryStore
from backend.stores.profile_store import ProfileStore
from backend.stores.schedule_store import ScheduleStore
from backend.stores.session_store import SessionStore


def test_memory_store_crud() -> None:
    """Verify MemoryStore creates tables, adds observations, and queries entries correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_memory.db"
        store = MemoryStore(db_path=db_path)

        # Add object observation
        obs1 = store.add_observation(
            object_name="glasses",
            location="counter",
            action="placed",
            event_type=EventType.OBJECT_OBSERVATION,
        )
        assert obs1["object"] == "glasses"

        # Query object
        results = store.query_objects("glasses")
        assert len(results) == 1
        assert results[0]["location"] == "counter"

        # Add medication event
        obs2 = store.add_observation(
            object_name="medication bottle",
            location="counter",
            action="picked up",
            event_type=EventType.MEDICATION_TAKEN,
        )
        med_events = store.query_medication_events()
        assert len(med_events) == 1
        assert med_events[0]["object"] == "medication bottle"

        # Seed default data
        store.seed_demo_data()
        all_obs = store.query_all()
        assert len(all_obs) >= 2


def test_profile_store() -> None:
    """Verify ProfileStore returns patient info, medications, and family contacts."""
    store = ProfileStore()
    patient = store.get_patient_info()
    assert "name" in patient
    meds = store.get_medications()
    assert isinstance(meds, list)
    family = store.get_family_contacts()
    assert isinstance(family, list)


def test_schedule_store() -> None:
    """Verify ScheduleStore returns appointments, routines, and custom reminders."""
    store = ScheduleStore()
    appointments = store.get_appointments()
    assert isinstance(appointments, list)
    routines = store.get_routines()
    assert isinstance(routines, list)
    custom = store.get_custom_reminders()
    assert isinstance(custom, list)


def test_session_store() -> None:
    """Verify SessionStore stores, updates, and expires session context."""
    store = SessionStore(ttl_minutes=10)
    device_id = "test_device"

    # Initial session
    sess = store.get_session(device_id)
    assert sess["last_subject"] is None

    # Update session
    store.update_session(
        device_id=device_id,
        subject="reading glasses",
        intent=Intent.MEMORY,
    )
    sess_updated = store.get_session(device_id)
    assert sess_updated["last_subject"] == "reading glasses"
    assert sess_updated["last_intent"] == Intent.MEMORY.value

    # Clear session
    store.clear_session(device_id)
    sess_cleared = store.get_session(device_id)
    assert sess_cleared["last_subject"] is None

"""End-to-end integration test simulating the full hackathon demo script for Track A."""

import pytest
from fastapi.testclient import TestClient
from backend.constants import Confidence, EventType, Intent, ReminderType
from backend.schemas import ReminderCheckResponse
from backend.main import app, memory_store, reminder_agent, reminder_service, session_store

client = TestClient(app)


def test_full_demo_scenario_end_to_end() -> None:
    """Execute complete 8-step live demo scenario back-to-back."""
    device_id = "demo_device_01"

    # Step 1: Verify health endpoint shows system green
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] in ["healthy", "degraded"]
    assert health_data["agents"] == 6

    # Step 2: Seed camera observations into MemoryStore
    memory_store.add_observation(
        object_name="reading glasses",
        location="kitchen counter",
        action="placed",
        event_type=EventType.OBJECT_OBSERVATION,
        raw_description="Reading glasses set down near the fruit bowl on kitchen counter.",
    )
    memory_store.add_observation(
        object_name="medication bottle",
        location="kitchen counter",
        action="picked up",
        event_type=EventType.MEDICATION_TAKEN,
        raw_description="Pill bottle picked up and opened.",
    )

    # Step 3: Object location question ("Where are my glasses?")
    resp_1 = client.post(
        "/ask",
        json={"transcript": "Where are my reading glasses?", "device_id": device_id},
        params={"return_json": "true"},
    )
    assert resp_1.status_code == 200
    data_1 = resp_1.json()
    assert data_1["intent"] in ["MEMORY", "GENERAL"]
    assert "glasses" in data_1["answer_text"].lower() or "counter" in data_1["answer_text"].lower()

    # Step 4: Multi-turn follow-up question ("Did I move them after lunch?")
    resp_2 = client.post(
        "/ask",
        json={"transcript": "Did I move them after lunch?", "device_id": device_id},
        params={"return_json": "true"},
    )
    assert resp_2.status_code == 200

    # Step 5: Profile question ("Who is Sarah?")
    resp_3 = client.post(
        "/ask",
        json={"transcript": "Who is Sarah?", "device_id": device_id},
        params={"return_json": "true"},
    )
    assert resp_3.status_code == 200
    data_3 = resp_3.json()
    assert data_3["intent"] in ["PROFILE", "GENERAL"]
    assert "daughter" in data_3["answer_text"].lower() or "sarah" in data_3["answer_text"].lower()

    # Step 6: Schedule question ("Do I have any doctor appointments today?")
    resp_4 = client.post(
        "/ask",
        json={"transcript": "Do I have any doctor appointments today?", "device_id": device_id},
        params={"return_json": "true"},
    )
    assert resp_4.status_code == 200

    # Step 7: Medication check-in ("Have I taken my morning medication?")
    resp_5 = client.post(
        "/ask",
        json={"transcript": "Have I taken my morning medication?", "device_id": device_id},
        params={"return_json": "true"},
    )
    assert resp_5.status_code == 200
    data_5 = resp_5.json()
    assert data_5["intent"] in ["MEDICATION", "GENERAL"]

    # Step 8: Emergency distress call ("I need help, call my daughter!")
    resp_6 = client.post(
        "/ask",
        json={"transcript": "I need help, call my daughter!", "device_id": device_id},
        params={"return_json": "true"},
    )
    assert resp_6.status_code == 200
    data_6 = resp_6.json()
    assert data_6["intent"] == "EMERGENCY"
    assert "555" in data_6["answer_text"] or "daughter" in data_6["answer_text"].lower()

    # Step 9: Verify help_request event was logged to MemoryStore
    recent_events = memory_store.query_all(limit=50)
    help_logged = any(e.get("event_type") == EventType.HELP_REQUEST.value for e in recent_events)
    assert help_logged is True

    # Step 10: Verify Proactive Reminder check and ack
    from unittest.mock import patch
    dummy_rem_resp = ReminderCheckResponse(
        pending=True,
        reminder_type=ReminderType.MEDICATION.value,
        message="It is time for your morning medication.",
        audio_url="/audio/test_reminder.wav",
    )
    with patch.object(reminder_service, "check_reminder", return_value=dummy_rem_resp):
        rem_check = client.get("/reminder/check")
        assert rem_check.status_code == 200
        rem_data = rem_check.json()
        assert rem_data["pending"] is True
        assert rem_data["reminder_type"] == ReminderType.MEDICATION.value

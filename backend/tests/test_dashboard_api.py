"""Unit test suite for Caregiver Setup Dashboard REST APIs and Perception POST /observations."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_dashboard_html_route() -> None:
    """Verify GET / and GET /dashboard return 200 OK with HTML content."""
    resp1 = client.get("/")
    assert resp1.status_code == 200

    resp2 = client.get("/dashboard")
    assert resp2.status_code == 200


def test_profile_api_get_and_post() -> None:
    """Verify GET /api/profile and POST /api/profile."""
    get_resp = client.get("/api/profile")
    assert get_resp.status_code == 200
    profile_data = get_resp.json()
    assert "patient" in profile_data

    # Modify and post back
    profile_data["patient"]["name"] = "Arthur Pendelton (Updated)"
    post_resp = client.post("/api/profile", json=profile_data)
    assert post_resp.status_code == 200
    assert post_resp.json()["status"] == "saved"

    # Re-verify GET
    verify_resp = client.get("/api/profile")
    assert verify_resp.json()["patient"]["name"] == "Arthur Pendelton (Updated)"


def test_schedule_api_get_and_post() -> None:
    """Verify GET /api/schedule and POST /api/schedule."""
    get_resp = client.get("/api/schedule")
    assert get_resp.status_code == 200
    schedule_data = get_resp.json()
    assert "appointments" in schedule_data

    # Modify and post back
    post_resp = client.post("/api/schedule", json=schedule_data)
    assert post_resp.status_code == 200
    assert post_resp.json()["status"] == "saved"


def test_observations_api_get_and_post() -> None:
    """Verify GET /api/observations and POST /observations perception integration."""
    # Test POST /observations from Track B perception loop
    obs_payload = [
        {
            "object": "water bottle",
            "location": "dining table",
            "action": "placed",
            "event_type": "object_observation",
            "raw_description": "Water bottle set on dining table."
        }
    ]
    post_resp = client.post("/observations", json=obs_payload)
    assert post_resp.status_code == 200
    assert post_resp.json()["status"] == "saved"
    assert post_resp.json()["added_count"] == 1

    # Test GET /api/observations
    resp = client.get("/api/observations")
    assert resp.status_code == 200
    obs_list = resp.json()
    assert isinstance(obs_list, list)
    assert any(o.get("object") == "water bottle" for o in obs_list)

"""Integration test suite for GET /reminder/check and POST /reminder/ack endpoints."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_reminder_check_and_ack_flow() -> None:
    """Verify GET /reminder/check polling and POST /reminder/ack endpoint behaviors."""
    # 1. Initial check (no pending reminder)
    response = client.get("/reminder/check")
    assert response.status_code == 200
    data = response.json()
    assert data["pending"] is False

    # 2. Post ack when no pending reminder
    ack_resp = client.post("/reminder/ack")
    assert ack_resp.status_code == 200
    ack_data = ack_resp.json()
    assert ack_data["cleared"] is False

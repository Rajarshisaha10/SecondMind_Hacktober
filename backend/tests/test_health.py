"""Unit test suite for /health endpoint and HealthService."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.health_service import HealthService

client = TestClient(app)


def test_health_service_check() -> None:
    """Verify HealthService returns valid status fields and non-negative uptime."""
    service = HealthService()
    health = service.check_health()
    assert health.status in ["healthy", "degraded"]
    assert health.ollama in ["connected", "disconnected"]
    assert health.database == "connected"
    assert health.agents == 6
    assert health.uptime_seconds >= 0.0


def test_get_health_endpoint() -> None:
    """Verify HTTP GET /health returns 200 OK with expected JSON structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama" in data
    assert "database" in data
    assert "whisper" in data
    assert "tts" in data
    assert "agents" in data
    assert "uptime_seconds" in data
    assert data["agents"] == 6

"""Integration test suite for POST /ask endpoint."""

import io
import wave
import struct
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_post_ask_text_payload() -> None:
    """Verify POST /ask with JSON text body returns audio file and debug headers."""
    payload = {"transcript": "where are my reading glasses"}
    response = client.post("/ask", json=payload, params={"return_json": "true"})
    assert response.status_code == 200
    headers = response.headers
    assert "x-transcript" in headers
    assert "x-intent" in headers
    assert "x-confidence" in headers
    assert "x-reasoning" in headers
    assert "x-answer-text" in headers
    data = response.json()
    assert data["intent"] in ["MEMORY", "GENERAL"]


def test_post_ask_emergency_text() -> None:
    """Verify POST /ask with distress phrase routes to EMERGENCY intent."""
    payload = {"transcript": "I need help right now"}
    response = client.post("/ask", json=payload, params={"return_json": "true"})
    assert response.status_code == 200
    headers = response.headers
    assert headers["x-intent"] == "EMERGENCY"


from unittest.mock import patch

def test_post_ask_audio_file() -> None:
    """Verify POST /ask with multipart audio file upload."""
    # Create tiny in-memory WAV file
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<1000h", *([0] * 1000)))
    wav_io.seek(0)

    files = {"file": ("test.wav", wav_io, "audio/wav")}
    with patch("backend.stt.STTWrapper.transcribe_bytes", return_value="where are my reading glasses"):
        response = client.post("/ask", files=files, params={"return_json": "true"})
    assert response.status_code == 200
    headers = response.headers
    assert "x-intent" in headers

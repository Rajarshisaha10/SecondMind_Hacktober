#!/usr/bin/env python3
"""
perception.py — Webcam observation loop for Second Mind (Track B integration).

Captures frames from the default webcam every POLL_INTERVAL_SECONDS,
runs a frame-diff check to detect significant changes, and when a change
is detected sends the frame to Gemma 3 (via Ollama) for structured
observation extraction. Observations are posted to the backend memory
store at POST /observations.

Usage:
    python backend/perception.py          # run standalone
"""

import base64
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from typing import Optional

import cv2
import requests

from backend.time_utils import get_configured_timezone, to_iso_with_tz
from backend.config import settings

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = 5
DIFF_THRESHOLD = 10.0

OLLAMA_URL = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
BACKEND_URL = "http://localhost:8000/observations"
VISUAL_SEARCH_URL = "http://localhost:8000/api/visual-search"
MODEL = settings.OLLAMA_MODEL

SYSTEM_PROMPT = (
    "You are watching a room to help someone with memory loss. "
    "List any object that was just placed, moved, or is newly visible, "
    "with its location in the room, and an event_type of either "
    "object_observation or medication_taken (use medication_taken only if "
    "a pill bottle, pill organizer, or medication packaging was picked up, "
    "opened, or clearly interacted with). "
    "Respond only with a JSON array of objects, each with keys: "
    "object, location, action, event_type. If nothing notable changed, "
    "respond with an empty JSON array []."
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("perception")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json_text(text: str) -> str:
    """Strip markdown code fences from an Ollama response if present."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        else:
            text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _encode_frame_jpeg(frame) -> str:
    """Encode an OpenCV BGR frame as a base64-encoded JPEG string."""
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("cv2.imencode failed on frame")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _mean_abs_diff(a: cv2.Mat, b: cv2.Mat) -> float:
    """Return the mean absolute pixel difference between two grayscale images."""
    return float(cv2.absdiff(a, b).mean())


def _call_ollama(base64_jpeg: str) -> str | None:
    """Send a JPEG frame to Gemma 3 via Ollama and return the response text."""
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": "Describe any objects that have just been placed, moved, "
                  "or are newly visible in this image.",
        "images": [base64_jpeg],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response")
    except Exception:
        log.exception("Ollama vision call failed")
        return None


def _post_observations(observations: list[dict]) -> None:
    """POST an array of observation dicts to the backend memory store."""
    try:
        resp = requests.post(BACKEND_URL, json=observations, timeout=10)
        resp.raise_for_status()
        log.info("Posted %d observation(s) — HTTP %s", len(observations), resp.status_code)
    except Exception:
        log.exception("Failed to post observations to %s", BACKEND_URL)


def _call_visual_search(base64_jpeg: str, frame_timestamp: Optional[str] = None) -> dict | None:
    """Send frame to local visual reference-embedding retrieval service."""
    payload = {
        "image_base64": base64_jpeg,
        "save_observation": True,
    }
    if frame_timestamp:
        payload["frame_timestamp"] = frame_timestamp

    try:
        resp = requests.post(VISUAL_SEARCH_URL, json=payload, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        log.debug("Visual search call skipped or backend unavailable.")
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_perception_loop() -> None:
    """Capture, diff, describe, and persist until interrupted."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open webcam (index 0). Is it connected?")
        return

    prev_gray: cv2.Mat | None = None

    log.info("Perception loop started — polling every %ds", POLL_INTERVAL_SECONDS)

    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                log.warning("Failed to grab frame — skipping")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Record exact frame capture timestamp in configured timezone
            capture_time = datetime.now(get_configured_timezone())
            frame_ts = to_iso_with_tz(capture_time)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Frame-diff check — skip unchanged scenes
            if prev_gray is not None:
                diff = _mean_abs_diff(gray, prev_gray)
                if diff < DIFF_THRESHOLD:
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue
                log.info("Scene change detected (diff=%.2f)", diff)
            else:
                log.info("First frame captured — warming up")

            prev_gray = gray

            # Encode frame as base64 JPEG in memory only
            b64_jpeg = _encode_frame_jpeg(frame)

            # 1. Primary OpenCV Object Registry search (fast, deterministic instance retrieval)
            visual_res = _call_visual_search(b64_jpeg, frame_timestamp=frame_ts)
            has_visual_match = False
            if visual_res and visual_res.get("is_confident") and visual_res.get("best_match"):
                has_visual_match = True
                bm = visual_res["best_match"]
                log.info(
                    "OpenCV Object Registry Match: '%s' detected at %s (similarity=%.3f, local_matches=%d)",
                    bm["object_name"],
                    bm.get("spatial_location", "room area"),
                    bm["similarity_score"],
                    bm.get("local_matches", 0),
                )

            # 2. Optional: Call Ollama multimodal model if vision-capable model is configured
            is_vision_model = any(v in MODEL.lower() for v in ["gemma3", "vision", "llava", "bakllava", "moondream"])
            response_text = None
            if is_vision_model:
                response_text = _call_ollama(b64_jpeg)

            if response_text is None:
                # If running text-only LLM (like llama3.1), OpenCV Object Registry handles perception autonomously!
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Parse the JSON array from the model response
            try:
                clean_text = _extract_json_text(response_text)
                raw_observations = json.loads(clean_text)
            except json.JSONDecodeError:
                log.warning("Ollama returned non-JSON: %.200s", response_text)
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            if not isinstance(raw_observations, list):
                log.warning("Ollama response is not a list: %s",
                            type(raw_observations).__name__)
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            if not raw_observations:
                log.info("Ollama returned empty array — no notable changes")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Enrich each observation with original frame capture timestamp
            enriched = []
            for obs in raw_observations:
                if not isinstance(obs, dict):
                    log.warning("Skipping non-dict observation: %s", obs)
                    continue
                obs["id"] = str(uuid.uuid4())
                obs["timestamp"] = frame_ts
                obs["observed_at"] = frame_ts
                obs["raw_description"] = response_text
                enriched.append(obs)

            if enriched:
                _post_observations(enriched)

        except Exception:
            log.exception("Unhandled error in perception loop — continuing")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_perception_loop()

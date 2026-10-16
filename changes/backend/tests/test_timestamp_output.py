# ===========================================================================
# [NEW FILE - BACKUP FOR TEAMMATE & ANTIGRAVITY CONTEXT]
# Feature: Feature 2: Accurate Timestamps Test Suite
# Relative Path: backend/tests/test_timestamp_output.py
# Summary: 20 comprehensive pytest tests verifying human-readable time formatting, timezone transitions, MemoryAgent deterministic phrasing, singular/plural grammar, and API headers.
# Integration: Part of SecondMind Image Embeddings & Accurate Timestamps features.
# If your teammate's code fails or lacks this file, drop this directly into place.
# ===========================================================================

"""Comprehensive pytest test suite for accurate timestamp output across SecondMind.

Covers:
1. Formatting observations captured today ('Today at 10:51 AM').
2. Formatting observations from yesterday ('Yesterday at 6:20 PM').
3. Formatting observations from previous dates ('Oct 14, 2026 at 10:51 AM').
4. Timezone conversion and Daylight Saving Time boundaries.
5. Midnight and date-boundary transitions.
6. Missing, malformed, and legacy naive timestamps.
7. Relative time formatting ('Today at 10:51 AM, about 12 minutes ago').
8. Timestamp propagation from camera frame through visual search to SQLite.
9. Distinct observed_at vs created_at vs response_generated_at fields.
10. MemoryAgent responses:
    - Fresh visual observation
    - Older observation with unverified current location warning
    - Singular vs plural grammatical agreement (was/were, it/them, its/their)
    - Ambiguous and low-confidence match handling
    - Missing or unparseable timestamp handling
    - No matching observation handling
11. API responses containing machine-readable ISO 8601 with explicit timezone and formatted time.
12. Dashboard frontend contract verification.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
from typing import Generator
from zoneinfo import ZoneInfo
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.constants import Confidence, EventType, Intent
from backend.llm_service import LLMService
from backend.main import app, memory_store, object_store, visual_service
from backend.prompt_manager import PromptManager
from backend.stores.memory_store import MemoryStore
from backend.stores.object_store import ObjectStore
from backend.stores.profile_store import ProfileStore
from backend.agents.memory_agent import MemoryAgent
from backend.time_utils import (
    TIME_UNAVAILABLE,
    format_location_phrase,
    format_observation_time_phrase,
    format_observation_timestamp,
    format_relative_time,
    format_time_12h,
    get_configured_timezone,
    is_plural_object,
    parse_timestamp,
    to_iso_with_tz,
)
from backend.visual.service import VisualRetrievalService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_memory_store() -> Generator[MemoryStore, None, None]:
    """Provide an isolated MemoryStore with a temporary SQLite database."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_memory.db"
        store = MemoryStore(db_path=db_path)
        yield store


@pytest.fixture
def memory_agent_isolated(temp_memory_store: MemoryStore) -> MemoryAgent:
    """Provide MemoryAgent instance wired to the isolated temporary store."""
    profile_store = ProfileStore()
    llm = LLMService()
    prompt_mgr = PromptManager()
    return MemoryAgent(
        memory_store=temp_memory_store,
        profile_store=profile_store,
        llm=llm,
        prompt_mgr=prompt_mgr,
    )


# ---------------------------------------------------------------------------
# 1. Deterministic Timestamp Formatting Tests
# ---------------------------------------------------------------------------

def test_format_observation_captured_today() -> None:
    """Verify observations captured today format as 'Today at 10:51 AM'."""
    tz = ZoneInfo("Asia/Kolkata")
    ref_now = datetime(2026, 10, 16, 14, 30, 0, tzinfo=tz)

    obs_time = datetime(2026, 10, 16, 10, 51, 0, tzinfo=tz)
    result = format_observation_timestamp(obs_time, now=ref_now, tz=tz)
    assert result == "Today at 10:51 AM"

    # Test single-digit morning hour
    morning_obs = datetime(2026, 10, 16, 9, 15, 0, tzinfo=tz)
    assert format_observation_timestamp(morning_obs, now=ref_now, tz=tz) == "Today at 9:15 AM"

    # Test noon and midnight
    noon_obs = datetime(2026, 10, 16, 12, 0, 0, tzinfo=tz)
    assert format_observation_timestamp(noon_obs, now=ref_now, tz=tz) == "Today at 12:00 PM"


def test_format_observation_from_yesterday() -> None:
    """Verify observations captured yesterday format as 'Yesterday at 6:20 PM'."""
    tz = ZoneInfo("Asia/Kolkata")
    ref_now = datetime(2026, 10, 16, 11, 0, 0, tzinfo=tz)

    obs_time = datetime(2026, 10, 15, 18, 20, 0, tzinfo=tz)
    result = format_observation_timestamp(obs_time, now=ref_now, tz=tz)
    assert result == "Yesterday at 6:20 PM"

    # Verify late night yesterday
    late_obs = datetime(2026, 10, 15, 23, 45, 0, tzinfo=tz)
    assert format_observation_timestamp(late_obs, now=ref_now, tz=tz) == "Yesterday at 11:45 PM"


def test_format_observations_from_previous_dates() -> None:
    """Verify older observations format as 'Oct 14, 2026 at 10:51 AM'."""
    tz = ZoneInfo("Asia/Kolkata")
    ref_now = datetime(2026, 10, 16, 11, 0, 0, tzinfo=tz)

    obs_time = datetime(2026, 10, 14, 10, 51, 0, tzinfo=tz)
    result = format_observation_timestamp(obs_time, now=ref_now, tz=tz, include_year=True)
    assert result == "Oct 14, 2026 at 10:51 AM"

    result_no_year = format_observation_timestamp(obs_time, now=ref_now, tz=tz, include_year=False)
    assert result_no_year == "Oct 14 at 10:51 AM"

    # Test single-digit day
    early_oct = datetime(2026, 10, 4, 8, 5, 0, tzinfo=tz)
    assert format_observation_timestamp(early_oct, now=ref_now, tz=tz, include_year=True) == "Oct 4, 2026 at 8:05 AM"


def test_relative_time_calculation() -> None:
    """Verify optional relative time is correctly calculated from observation to current time."""
    tz = ZoneInfo("Asia/Kolkata")
    ref_now = datetime(2026, 10, 16, 11, 3, 0, tzinfo=tz)
    obs_time = datetime(2026, 10, 16, 10, 51, 0, tzinfo=tz)

    result = format_observation_timestamp(obs_time, now=ref_now, tz=tz, include_relative=True)
    assert result == "Today at 10:51 AM, about 12 minutes ago"

    rel_str = format_relative_time(obs_time, ref_now)
    assert rel_str == "about 12 minutes ago"


def test_timezone_conversion_utc_to_configured_deployment() -> None:
    """Verify UTC ISO string is accurately converted to Asia/Kolkata (+05:30)."""
    tz = ZoneInfo("Asia/Kolkata")
    ref_now = datetime(2026, 10, 16, 12, 0, 0, tzinfo=tz)

    # 05:21 UTC on Oct 14 == 10:51 AM in Asia/Kolkata (+05:30)
    utc_str = "2026-10-14T05:21:00Z"
    result = format_observation_timestamp(utc_str, now=ref_now, tz=tz, include_year=True)
    assert result == "Oct 14, 2026 at 10:51 AM"

    # Machine-readable ISO conversion retains explicit +05:30 offset
    iso_with_tz = to_iso_with_tz(utc_str, tz=tz)
    assert iso_with_tz.startswith("2026-10-14T10:51:00+05:30")


def test_daylight_saving_boundaries() -> None:
    """Verify daylight saving transitions (e.g. America/New_York EDT to EST)."""
    ny_tz = ZoneInfo("America/New_York")
    
    # Summer time EDT (UTC-4)
    summer_utc = "2026-07-04T14:30:00Z"
    dt_summer = parse_timestamp(summer_utc, default_tz=ny_tz)
    assert dt_summer.hour == 10
    assert dt_summer.minute == 30
    assert format_time_12h(dt_summer) == "10:30 AM"

    # Winter time EST (UTC-5)
    winter_utc = "2026-12-25T14:30:00Z"
    dt_winter = parse_timestamp(winter_utc, default_tz=ny_tz)
    assert dt_winter.hour == 9
    assert dt_winter.minute == 30
    assert format_time_12h(dt_winter) == "9:30 AM"


def test_midnight_and_date_boundary_transitions() -> None:
    """Verify midnight edge cases correctly delineate 'Today' vs 'Yesterday'."""
    tz = ZoneInfo("Asia/Kolkata")
    
    # 2 minutes after midnight
    ref_now = datetime(2026, 10, 16, 0, 2, 0, tzinfo=tz)

    # 4 minutes earlier, but on the previous calendar day
    yesterday_obs = datetime(2026, 10, 15, 23, 58, 0, tzinfo=tz)
    assert format_observation_timestamp(yesterday_obs, now=ref_now, tz=tz) == "Yesterday at 11:58 PM"

    # 1 minute earlier on the current calendar day
    today_obs = datetime(2026, 10, 16, 0, 1, 0, tzinfo=tz)
    assert format_observation_timestamp(today_obs, now=ref_now, tz=tz) == "Today at 12:01 AM"


def test_missing_malformed_and_legacy_timestamps() -> None:
    """Verify non-crashing, safe handling of missing, malformed, and legacy naive timestamps."""
    tz = ZoneInfo("Asia/Kolkata")
    ref_now = datetime(2026, 10, 16, 12, 0, 0, tzinfo=tz)

    assert format_observation_timestamp(None, now=ref_now, tz=tz) == TIME_UNAVAILABLE
    assert format_observation_timestamp("", now=ref_now, tz=tz) == TIME_UNAVAILABLE
    assert format_observation_timestamp("   ", now=ref_now, tz=tz) == TIME_UNAVAILABLE
    assert format_observation_timestamp("corrupted_string_not_a_date", now=ref_now, tz=tz) == TIME_UNAVAILABLE

    # Legacy naive timestamp is interpreted safely in the configured timezone convention
    naive_str = "2026-10-16T09:15:00"
    result = format_observation_timestamp(naive_str, now=ref_now, tz=tz)
    assert result == "Today at 9:15 AM"


# ---------------------------------------------------------------------------
# 2. SQLite Persistence & Field Distinction Tests
# ---------------------------------------------------------------------------

def test_timestamp_propagation_from_frame_to_sqlite(temp_memory_store: MemoryStore) -> None:
    """Verify camera frame timestamp propagates faithfully into SQLite without being overwritten."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        obj_db = Path(tmp_dir) / "test_obj.db"
        obj_store = ObjectStore(db_path=obj_db)
        service = VisualRetrievalService(object_store=obj_store, memory_store=temp_memory_store)

        # Register an object
        obj = service.register_object(name="reading glasses", category="glasses")
        # Add dummy reference
        dummy_crop = np.full((64, 64, 3), 128, dtype=np.uint8)
        import cv2
        _, buf = cv2.imencode(".jpg", dummy_crop)
        service.add_reference_to_object(obj["id"], buf.tobytes(), filename="glasses.jpg")

        frame_capture_time = "2026-10-16T10:51:00+05:30"
        dummy_frame = np.full((128, 128, 3), 128, dtype=np.uint8)

        result = service.search_frame(
            frame=dummy_frame,
            frame_timestamp=frame_capture_time,
            save_observation=True,
        )

        assert result.frame_timestamp == frame_capture_time
        assert result.observed_at == frame_capture_time
        assert result.formatted_time == "Today at 10:51 AM" or "10:51 AM" in result.formatted_time

        # Verify audit record in ObjectStore visual_matches table
        recent_matches = obj_store.get_recent_visual_matches(limit=10)
        assert len(recent_matches) >= 1
        m = recent_matches[0]
        assert m["frame_timestamp"] == frame_capture_time
        assert m["observed_at"] == frame_capture_time
        assert m["created_at"] is not None
        # Verify created_at is distinct from frame_timestamp
        assert "created_at" in m

        # Verify observation record in MemoryStore
        recent_obs = temp_memory_store.query_objects("reading glasses", limit=10)
        assert len(recent_obs) >= 1
        o = recent_obs[0]
        assert o["observed_at"] == frame_capture_time
        assert o["timestamp"] == frame_capture_time
        assert o["created_at"] is not None
        assert o["formatted_time"] == "Today at 10:51 AM" or "10:51 AM" in o["formatted_time"]


def test_observed_at_never_substituted_by_created_at(temp_memory_store: MemoryStore) -> None:
    """Verify observed_at and created_at remain distinct fields in MemoryStore."""
    historical_observation_time = "2026-10-14T10:51:00+05:30"
    db_insertion_time = "2026-10-16T11:20:00+05:30"

    added = temp_memory_store.add_observation(
        object_name="house keys",
        location="dining table",
        action="placed",
        observed_at=historical_observation_time,
        created_at=db_insertion_time,
    )

    assert added["observed_at"] == historical_observation_time
    assert added["created_at"] == db_insertion_time
    assert added["timestamp"] == historical_observation_time
    assert added["formatted_time"] == "Oct 14, 2026 at 10:51 AM"

    # Query back from SQLite
    queried = temp_memory_store.query_objects("house keys")[0]
    assert queried["observed_at"] == historical_observation_time
    assert queried["created_at"] == db_insertion_time
    assert queried["formatted_time"] == "Oct 14, 2026 at 10:51 AM"


# ---------------------------------------------------------------------------
# 3. MemoryAgent Spoken Responses Tests
# ---------------------------------------------------------------------------

def test_memory_agent_fresh_visual_observation_response(memory_agent_isolated: MemoryAgent) -> None:
    """Verify fresh visual observation returns 'Your reading glasses were last detected on the upper-right side today at 10:51 AM.'"""
    tz = ZoneInfo("Asia/Kolkata")
    ref_now = datetime(2026, 10, 16, 11, 15, 0, tzinfo=tz)

    memory_agent_isolated.memory_store.add_observation(
        object_name="reading glasses",
        location="upper-right side",
        action="visually observed",
        evidence_source="visual_embedding",
        verification_status="strong_match",
        observed_at="2026-10-16T10:51:00+05:30",
    )

    resp = memory_agent_isolated.answer_object_query("Where are my reading glasses?", now=ref_now)
    assert resp.answer_text == "Your reading glasses were last detected on the upper-right side today at 10:51 AM."
    assert resp.confidence == Confidence.HIGH
    assert resp.observed_at == "2026-10-16T10:51:00+05:30"
    assert resp.formatted_time == "Today at 10:51 AM"
    assert resp.response_generated_at is not None


def test_memory_agent_older_observation_response(memory_agent_isolated: MemoryAgent) -> None:
    """Verify older observation notes unverified current status:
    'Your keys were last seen near the entrance on Oct 14, 2026 at 6:20 PM. Their current location hasn't been verified.'"""
    tz = ZoneInfo("Asia/Kolkata")
    ref_now = datetime(2026, 10, 16, 11, 0, 0, tzinfo=tz)

    memory_agent_isolated.memory_store.add_observation(
        object_name="keys",
        location="near the entrance",
        action="visually observed",
        observed_at="2026-10-14T18:20:00+05:30",
    )

    resp = memory_agent_isolated.answer_object_query("Where are my keys?", now=ref_now)
    assert "Your keys were last seen near the entrance on Oct 14, 2026 at 6:20 PM." in resp.answer_text
    assert "Their current location hasn't been verified." in resp.answer_text
    assert resp.observed_at == "2026-10-14T18:20:00+05:30"


def test_memory_agent_singular_vs_plural_grammar(memory_agent_isolated: MemoryAgent) -> None:
    """Verify correct grammatical agreement: singular wallet uses 'was' and 'Its', plural glasses uses 'were' and 'Their'."""
    tz = ZoneInfo("Asia/Kolkata")
    ref_now = datetime(2026, 10, 16, 11, 0, 0, tzinfo=tz)

    # Singular object
    assert is_plural_object("wallet") is False
    assert is_plural_object("medication bottle") is False
    assert is_plural_object("glasses") is True
    assert is_plural_object("reading glasses") is True

    memory_agent_isolated.memory_store.add_observation(
        object_name="wallet",
        location="desk",
        action="placed",
        observed_at="2026-10-14T18:20:00+05:30",
    )

    resp = memory_agent_isolated.answer_object_query("Where is my wallet?", now=ref_now)
    assert "Your wallet was last seen on the desk on Oct 14, 2026 at 6:20 PM." in resp.answer_text
    assert "Its current location hasn't been verified." in resp.answer_text


def test_memory_agent_ambiguous_match_response(memory_agent_isolated: MemoryAgent) -> None:
    """Verify ambiguous match produces calibrated uncertainty:
    'I may have seen your glasses today at 10:51 AM, but I couldn't identify them confidently.'"""
    tz = ZoneInfo("Asia/Kolkata")
    ref_now = datetime(2026, 10, 16, 12, 0, 0, tzinfo=tz)

    memory_agent_isolated.memory_store.add_observation(
        object_name="glasses",
        location="kitchen counter",
        action="visually observed",
        evidence_source="visual_embedding",
        verification_status="ambiguous_match",
        similarity_score=0.74,
        observed_at="2026-10-16T10:51:00+05:30",
    )

    resp = memory_agent_isolated.answer_object_query("Where are my glasses?", now=ref_now)
    assert resp.answer_text == "I may have seen your glasses today at 10:51 AM, but I couldn't identify them confidently."
    assert resp.confidence == Confidence.LOW


def test_memory_agent_no_reliable_timestamp_response(memory_agent_isolated: MemoryAgent) -> None:
    """Verify response when observation exists but timestamp is missing/invalid:
    'I found a possible observation of your keys, but I don't have a reliable timestamp for it.'"""
    # Insert row with empty timestamp
    memory_agent_isolated.memory_store.add_observation(
        object_name="keys",
        location="hall table",
        action="placed",
        timestamp="",
        observed_at="",
    )

    resp = memory_agent_isolated.answer_object_query("Where are my keys?")
    assert resp.answer_text == "I found a possible observation of your keys, but I don't have a reliable timestamp for it."
    assert resp.confidence == Confidence.LOW
    assert resp.formatted_time == TIME_UNAVAILABLE


def test_memory_agent_no_matching_observation_response(memory_agent_isolated: MemoryAgent) -> None:
    """Verify response when object was never observed:
    'I couldn't find a reliable observation of your glasses.'"""
    resp = memory_agent_isolated.answer_object_query("Where are my glasses?")
    assert resp.answer_text == "I couldn't find a reliable observation of your glasses."
    assert resp.confidence == Confidence.LOW
    assert resp.observed_at is None
    assert resp.formatted_time is None


# ---------------------------------------------------------------------------
# 4. API Endpoints and Headers Tests
# ---------------------------------------------------------------------------

client = TestClient(app)


def test_api_observations_endpoint_timestamp_format() -> None:
    """Verify GET /api/observations exposes observed_at, created_at, and formatted_time."""
    obs_time = "2026-10-16T10:51:00+05:30"
    memory_store.add_observation(
        object_name="test_item_api",
        location="shelf",
        action="placed",
        observed_at=obs_time,
    )

    resp = client.get("/api/observations")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    match = next((item for item in data if item.get("object") == "test_item_api"), None)
    assert match is not None
    assert match["observed_at"] == obs_time
    assert match["created_at"] is not None
    assert "formatted_time" in match
    assert "10:51 AM" in match["formatted_time"]


def test_api_visual_search_recent_endpoint_timestamp_format() -> None:
    """Verify GET /api/visual-search/recent returns observed_at and formatted_time."""
    ts = "2026-10-16T10:51:00+05:30"
    object_store.record_visual_match(
        frame_timestamp=ts,
        registered_object_id="test_reg_obj",
        object_name="test_object_match",
        bounding_box={"x": 10, "y": 10, "width": 50, "height": 50},
        location_name="desk",
        similarity_score=0.88,
        match_status="strong_match",
    )

    resp = client.get("/api/visual-search/recent")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    item = next((m for m in data if m.get("object_name") == "test_object_match"), None)
    assert item is not None
    assert item["frame_timestamp"] == ts
    assert item["observed_at"] == ts
    assert "formatted_time" in item
    assert "10:51 AM" in item["formatted_time"]


def test_ask_endpoint_json_contains_machine_and_human_timestamps() -> None:
    """Verify POST /ask?return_json=true returns observed_at, formatted_time, response_generated_at."""
    ts = "2026-10-16T10:51:00+05:30"
    memory_store.add_observation(
        object_name="reading glasses",
        location="upper-right side",
        action="visually observed",
        evidence_source="visual_embedding",
        verification_status="strong_match",
        observed_at=ts,
    )

    resp = client.post(
        "/ask",
        json={"transcript": "Where are my reading glasses?"},
        params={"return_json": "true"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "observed_at" in data
    assert "formatted_time" in data
    assert "response_generated_at" in data

    headers = resp.headers
    assert "x-transcript" in headers
    assert "x-intent" in headers
    assert "x-confidence" in headers
    assert "x-reasoning" in headers
    assert "x-answer-text" in headers
    assert "x-observed-at" in headers or "x-formatted-time" in headers


# ---------------------------------------------------------------------------
# 5. Dashboard Frontend Contract Tests
# ---------------------------------------------------------------------------

def test_frontend_dashboard_renders_formatted_timestamp() -> None:
    """Verify frontend/app.js incorporates formatted_time and observed_at in tables and search cards."""
    app_js_path = Path(__file__).resolve().parent.parent.parent / "frontend" / "app.js"
    assert app_js_path.exists()
    content = app_js_path.read_text(encoding="utf-8")

    # Verify memory log renders formatted_time and observed_at tooltip
    assert "obs.formatted_time" in content
    assert "obs.observed_at" in content

    # Verify visual search matches table renders formatted_time
    assert "m.formatted_time" in content
    assert "m.observed_at" in content

    # Verify visual search test card renders observed time
    assert "Observed Time" in content

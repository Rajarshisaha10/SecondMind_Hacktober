# ===========================================================================
# [NEW FILE - BACKUP FOR TEAMMATE & ANTIGRAVITY CONTEXT]
# Feature: Feature 2: Accurate Timestamps & Temporal Reasoning
# Relative Path: backend/time_utils.py
# Summary: Reusable deterministic timestamp formatting, timezone handling, relative time calculation, and natural language location/time phrase helpers for MemoryAgent and APIs.
# Integration: Part of SecondMind Image Embeddings & Accurate Timestamps features.
# If your teammate's code fails or lacks this file, drop this directly into place.
# ===========================================================================

"""Deterministic, reusable timestamp formatting and timezone conversion utilities.

Implements human-friendly, deterministic formatting rules for observation timestamps:
- Same day: Today at 10:51 AM
- Previous day: Yesterday at 6:20 PM
- Older observations: Oct 14, 2026 at 10:51 AM (or without year when requested)
- Missing or invalid timestamp: Time unavailable
- Optional relative time: e.g. "Today at 10:51 AM, about 12 minutes ago"

Also provides:
- Timezone handling defaulting to application configuration (Asia/Kolkata)
- Cross-platform 12-hour clock formatting
- Legacy naive vs aware timestamp conversion
- ISO 8601 formatting with explicit timezone offsets
- Spoken preposition and plural/singular grammar helpers
- Controllable clock reference for deterministic testing
"""

from datetime import datetime, timezone, timedelta, tzinfo
from typing import Any, Optional, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.config import settings

TIME_UNAVAILABLE = "Time unavailable"

MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

PLURAL_OBJECTS = {
    "glasses", "reading glasses", "sunglasses", "spectacles", "eyewear",
    "keys", "house keys", "car keys",
    "scissors", "headphones", "earphones", "earbuds",
    "clothes", "shoes", "pants", "slippers", "socks", "boots"
}


def get_configured_timezone() -> ZoneInfo:
    """Return the application's configured ZoneInfo timezone.
    
    Defaults to settings.TIMEZONE or 'Asia/Kolkata'. Falls back to ZoneInfo('Asia/Kolkata')
    if an invalid name is encountered.
    """
    tz_str = getattr(settings, "TIMEZONE", "Asia/Kolkata")
    try:
        return ZoneInfo(tz_str)
    except Exception:
        try:
            return ZoneInfo("Asia/Kolkata")
        except Exception:
            return timezone.utc  # type: ignore[return-value]


def parse_timestamp(
    ts: Any,
    default_tz: Optional[tzinfo] = None,
) -> Optional[datetime]:
    """Parse a timestamp into a timezone-aware datetime in default_tz (or configured timezone).
    
    Handles:
    - ISO 8601 strings with timezone offset (+05:30, +00:00, Z)
    - Naive ISO strings (assumed to be in default_tz / configured timezone)
    - datetime instances (aware converted, naive localized)
    - Unix numeric timestamps (seconds since epoch)
    - None, empty, or malformed values (returns None safely without crashing)
    """
    if ts is None:
        return None
    
    target_tz = default_tz or get_configured_timezone()
    
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts, tz=target_tz)
        except Exception:
            return None

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=target_tz)
        return ts.astimezone(target_tz)

    if not isinstance(ts, str):
        return None

    cleaned = ts.strip()
    if not cleaned or cleaned.lower() in ("none", "null", "time unavailable"):
        return None

    # Replace trailing 'Z' or 'z' with UTC offset
    if cleaned.endswith("Z") or cleaned.endswith("z"):
        cleaned = cleaned[:-1] + "+00:00"

    # Attempt ISO format parsing
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=target_tz)
        return dt.astimezone(target_tz)
    except ValueError:
        pass

    # Fallback to common date/time formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.replace(tzinfo=target_tz)
        except ValueError:
            continue

    return None


def format_time_12h(dt: datetime) -> str:
    """Format the time component in cross-platform 12-hour format without leading zero.
    
    Examples:
    - 10:51 -> '10:51 AM'
    - 18:20 -> '6:20 PM'
    - 00:05 -> '12:05 AM'
    - 12:00 -> '12:00 PM'
    """
    hour_12 = dt.hour % 12 or 12
    minute = f"{dt.minute:02d}"
    meridiem = "AM" if dt.hour < 12 else "PM"
    return f"{hour_12}:{minute} {meridiem}"


def format_relative_time(dt: datetime, now: datetime) -> str:
    """Format relative time difference between an observation and a reference time.
    
    Examples:
    - 'less than a minute ago'
    - 'about 12 minutes ago'
    - 'about 2 hours ago'
    - 'about 3 days ago'
    """
    total_seconds = (now - dt).total_seconds()
    if total_seconds < 0:
        return "just now"
    if total_seconds < 60:
        return "less than a minute ago"
    
    total_minutes = int(round(total_seconds / 60.0))
    if total_minutes == 1:
        return "about 1 minute ago"
    if total_minutes < 60:
        return f"about {total_minutes} minutes ago"
    
    total_hours = int(round(total_seconds / 3600.0))
    if total_hours == 1:
        return "about 1 hour ago"
    if total_hours < 24:
        return f"about {total_hours} hours ago"
    
    total_days = int(round(total_seconds / 86400.0))
    if total_days == 1:
        return "about 1 day ago"
    return f"about {total_days} days ago"


def format_observation_timestamp(
    ts: Any,
    now: Optional[datetime] = None,
    tz: Optional[tzinfo] = None,
    include_year: bool = True,
    include_relative: bool = False,
) -> str:
    """Deterministically format an observation timestamp for humans.
    
    Display rules:
    - Same day: 'Today at 10:51 AM'
    - Previous day: 'Yesterday at 6:20 PM'
    - Older observations: 'Oct 14, 2026 at 10:51 AM' (or 'Oct 14 at 10:51 AM' if include_year=False)
    - Missing or invalid timestamp: 'Time unavailable'
    
    Args:
        ts: Any timestamp representation (ISO string, datetime, epoch, None).
        now: Optional reference current time for controllable clock in tests.
        tz: Optional timezone override (defaults to configured application timezone).
        include_year: Whether to include the year in older date formats.
        include_relative: Optionally append relative time e.g. ', about 12 minutes ago'.
        
    Returns:
        Formatted human-readable string.
    """
    target_tz = tz or get_configured_timezone()
    dt = parse_timestamp(ts, default_tz=target_tz)
    if dt is None:
        return TIME_UNAVAILABLE

    # Ensure reference now is timezone-aware in target_tz
    if now is not None:
        if now.tzinfo is None:
            ref_now = now.replace(tzinfo=target_tz)
        else:
            ref_now = now.astimezone(target_tz)
    else:
        ref_now = datetime.now(target_tz)

    time_str = format_time_12h(dt)
    obs_date = dt.date()
    ref_date = ref_now.date()
    delta_days = (ref_date - obs_date).days

    if obs_date == ref_date:
        base = f"Today at {time_str}"
    elif delta_days == 1:
        base = f"Yesterday at {time_str}"
    elif delta_days < 0:
        # Slightly in the future or clock skew
        month_abbr = MONTH_NAMES[dt.month]
        if include_year:
            base = f"{month_abbr} {dt.day}, {dt.year} at {time_str}"
        else:
            base = f"{month_abbr} {dt.day} at {time_str}"
    else:
        # 2 or more days ago
        month_abbr = MONTH_NAMES[dt.month]
        if include_year:
            base = f"{month_abbr} {dt.day}, {dt.year} at {time_str}"
        else:
            base = f"{month_abbr} {dt.day} at {time_str}"

    if include_relative:
        rel = format_relative_time(dt, ref_now)
        return f"{base}, {rel}"

    return base


def format_observation_time_phrase(formatted_time: str) -> str:
    """Format a display timestamp into a spoken sentence phrase.
    
    Examples:
    - 'Today at 10:51 AM' -> 'today at 10:51 AM'
    - 'Yesterday at 6:20 PM' -> 'yesterday at 6:20 PM'
    - 'Oct 14, 2026 at 10:51 AM' -> 'on Oct 14, 2026 at 10:51 AM'
    - 'Oct 14 at 6:20 PM' -> 'on Oct 14 at 6:20 PM'
    - 'Time unavailable' -> 'at an unknown time'
    """
    if not formatted_time or formatted_time == TIME_UNAVAILABLE:
        return "at an unknown time"
    
    if formatted_time.startswith("Today"):
        return "today" + formatted_time[5:]
    if formatted_time.startswith("Yesterday"):
        return "yesterday" + formatted_time[9:]
    
    return f"on {formatted_time}"


def format_location_phrase(location: str) -> str:
    """Format room location naturally with appropriate preposition.
    
    Examples:
    - 'upper-right side' -> 'on the upper-right side'
    - 'kitchen counter' -> 'on the kitchen counter'
    - 'near the entrance' -> 'near the entrance'
    - 'on the desk' -> 'on the desk'
    """
    loc = location.strip()
    if " (bounding box" in loc:
        loc = loc.split(" (bounding box")[0].strip()

    lower = loc.lower()
    for prep in ["on the ", "in the ", "at the ", "near the ", "under the "]:
        if lower.startswith(prep):
            return loc
    for prep in ["on ", "in ", "at ", "near ", "under "]:
        if lower.startswith(prep):
            return loc

    return f"on the {loc}"


def is_plural_object(object_name: str) -> bool:
    """Return True if the object name is plural (e.g. glasses, keys, scissors).
    
    Used to select grammatical verb ('was' vs 'were') and pronouns ('it' vs 'them', 'its' vs 'their').
    """
    name = object_name.strip().lower()
    if name in PLURAL_OBJECTS:
        return True
    words = name.split()
    if words and words[-1] in PLURAL_OBJECTS:
        return True
    if name.endswith("s") and not name.endswith(("ss", "us", "is")):
        return True
    return False


def to_iso_with_tz(
    ts: Any,
    tz: Optional[tzinfo] = None,
) -> Optional[str]:
    """Convert a timestamp to an ISO 8601 string with explicit timezone offset.
    
    Example: '2026-10-16T11:17:21+05:30'
    Returns None if ts is invalid or unparseable.
    """
    target_tz = tz or get_configured_timezone()
    dt = parse_timestamp(ts, default_tz=target_tz)
    if dt is None:
        return None
    return dt.isoformat()

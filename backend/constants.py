"""Constants and enumeration definitions for Second Mind backend.

Defines all intent categories, confidence levels, event types, and reminder types
used across agents, services, and API endpoints.
"""

from enum import Enum


class Intent(str, Enum):
    """Categorized user query intent categories."""

    MEMORY = "MEMORY"
    MEDICATION = "MEDICATION"
    SCHEDULE = "SCHEDULE"
    PROFILE = "PROFILE"
    EMERGENCY = "EMERGENCY"
    GENERAL = "GENERAL"


class Confidence(str, Enum):
    """Deterministically computed confidence levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EventType(str, Enum):
    """Observation log event classifications."""

    OBJECT_OBSERVATION = "object_observation"
    MEDICATION_TAKEN = "medication_taken"
    HELP_REQUEST = "help_request"


class ReminderType(str, Enum):
    """Proactive reminder categories."""

    MEDICATION = "medication"
    APPOINTMENT = "appointment"
    ROUTINE = "routine"
    CUSTOM = "custom"

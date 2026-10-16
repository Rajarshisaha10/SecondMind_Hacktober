# ===========================================================================
# [CHANGES BACKUP - INCLUDES FEATURE 1, 2, AND FEATURE 3: REALITY CHECKSUM]
# Target File: backend/schemas.py
# ===========================================================================

"""Pydantic DTOs and internal dataclass contracts for Second Mind backend.

Defines the core data contracts exchanged between layers and API endpoints.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from backend.constants import Confidence, EventType, Intent, ReminderType


@dataclass
class AgentResponse:
    """Standardized response object returned by all specialized agents."""

    answer_text: str
    confidence: Confidence
    reasoning: str
    observed_at: Optional[str] = None
    formatted_time: Optional[str] = None
    response_generated_at: Optional[str] = None


class ObservationCreate(BaseModel):
    """Schema for creating a new memory store observation entry."""

    object: str = Field(..., description="Name of the observed object or entity")
    location: str = Field(..., description="Location where object was observed")
    action: str = Field(..., description="Action performed, e.g. placed, picked up")
    event_type: EventType = Field(
        default=EventType.OBJECT_OBSERVATION,
        description="Type of observation event",
    )
    raw_description: Optional[str] = Field(
        default="", description="Full raw description from vision agent"
    )
    observed_at: Optional[str] = Field(default=None, description="ISO timestamp of observation capture")


class AskResponse(BaseModel):
    """Debug/JSON response schema for the /ask endpoint."""

    transcript: str = Field(..., description="Transcribed user question")
    intent: Intent = Field(..., description="Classified intent category")
    answer_text: str = Field(..., description="Spoken natural language response")
    confidence: Confidence = Field(..., description="Computed confidence level")
    reasoning: str = Field(..., description="One-line natural language reasoning trace")
    audio_url: Optional[str] = Field(default=None, description="URL or path to synthesized audio file")
    observed_at: Optional[str] = Field(default=None, description="ISO 8601 observation timestamp with explicit timezone")
    formatted_time: Optional[str] = Field(default=None, description="Human-readable formatted observation timestamp")
    response_generated_at: Optional[str] = Field(default=None, description="ISO 8601 timestamp when response was generated")


class ReminderCheckResponse(BaseModel):
    """Response schema for the GET /reminder/check polling endpoint."""

    pending: bool = Field(..., description="True if a proactive reminder is waiting")
    audio_url: Optional[str] = Field(default=None, description="URL to synthesized reminder WAV audio")
    message: Optional[str] = Field(default=None, description="Text message of the reminder")
    reminder_type: Optional[ReminderType] = Field(default=None, description="Category of reminder")


class ReminderAckResponse(BaseModel):
    """Response schema for POST /reminder/ack endpoint."""

    status: str = Field(default="acknowledged", description="Acknowledgement status")
    cleared: bool = Field(default=True, description="True if pending reminder slot was cleared")


class HealthResponse(BaseModel):
    """Detailed health and readiness status response for GET /health."""

    status: str = Field(..., description="Overall liveness status ('healthy' or 'unhealthy')")
    ollama: str = Field(..., description="Ollama local LLM connectivity status")
    database: str = Field(..., description="SQLite database connectivity status")
    whisper: str = Field(..., description="Whisper STT status")
    tts: str = Field(..., description="TTS engine status")
    agents: int = Field(..., description="Total number of active registered agents")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")


class ObjectRegisterRequest(BaseModel):
    """JSON payload schema for registering an object."""

    name: str = Field(..., min_length=1, description="Name of the physical object")
    category: str = Field(default="personal", description="Category of the object")
    description: str = Field(default="", description="Optional caregiver description")
    image_base64: Optional[str] = Field(default=None, description="Optional base64 encoded reference image")


class VisualSearchRequest(BaseModel):
    """JSON payload schema for requesting visual search on an image frame."""

    image_base64: Optional[str] = Field(default=None, description="Base64 encoded BGR/RGB image frame")
    target_object_id: Optional[str] = Field(default=None, description="Optional filter to search specific object")
    save_observation: bool = Field(default=True, description="Save confident detection into memory observations")
    frame_timestamp: Optional[str] = Field(default=None, description="ISO timestamp of frame capture")
    observed_at: Optional[str] = Field(default=None, description="Alias for frame_timestamp")


class RealityConflictVerifyRequest(BaseModel):
    """Payload schema for verifying or resolving a reality discrepancy."""

    conflict_id: str = Field(..., description="UUID of the conflict to resolve")
    verified_location: str = Field(..., description="Confirmed true location of the object")
    verified_by: str = Field(default="caregiver", description="Name or role of the verifier")



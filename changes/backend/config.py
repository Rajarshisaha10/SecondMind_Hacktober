"""Configuration module for Second Mind Track A backend.

# ===========================================================================
# [CHANGES BACKUP - FEATURE 1: IMAGE EMBEDDINGS & FEATURE 2: ACCURATE TIMESTAMPS]
# Target File: backend/config.py
# ---------------------------------------------------------------------------
# SUMMARY OF CHANGES:
# 1. Added TIMEZONE configuration (Feature 2: Accurate Timestamps).
#    - TIMEZONE: str = "Asia/Kolkata" (default timezone for human formatting).
# 2. Added Visual Embeddings & Object Retrieval configurations (Feature 1):
#    - VISUAL_ENCODER_MODEL: "mobilenet_v3_small"
#    - VISUAL_DEVICE: "auto"
#    - SIMILARITY_STRONG_THRESHOLD: 0.72
#    - SIMILARITY_WEAK_THRESHOLD: 0.45
#    - AMBIGUITY_MARGIN: 0.06
#    - MAX_IMAGE_UPLOAD_BYTES: 10 MB
#    - REFERENCE_IMAGE_DIR: data/reference_images
#    - ENABLE_LOCAL_FEATURE_VERIFICATION: True
#    - LOCAL_FEATURE_MIN_MATCHES: 4
#
# Original code (commit 584d7b5) had neither of these sections.
# ===========================================================================
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with type validation and environment overrides."""

    # LLM configuration (Original)
    OLLAMA_MODEL: str = "gemma3:4b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT_SECONDS: float = 30.0

    # Storage paths (Original)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATABASE_PATH: Path = BASE_DIR / "second_mind.db"
    PROFILE_PATH: Path = BASE_DIR / "profile_setup.json"
    SCHEDULE_PATH: Path = BASE_DIR / "schedule_setup.json"
    PROMPTS_DIR: Path = BASE_DIR / "backend" / "prompts"
    AUDIO_OUTPUT_DIR: Path = BASE_DIR / "audio_cache"

    # Session configuration (Original)
    SESSION_TTL_MINUTES: int = 10
    DEFAULT_DEVICE_ID: str = "default_device"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # [NEW - Feature 2: Accurate Timestamps]
    # Added TIMEZONE setting so all observation times are converted and
    # formatted in a deterministic timezone (defaults to Asia/Kolkata).
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    TIMEZONE: str = "Asia/Kolkata"

    # Speech configuration (Original)
    STT_MODEL_SIZE: str = "base"
    STT_DEVICE: str = "cpu"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # [NEW - Feature 1: Visual Embeddings & Object Retrieval]
    # Added configuration for local deep feature extractor, cosine similarity
    # thresholds, spatial proposal extraction, and SIFT/ORB keypoint verification.
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    VISUAL_ENCODER_MODEL: str = "mobilenet_v3_small"
    VISUAL_DEVICE: str = "auto"
    SIMILARITY_STRONG_THRESHOLD: float = 0.72
    SIMILARITY_WEAK_THRESHOLD: float = 0.45
    AMBIGUITY_MARGIN: float = 0.06
    MAX_IMAGE_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB limit
    REFERENCE_IMAGE_DIR: Path = BASE_DIR / "data" / "reference_images"
    ENABLE_LOCAL_FEATURE_VERIFICATION: bool = True
    LOCAL_FEATURE_MIN_MATCHES: int = 4

    model_config = SettingsConfigDict(
        env_prefix="SECOND_MIND_",
        env_file=".env",
        extra="ignore"
    )


settings = Settings()

"""Configuration module for Second Mind Track A backend.

Provides centralized configuration variables loaded from environment variables
or sensible defaults. Modifying the target LLM model or storage locations only
requires updating values here.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with type validation and environment overrides."""

    # LLM configuration
    OLLAMA_MODEL: str = "gemma3:4b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT_SECONDS: float = 30.0

    # Storage paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATABASE_PATH: Path = BASE_DIR / "second_mind.db"
    PROFILE_PATH: Path = BASE_DIR / "profile_setup.json"
    SCHEDULE_PATH: Path = BASE_DIR / "schedule_setup.json"
    PROMPTS_DIR: Path = BASE_DIR / "backend" / "prompts"
    AUDIO_OUTPUT_DIR: Path = BASE_DIR / "audio_cache"

    # Session configuration
    SESSION_TTL_MINUTES: int = 10
    DEFAULT_DEVICE_ID: str = "default_device"

    # Speech configuration
    STT_MODEL_SIZE: str = "base"
    STT_DEVICE: str = "cpu"

    model_config = SettingsConfigDict(
        env_prefix="SECOND_MIND_",
        env_file=".env",
        extra="ignore"
    )


settings = Settings()

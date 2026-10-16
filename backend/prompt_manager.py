"""Prompt Manager module loading and formatting prompt templates.

Centralizes prompt loading from backend/prompts/ directory and variable substitution.
"""

from pathlib import Path
from typing import Dict, Optional
from backend.config import settings
from backend.logger import logger


class PromptManager:
    """Manager class for reading and populating prompt templates."""

    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        """Initialize PromptManager with prompts directory path.

        Args:
            prompts_dir: Optional custom path to prompts directory.
        """
        self.prompts_dir = prompts_dir or settings.PROMPTS_DIR
        self._cache: Dict[str, str] = {}

    def load_template(self, template_name: str) -> str:
        """Load prompt template text from file.

        Args:
            template_name: Name of template file (e.g. 'orchestrator.txt' or 'orchestrator').

        Returns:
            Raw template string.
        """
        if not template_name.endswith(".txt"):
            template_name = f"{template_name}.txt"

        if template_name in self._cache:
            return self._cache[template_name]

        file_path = self.prompts_dir / template_name
        if not file_path.exists():
            logger.error("PromptManager: Template file not found: '%s'", file_path)
            raise FileNotFoundError(f"Prompt template file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            template_text = f.read()

        self._cache[template_name] = template_text
        return template_text

    def get_prompt(self, template_name: str, **kwargs) -> str:
        """Get formatted prompt string with kwargs substituted.

        Args:
            template_name: Template filename or stem name.
            **kwargs: Substitution key-value pairs.

        Returns:
            Formatted prompt string.
        """
        template_text = self.load_template(template_name)
        try:
            return template_text.format(**kwargs)
        except KeyError as e:
            logger.warning("PromptManager: Missing variable in template '%s': %s", template_name, e)
            # Safe fallback if a key is missing
            return template_text

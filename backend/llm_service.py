"""High-level LLM Service module consumed by agents.

Wraps low-level OllamaClient, providing safe generation methods and JSON parsing.
"""

import json
from typing import Any, Dict, Optional
from backend.ollama_client import OllamaClient
from backend.logger import logger


class LLMService:
    """High-level service interface for LLM completions."""

    def __init__(self, client: Optional[OllamaClient] = None) -> None:
        """Initialize LLMService with low-level OllamaClient.

        Args:
            client: Optional pre-configured OllamaClient instance.
        """
        self.client = client or OllamaClient()

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        response_format: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """Generate LLM response text.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt context.
            response_format: Optional 'json' format specifier.
            temperature: Sampling temperature.

        Returns:
            Raw response text from LLM.
        """
        text = self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=response_format,
            temperature=temperature,
        )
        if text:
            return (
                text.replace("’", "'")
                .replace("‘", "'")
                .replace("“", '"')
                .replace("”", '"')
            )
        return text

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Generate and parse JSON output from LLM safely.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt context.
            temperature: Sampling temperature.

        Returns:
            Parsed dictionary, or empty dict if JSON parsing fails.
        """
        raw_output = self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format="json",
            temperature=temperature,
        )
        if not raw_output:
            return {}

        try:
            return json.loads(raw_output)
        except Exception as e:
            logger.warning("LLMService: Failed to parse JSON output: %s | Raw: '%s'", e, raw_output)
            # Basic fallback attempt if wrapped in markdown code blocks
            clean = raw_output.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            try:
                return json.loads(clean.strip())
            except Exception:
                return {}

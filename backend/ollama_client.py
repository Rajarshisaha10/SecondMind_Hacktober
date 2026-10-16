"""Low-level Ollama API HTTP client wrapper.

Handles raw HTTP interaction with local Ollama server instance for completion and JSON generation.
"""

import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional
from backend.config import settings
from backend.logger import logger


class OllamaClient:
    """Low-level HTTP client wrapping local Ollama instance."""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_MODEL,
        timeout: float = settings.OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize Ollama client.

        Args:
            base_url: Base URL of local Ollama server.
            model: Default model name (defaults to gemma3:4b).
            timeout: HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        response_format: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """Send prompt to local Ollama instance and return raw response string.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt context.
            response_format: Optional 'json' response format string.
            temperature: Inference sampling temperature.

        Returns:
            Generated response string.
        """
        url = f"{self.base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        if system_prompt:
            payload["system"] = system_prompt

        if response_format == "json":
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    resp_body = response.read().decode("utf-8")
                    result = json.loads(resp_body)
                    response_text = result.get("response", "").strip()
                    logger.debug("OllamaClient: Call succeeded. Response len: %d", len(response_text))
                    return response_text
                else:
                    logger.error("OllamaClient: HTTP Error %d from Ollama", response.status)
        except Exception as e:
            logger.error("OllamaClient: Exception during Ollama call: %s", e)

        # Retrying once if JSON failed
        if response_format == "json":
            logger.warning("OllamaClient: Retrying JSON generation with explicit suffix...")
            retry_prompt = prompt + "\nCRITICAL: Respond ONLY with valid raw JSON."
            payload["prompt"] = retry_prompt
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    if response.status == 200:
                        resp_body = response.read().decode("utf-8")
                        return json.loads(resp_body).get("response", "").strip()
            except Exception as ex:
                logger.error("OllamaClient: Retry failed: %s", ex)

        return ""

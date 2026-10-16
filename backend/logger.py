"""Structured logging module for Second Mind.

Provides structured formatting to record request lifecycles, intent routing,
agent execution, latency, and confidence calculations for demo auditing.
"""

import sys
import time
import logging
from typing import Any, Optional
from backend.constants import Confidence, Intent


def setup_logger(name: str = "second_mind") -> logging.Logger:
    """Configure and return a standard logger instance.

    Args:
        name: Name of the logger instance.

    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def log_request_lifecycle(
    question: str,
    resolved_question: str,
    intent: Intent,
    agent_name: str,
    confidence: Confidence,
    reasoning: str,
    answer_text: str,
    latency_ms: float,
    device_id: str = "default_device",
) -> None:
    """Log a complete user request lifecycle trace.

    Args:
        question: Original raw transcript / question from user.
        resolved_question: Standalone resolved question after session context pass.
        intent: Classified intent enum.
        agent_name: Name of the agent handling the request.
        confidence: Deterministically computed confidence level.
        reasoning: One-line natural language explanation of confidence.
        answer_text: Final spoken answer generated.
        latency_ms: Execution duration in milliseconds.
        device_id: Identifier of the originating device.
    """
    logger.info(
        "REQUEST_TRACE | Device: %s | Question: '%s' | Resolved: '%s' | "
        "Intent: %s | Agent: %s | Confidence: %s | Latency: %.2fms | "
        "Reasoning: '%s' | Answer: '%s'",
        device_id,
        question,
        resolved_question,
        intent.value if isinstance(intent, Intent) else intent,
        agent_name,
        confidence.value if isinstance(confidence, Confidence) else confidence,
        latency_ms,
        reasoning,
        answer_text,
    )

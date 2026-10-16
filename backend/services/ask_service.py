"""Ask Service module orchestrating the active question-answer voice loop.

Connects STT, OrchestratorAgent, target agents, TTS, and structured logging.
"""

import time
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
from backend.config import settings
from backend.constants import Confidence, Intent
from backend.logger import log_request_lifecycle, logger
from backend.schemas import AgentResponse, AskResponse
from backend.stt import STTWrapper
from backend.time_utils import get_configured_timezone, to_iso_with_tz
from backend.tts import TTSWrapper
from backend.agents.orchestrator import OrchestratorAgent


class AskService:
    """Service handling active user voice or text queries."""

    def __init__(
        self,
        orchestrator: OrchestratorAgent,
        stt: Optional[STTWrapper] = None,
        tts: Optional[TTSWrapper] = None,
    ) -> None:
        """Initialize AskService with injected dependencies.

        Args:
            orchestrator: Orchestrator agent.
            stt: Optional STT wrapper instance.
            tts: Optional TTS wrapper instance.
        """
        self.orchestrator = orchestrator
        self.stt = stt or STTWrapper()
        self.tts = tts or TTSWrapper()

    def process_ask_text(
        self, transcript: str, device_id: str = settings.DEFAULT_DEVICE_ID
    ) -> Tuple[AskResponse, str]:
        """Process text transcript question end-to-end.

        Args:
            transcript: Transcribed user question string.
            device_id: Unique device identifier.

        Returns:
            Tuple of (AskResponse object, path to generated WAV audio file).
        """
        start_time = time.time()

        # Dispatch via orchestrator
        agent_response, resolved_transcript, intent = self.orchestrator.dispatch(
            transcript=transcript, device_id=device_id
        )

        # Synthesize audio response
        audio_file_path = self.tts.synthesize_to_file(agent_response.answer_text)

        latency_ms = (time.time() - start_time) * 1000.0

        # Structured lifecycle logging
        log_request_lifecycle(
            question=transcript,
            resolved_question=resolved_transcript,
            intent=intent,
            agent_name="Orchestrator",
            confidence=agent_response.confidence,
            reasoning=agent_response.reasoning,
            answer_text=agent_response.answer_text,
            latency_ms=latency_ms,
            device_id=device_id,
        )

        ask_response = AskResponse(
            transcript=transcript,
            intent=intent,
            answer_text=agent_response.answer_text,
            confidence=agent_response.confidence,
            reasoning=agent_response.reasoning,
            audio_url=f"/audio/{Path(audio_file_path).name}",
            observed_at=agent_response.observed_at,
            formatted_time=agent_response.formatted_time,
            response_generated_at=agent_response.response_generated_at or to_iso_with_tz(datetime.now(get_configured_timezone())),
        )

        return ask_response, audio_file_path

    def process_ask_audio(
        self, audio_bytes: bytes, device_id: str = settings.DEFAULT_DEVICE_ID
    ) -> Tuple[AskResponse, str]:
        """Process raw WAV audio bytes question end-to-end.

        Args:
            audio_bytes: Incoming audio bytes.
            device_id: Unique device identifier.

        Returns:
            Tuple of (AskResponse object, path to generated WAV audio file).
        """
        # Speech-to-Text
        transcript = self.stt.transcribe_bytes(audio_bytes)
        if not transcript:
            logger.warning("AskService: STT returned empty transcript for audio request")
            answer_msg = "I couldn't hear that clearly. Please hold Button A and speak into the device."
            audio_file_path = self.tts.synthesize_to_file(answer_msg)
            ask_response = AskResponse(
                transcript="",
                intent=Intent.GENERAL,
                answer_text=answer_msg,
                confidence=Confidence.LOW,
                reasoning="Unrecognized or soft audio recording.",
                audio_url=f"/audio/{Path(audio_file_path).name}",
                response_generated_at=to_iso_with_tz(datetime.now(get_configured_timezone())),
            )
            return ask_response, audio_file_path

        return self.process_ask_text(transcript=transcript, device_id=device_id)

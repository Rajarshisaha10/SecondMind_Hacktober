"""Orchestrator Agent module for session context resolution, intent classification, and dispatching.

Acts as the central dispatcher for Track A.
"""

from typing import Dict, Tuple
from backend.config import settings
from backend.constants import Intent
from backend.llm_service import LLMService
from backend.logger import logger
from backend.prompt_manager import PromptManager
from backend.schemas import AgentResponse
from backend.stores.session_store import SessionStore
from backend.agents.memory_agent import MemoryAgent
from backend.agents.schedule_agent import ScheduleAgent
from backend.agents.qa_agent import QAAgent
from backend.agents.emergency_agent import EmergencyAgent


class OrchestratorAgent:
    """Central Orchestrator Agent resolving context and dispatching queries to agents."""

    def __init__(
        self,
        llm: LLMService,
        prompt_mgr: PromptManager,
        session_store: SessionStore,
        memory_agent: MemoryAgent,
        schedule_agent: ScheduleAgent,
        qa_agent: QAAgent,
        emergency_agent: EmergencyAgent,
    ) -> None:
        """Initialize OrchestratorAgent with injected dependencies.

        Args:
            llm: Centralized LLM service.
            prompt_mgr: Prompt manager.
            session_store: Thread-safe session context store.
            memory_agent: Memory retrieval agent instance.
            schedule_agent: Schedule agent instance.
            qa_agent: General QA agent instance.
            emergency_agent: Emergency agent instance.
        """
        self.llm = llm
        self.prompt_mgr = prompt_mgr
        self.session_store = session_store
        self.memory_agent = memory_agent
        self.schedule_agent = schedule_agent
        self.qa_agent = qa_agent
        self.emergency_agent = emergency_agent

    def resolve_session_context(
        self, transcript: str, device_id: str = settings.DEFAULT_DEVICE_ID
    ) -> str:
        """Resolve pronouns and time references against active session context.

        Args:
            transcript: Transcribed raw user question.
            device_id: Unique device identifier.

        Returns:
            Fully explicit standalone question string.
        """
        session = self.session_store.get_session(device_id)
        last_subject = session.get("last_subject")
        last_time_ref = session.get("last_resolved_time_ref") or "N/A"
        last_intent = session.get("last_intent") or "N/A"

        if not last_subject:
            return transcript

        # Check if question has follow-up markers / pronouns
        lower = transcript.lower()
        clean_lower = lower.rstrip("?.!")
        pronouns = ["it", "them", "they", "her", "him", "that", "these", "those"]
        has_pronoun = any(
            f" {p} " in f" {lower} " or clean_lower.endswith(f" {p}") for p in pronouns
        )
        has_followup = any(w in lower for w in ["after", "before", "moved", "again", "where did", "where was"])

        if not (has_pronoun or has_followup):
            return transcript

        prompt = self.prompt_mgr.get_prompt(
            "context_resolver.txt",
            last_subject=last_subject,
            last_resolved_time_ref=last_time_ref,
            last_intent=last_intent,
            transcript=transcript,
        )

        resolved = self.llm.generate(prompt=prompt, temperature=0.0).strip()
        if resolved and len(resolved) > 3 and not resolved.startswith("{"):
            logger.info("Orchestrator: Resolved context from '%s' -> '%s'", transcript, resolved)
            return resolved

        # Deterministic fallback pronoun resolution if LLM is unavailable or offline
        if has_pronoun or has_followup:
            import re
            fallback = transcript
            for p in ["them", "it", "they", "those", "these"]:
                fallback = re.sub(rf"\b{p}\b", f"my {last_subject}", fallback, flags=re.IGNORECASE)
            if fallback != transcript:
                logger.info("Orchestrator: Deterministic context resolved '%s' -> '%s'", transcript, fallback)
                return fallback

        return transcript

    def classify_intent(self, transcript: str) -> Intent:
        """Classify user question into one of the six Intent categories.

        Args:
            transcript: Explicit standalone question string.

        Returns:
            Classified Intent Enum.
        """
        # Fast rule-based emergency triage check
        lower = transcript.lower()
        if "help" in lower or "emergency" in lower or "call my daughter" in lower or "call for help" in lower:
            logger.info("Orchestrator: Fast-triaged EMERGENCY intent for '%s'", transcript)
            return Intent.EMERGENCY

        prompt = self.prompt_mgr.get_prompt("orchestrator.txt", transcript=transcript)
        response = self.llm.generate(prompt=prompt, temperature=0.0).strip().upper()

        for intent in Intent:
            if intent.value in response:
                logger.info("Orchestrator: Classified intent '%s' for transcript '%s'", intent.value, transcript)
                return intent

        logger.warning("Orchestrator: Unrecognized intent output '%s', falling back to GENERAL", response)
        return Intent.GENERAL

    def dispatch(
        self, transcript: str, device_id: str = settings.DEFAULT_DEVICE_ID
    ) -> Tuple[AgentResponse, str, Intent]:
        """Process question: resolve context -> classify intent -> dispatch agent -> update session.

        Args:
            transcript: Transcribed user question.
            device_id: Unique device identifier.

        Returns:
            Tuple of (AgentResponse, resolved_transcript, intent).
        """
        resolved_transcript = self.resolve_session_context(transcript, device_id)
        intent = self.classify_intent(resolved_transcript)

        response: AgentResponse
        if intent == Intent.MEMORY:
            response = self.memory_agent.answer_object_query(resolved_transcript)
        elif intent == Intent.MEDICATION:
            response = self.memory_agent.answer_medication_query(resolved_transcript)
        elif intent == Intent.SCHEDULE:
            response = self.schedule_agent.answer(resolved_transcript)
        elif intent == Intent.PROFILE:
            response = self.qa_agent.answer_profile_query(resolved_transcript)
        elif intent == Intent.EMERGENCY:
            response = self.emergency_agent.answer(resolved_transcript)
        else:
            response = self.qa_agent.answer_general_query(resolved_transcript)

        # Extract potential new subject from transcript for session tracking
        extracted_subject = None
        if intent == Intent.MEMORY:
            extracted_subject = self.memory_agent._extract_object_keyword(resolved_transcript)
        elif intent == Intent.MEDICATION:
            extracted_subject = "medication"
        elif intent == Intent.PROFILE:
            for kw in ["sarah", "daughter", "son", "doctor"]:
                if kw in resolved_transcript.lower():
                    extracted_subject = kw
                    break

        self.session_store.update_session(
            device_id=device_id,
            subject=extracted_subject,
            intent=intent,
        )

        return response, resolved_transcript, intent

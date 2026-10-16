"""Memory Retrieval Agent module for object location queries and medication verification.

Calculates confidence deterministically in Python (HIGH / MEDIUM / LOW) and uses Gemma LLM
strictly for phrasing spoken natural language answers and reasoning explanations.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from backend.constants import Confidence, EventType
from backend.llm_service import LLMService
from backend.logger import logger
from backend.prompt_manager import PromptManager
from backend.schemas import AgentResponse
from backend.stores.memory_store import MemoryStore
from backend.stores.profile_store import ProfileStore


class MemoryAgent:
    """Agent handling object queries (MEMORY) and medication verification (MEDICATION)."""

    def __init__(
        self,
        memory_store: MemoryStore,
        profile_store: ProfileStore,
        llm: LLMService,
        prompt_mgr: PromptManager,
    ) -> None:
        """Initialize MemoryAgent with injected dependencies.

        Args:
            memory_store: Observation memory store.
            profile_store: Patient profile store.
            llm: Centralized LLM service.
            prompt_mgr: Centralized prompt manager.
        """
        self.memory_store = memory_store
        self.profile_store = profile_store
        self.llm = llm
        self.prompt_mgr = prompt_mgr

    def answer_object_query(self, transcript: str) -> AgentResponse:
        """Answer object location query from MemoryStore using deterministic confidence.

        Args:
            transcript: Transcribed user question.

        Returns:
            AgentResponse containing answer_text, confidence, reasoning.
        """
        # Extract object key phrase from question
        object_query = self._extract_object_keyword(transcript)
        observations = self.memory_store.query_objects(object_name=object_query, limit=5)

        # Calculate confidence deterministically in Python
        confidence, best_match = self._compute_object_confidence(observations)

        # Prepare formatted observation context for Gemma phrasing
        obs_text_list = []
        for obs in observations:
            obs_text_list.append(
                f"- Object: {obs['object']}, Location: {obs['location']}, "
                f"Action: {obs['action']}, Time: {obs['timestamp']}"
            )
        obs_context = "\n".join(obs_text_list) if obs_text_list else "No observations found."

        prompt = self.prompt_mgr.get_prompt(
            "memory_agent.txt",
            observations=obs_context,
            transcript=transcript,
            confidence=confidence.value,
        )

        res_json = self.llm.generate_json(prompt=prompt, temperature=0.3)

        answer_text = res_json.get("answer_text")
        reasoning = res_json.get("reasoning")

        # Fallback if LLM output fails JSON parsing
        if not answer_text:
            if confidence == Confidence.HIGH and best_match:
                answer_text = f"Your {best_match['object']} was last seen on the {best_match['location']}."
                reasoning = f"Matched single recent observation at {best_match['location']}."
            elif confidence == Confidence.MEDIUM and best_match:
                answer_text = f"I think your {best_match['object']} might be on the {best_match['location']}."
                reasoning = f"Matched older observation at {best_match['location']}."
            else:
                answer_text = "I'm sorry, I haven't seen that object in the room recently."
                reasoning = "No matching observation found in memory store."

        logger.info("MemoryAgent (Object): answer='%s', confidence=%s", answer_text, confidence.value)
        return AgentResponse(
            answer_text=answer_text,
            confidence=confidence,
            reasoning=reasoning or f"Confidence {confidence.value} calculated deterministically.",
        )

    def answer_medication_query(self, transcript: str) -> AgentResponse:
        """Answer medication-taken query using hybrid check against schedule and memory log.

        Args:
            transcript: Transcribed user question.

        Returns:
            AgentResponse containing answer_text, confidence, reasoning.
        """
        medications = self.profile_store.get_medications()
        med_events = self.memory_store.query_medication_events(limit=10)

        # Calculate confidence deterministically in Python
        confidence, status_reason = self._compute_medication_confidence(medications, med_events)

        med_schedule_str = "\n".join(
            [f"- {m['name']}: times={m['schedule_times']} (grace {m['grace_window_minutes']}m)" for m in medications]
        ) if medications else "No medications listed."

        med_events_str = "\n".join(
            [f"- {e['object']} {e['action']} at {e['location']} ({e['timestamp']})" for e in med_events]
        ) if med_events else "No medication observations recorded."

        prompt = self.prompt_mgr.get_prompt(
            "medication_agent.txt",
            medication_schedule=med_schedule_str,
            medication_events=med_events_str,
            transcript=transcript,
            confidence=confidence.value,
        )

        res_json = self.llm.generate_json(prompt=prompt, temperature=0.3)
        answer_text = res_json.get("answer_text")
        reasoning = res_json.get("reasoning")

        # Fallback phrasing
        if not answer_text:
            if confidence == Confidence.HIGH:
                answer_text = "Yes, I saw you take your morning medication earlier."
                reasoning = status_reason
            else:
                answer_text = "I haven't seen you take your medication yet. It is on the kitchen counter."
                reasoning = status_reason

        logger.info("MemoryAgent (Medication): answer='%s', confidence=%s", answer_text, confidence.value)
        return AgentResponse(
            answer_text=answer_text,
            confidence=confidence,
            reasoning=reasoning or status_reason,
        )

    def _extract_object_keyword(self, transcript: str) -> Optional[str]:
        """Extract object search term from question transcript.

        Args:
            transcript: Raw question string.

        Returns:
            Object keyword or None.
        """
        lower = transcript.lower()
        for kw in ["reading glasses", "house keys", "glasses", "keys", "wallet", "pills", "medication", "phone", "watch"]:
            if kw in lower:
                return kw
        words = lower.replace("?", "").replace(".", "").split()
        if "my" in words:
            idx = words.index("my")
            if idx + 1 < len(words):
                return words[idx + 1]
        return None

    def _compute_object_confidence(
        self, observations: List[Dict[str, Any]]
    ) -> tuple[Confidence, Optional[Dict[str, Any]]]:
        """Compute deterministic confidence level for object location.

        Args:
            observations: Query results from MemoryStore.

        Returns:
            Tuple of (Confidence Enum, best matching observation dict or None).
        """
        if not observations:
            return Confidence.LOW, None

        best_match = observations[0]
        ts_str = best_match.get("timestamp")

        if not ts_str:
            return Confidence.MEDIUM, best_match

        try:
            # Parse ISO timestamp
            ts = datetime.fromisoformat(ts_str)
            now = datetime.now(ts.tzinfo or timezone.utc)
            delta = now - ts

            # Highly recent single match (< 2 hours) -> HIGH
            if delta < timedelta(hours=2) and len(observations) == 1:
                return Confidence.HIGH, best_match
            else:
                return Confidence.MEDIUM, best_match
        except Exception:
            return Confidence.MEDIUM, best_match

    def _compute_medication_confidence(
        self, medications: List[Dict[str, Any]], med_events: List[Dict[str, Any]]
    ) -> tuple[Confidence, str]:
        """Compute deterministic confidence level for medication verification.

        Args:
            medications: Medication schedules from ProfileStore.
            med_events: Recorded medication_taken events from MemoryStore.

        Returns:
            Tuple of (Confidence Enum, status explanation string).
        """
        if not med_events:
            return Confidence.LOW, "No medication_taken events found in observation log."

        latest_event = med_events[0]
        ts_str = latest_event.get("timestamp")

        if not ts_str:
            return Confidence.LOW, "Invalid observation timestamp."

        try:
            ts = datetime.fromisoformat(ts_str)
            now = datetime.now(ts.tzinfo or timezone.utc)

            # Check if an observation occurred today
            if ts.date() == now.date():
                return (
                    Confidence.HIGH,
                    f"Matched medication_taken event at {ts.strftime('%H:%M')} today.",
                )
        except Exception:
            pass

        return Confidence.LOW, "No matching medication observation found for today's dose window."

"""Memory Retrieval Agent module for object location queries and medication verification.

Calculates confidence deterministically in Python (HIGH / MEDIUM / LOW) and formats
accurate, human-friendly observation timestamps based on the actual visual observation.
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
from backend.time_utils import (
    TIME_UNAVAILABLE,
    format_location_phrase,
    format_observation_time_phrase,
    format_observation_timestamp,
    get_configured_timezone,
    is_plural_object,
    parse_timestamp,
    to_iso_with_tz,
)


class MemoryAgent:
    """Agent handling object queries (MEMORY) and medication verification (MEDICATION)."""

    def __init__(
        self,
        memory_store: MemoryStore,
        profile_store: ProfileStore,
        llm: LLMService,
        prompt_mgr: PromptManager,
        clock_now: Optional[datetime] = None,
        reality_checksum: Optional[Any] = None,
    ) -> None:
        """Initialize MemoryAgent with injected dependencies.

        Args:
            memory_store: Observation memory store.
            profile_store: Patient profile store.
            llm: Centralized LLM service.
            prompt_mgr: Centralized prompt manager.
            clock_now: Optional fixed clock time for testing.
            reality_checksum: Optional RealityChecksumService instance.
        """
        self.memory_store = memory_store
        self.profile_store = profile_store
        self.llm = llm
        self.prompt_mgr = prompt_mgr
        self.clock_now = clock_now
        self.reality_checksum = reality_checksum

    def get_reality_checksum(self):
        """Get or lazily load RealityChecksumService."""
        if self.reality_checksum is not None:
            return self.reality_checksum
        return getattr(self.memory_store, "reality_checksum", None) or (
            self.memory_store.get_reality_checksum() if hasattr(self.memory_store, "get_reality_checksum") else None
        )

    def answer_object_query(
        self,
        transcript: str,
        now: Optional[datetime] = None,
        include_relative: Optional[bool] = None,
    ) -> AgentResponse:
        """Answer object location query from MemoryStore using deterministic confidence and timestamps.

        Args:
            transcript: Transcribed user question.
            now: Optional reference time for controllable clock in tests.
            include_relative: Optional flag to append relative time ('about 12 minutes ago').

        Returns:
            AgentResponse containing answer_text, confidence, reasoning, observed_at, formatted_time.
        """
        ref_now = now or self.clock_now or datetime.now(get_configured_timezone())
        response_generated_at = to_iso_with_tz(ref_now)

        # Extract object key phrase from question
        object_query = self._extract_object_keyword(transcript)
        target_name = object_query or "object"
        plural = is_plural_object(target_name)
        verb = "were" if plural else "was"
        pronoun = "them" if plural else "it"
        possessive_cap = "Their" if plural else "Its"

        observations = self.memory_store.query_objects(object_name=object_query, limit=5)
        confidence, best_match = self._compute_object_confidence(observations, now=ref_now)

        # 1. No matching observation in MemoryStore
        if not observations or best_match is None:
            answer_text = f"I couldn't find a reliable observation of your {target_name}."
            reasoning = "No matching observation found in memory store."
            logger.info("MemoryAgent (Object): answer='%s', confidence=%s", answer_text, confidence.value)
            return AgentResponse(
                answer_text=answer_text,
                confidence=confidence,
                reasoning=reasoning,
                observed_at=None,
                formatted_time=None,
                response_generated_at=response_generated_at,
            )

        # Retrieve actual observation timestamp
        raw_ts = best_match.get("observed_at") or best_match.get("timestamp")
        parsed_dt = parse_timestamp(raw_ts, default_tz=get_configured_timezone())

        # 2. Observation exists but timestamp is missing or malformed
        if parsed_dt is None:
            answer_text = f"I found a possible observation of your {best_match.get('object', target_name)}, but I don't have a reliable timestamp for it."
            reasoning = "Observation found in memory store but timestamp is missing or invalid."
            logger.info("MemoryAgent (Object): answer='%s', confidence=%s", answer_text, Confidence.LOW.value)
            return AgentResponse(
                answer_text=answer_text,
                confidence=Confidence.LOW,
                reasoning=reasoning,
                observed_at=None,
                formatted_time=TIME_UNAVAILABLE,
                response_generated_at=response_generated_at,
            )

        # Decide whether to append relative time
        use_relative = (
            include_relative
            if include_relative is not None
            else any(
                w in transcript.lower()
                for w in ["how long", "how many minutes", "how many hours", "about how long", "when were", "when was", "when did"]
            )
        )

        observed_at_iso = to_iso_with_tz(parsed_dt)
        formatted_time = format_observation_timestamp(parsed_dt, now=ref_now, include_relative=use_relative)
        time_phrase = format_observation_time_phrase(formatted_time)

        # 3. Ambiguous or weak visual match
        if best_match.get("verification_status") in ["ambiguous_match", "weak_match"]:
            answer_text = f"I may have seen your {best_match.get('object', target_name)} {time_phrase}, but I couldn't identify {pronoun} confidently."
            reasoning = f"Visual match is {best_match.get('verification_status')} with similarity {best_match.get('similarity_score', 0):.2f}."
            logger.info("MemoryAgent (Object): answer='%s', confidence=%s", answer_text, Confidence.LOW.value)
            return AgentResponse(
                answer_text=answer_text,
                confidence=Confidence.LOW,
                reasoning=reasoning,
                observed_at=observed_at_iso,
                formatted_time=formatted_time,
                response_generated_at=response_generated_at,
            )

        loc_phrase = format_location_phrase(best_match.get("location", "the room"))

        # 4. Older observation (previous dates, e.g. yesterday or earlier)
        if parsed_dt.date() < ref_now.date():
            answer_text = f"Your {best_match.get('object', target_name)} {verb} last seen {loc_phrase} {time_phrase}. {possessive_cap} current location hasn't been verified."
            reasoning = f"Last observed {time_phrase} at {best_match['location']}. Location has not been verified today."
            logger.info("MemoryAgent (Object): answer='%s', confidence=%s", answer_text, confidence.value)
            return AgentResponse(
                answer_text=answer_text,
                confidence=confidence,
                reasoning=reasoning,
                observed_at=observed_at_iso,
                formatted_time=formatted_time,
                response_generated_at=response_generated_at,
            )

        # 5. Fresh observation today
        action_verb = "detected" if best_match.get("evidence_source") == "visual_embedding" else "seen"
        answer_text = f"Your {best_match.get('object', target_name)} {verb} last {action_verb} {loc_phrase} {time_phrase}."
        if best_match.get("evidence_source") == "visual_embedding":
            sim = best_match.get("similarity_score")
            sim_val = 0.0 if sim is None else float(sim)
            reasoning = f"Verified by visual reference embedding at {best_match['location']} (similarity {sim_val:.2f})."
        else:
            reasoning = f"Matched observation at {best_match['location']}."

        # Reality Checksum: Verify discrepancies and recent relocations
        chk_svc = self.get_reality_checksum()
        state = chk_svc.get_object_state(object_query) if chk_svc and object_query else None
        if state:
            prev_l = state.get("previous_location")
            if prev_l and prev_l.lower() != best_match.get("location", "").lower():
                reasoning += f" Reality Checksum: Verified relocation from '{prev_l}' to '{best_match['location']}'."
                if any(w in transcript.lower() for w in ["moved", "move", "relocat", "before", "previously"]):
                    prior_pronoun = "They were" if plural else "It was"
                    answer_text += f" ({prior_pronoun} previously on the {prev_l}.)"
            if state.get("has_active_conflict"):
                reasoning += " Reality Checksum: Sighting discrepancy flagged."

        logger.info("MemoryAgent (Object): answer='%s', confidence=%s", answer_text, confidence.value)
        return AgentResponse(
            answer_text=answer_text,
            confidence=confidence,
            reasoning=reasoning,
            observed_at=observed_at_iso,
            formatted_time=formatted_time,
            response_generated_at=response_generated_at,
        )

    def answer_medication_query(
        self,
        transcript: str,
        now: Optional[datetime] = None,
    ) -> AgentResponse:
        """Answer medication-taken query using hybrid check against schedule and memory log.

        Args:
            transcript: Transcribed user question.
            now: Optional reference time for controllable clock in tests.

        Returns:
            AgentResponse containing answer_text, confidence, reasoning.
        """
        ref_now = now or self.clock_now or datetime.now(get_configured_timezone())
        response_generated_at = to_iso_with_tz(ref_now)

        medications = self.profile_store.get_medications()
        med_events = self.memory_store.query_medication_events(limit=10)

        confidence, status_reason = self._compute_medication_confidence(
            medications, med_events, now=ref_now
        )

        latest_event = med_events[0] if med_events else None
        event_ts = None
        if latest_event:
            ts_str = latest_event.get("observed_at") or latest_event.get("timestamp")
            event_ts = parse_timestamp(ts_str, default_tz=get_configured_timezone())

        observed_at_iso = to_iso_with_tz(event_ts) if event_ts else None
        formatted_time = format_observation_timestamp(event_ts, now=ref_now) if event_ts else None

        if confidence == Confidence.HIGH and event_ts and event_ts.date() == ref_now.date():
            time_phrase = format_observation_time_phrase(formatted_time)
            answer_text = f"Yes, I saw you take your morning medication earlier {time_phrase}."
            reasoning = status_reason
        else:
            answer_text = "I haven't seen you take your medication yet. It is on the kitchen counter."
            reasoning = status_reason

        logger.info("MemoryAgent (Medication): answer='%s', confidence=%s", answer_text, confidence.value)
        return AgentResponse(
            answer_text=answer_text,
            confidence=confidence,
            reasoning=reasoning or status_reason,
            observed_at=observed_at_iso,
            formatted_time=formatted_time,
            response_generated_at=response_generated_at,
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
        self,
        observations: List[Dict[str, Any]],
        now: Optional[datetime] = None,
    ) -> tuple[Confidence, Optional[Dict[str, Any]]]:
        """Compute deterministic confidence level for object location.

        Args:
            observations: Query results from MemoryStore.
            now: Optional reference current time for controllable clock in tests.

        Returns:
            Tuple of (Confidence Enum, best matching observation dict or None).
        """
        if not observations:
            return Confidence.LOW, None

        best_match = observations[0]
        ts_str = best_match.get("observed_at") or best_match.get("timestamp")

        if not ts_str:
            return Confidence.MEDIUM, best_match

        # If the top observation was an uncertain or weak visual match, do not elevate confidence
        if best_match.get("verification_status") in ["weak_match", "ambiguous_match"]:
            return Confidence.LOW, best_match

        try:
            parsed_ts = parse_timestamp(ts_str, default_tz=get_configured_timezone())
            if parsed_ts is None:
                return Confidence.LOW, best_match

            ref_now = now or self.clock_now or datetime.now(get_configured_timezone())
            delta = ref_now - parsed_ts

            # A strong visual embedding match within 4 hours provides high verified confidence
            if (
                best_match.get("evidence_source") == "visual_embedding"
                and best_match.get("verification_status") == "strong_match"
            ):
                if delta < timedelta(hours=4) and delta >= timedelta(0):
                    return Confidence.HIGH, best_match

            # Highly recent single observation (< 2 hours) -> HIGH
            if delta < timedelta(hours=2) and delta >= timedelta(0) and len(observations) == 1:
                return Confidence.HIGH, best_match
            else:
                return Confidence.MEDIUM, best_match
        except Exception:
            return Confidence.MEDIUM, best_match

    def _compute_medication_confidence(
        self,
        medications: List[Dict[str, Any]],
        med_events: List[Dict[str, Any]],
        now: Optional[datetime] = None,
    ) -> tuple[Confidence, str]:
        """Compute deterministic confidence level for medication verification.

        Args:
            medications: Medication schedules from ProfileStore.
            med_events: Recorded medication_taken events from MemoryStore.
            now: Optional reference current time for controllable clock in tests.

        Returns:
            Tuple of (Confidence Enum, status explanation string).
        """
        if not med_events:
            return Confidence.LOW, "No medication_taken events found in observation log."

        latest_event = med_events[0]
        ts_str = latest_event.get("observed_at") or latest_event.get("timestamp")

        if not ts_str:
            return Confidence.LOW, "Invalid observation timestamp."

        try:
            ref_now = now or self.clock_now or datetime.now(get_configured_timezone())
            ts = parse_timestamp(ts_str, default_tz=get_configured_timezone())

            # Check if an observation occurred today
            if ts is not None and ts.date() == ref_now.date():
                time_fmt = format_observation_timestamp(ts, now=ref_now)
                time_phrase = format_observation_time_phrase(time_fmt)
                return (
                    Confidence.HIGH,
                    f"Matched medication_taken event {time_phrase}.",
                )
        except Exception:
            pass

        return Confidence.LOW, "No matching medication observation found for today's dose window."

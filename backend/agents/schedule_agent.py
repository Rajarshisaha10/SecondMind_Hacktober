"""Schedule Agent module for appointments, routines, and daily plans.

Calculates confidence deterministically and uses Gemma for spoken phrasing.
"""

from datetime import datetime
from typing import Any, Dict, List
from backend.constants import Confidence
from backend.llm_service import LLMService
from backend.logger import logger
from backend.prompt_manager import PromptManager
from backend.schemas import AgentResponse
from backend.stores.schedule_store import ScheduleStore


class ScheduleAgent:
    """Agent answering appointment, routine, and daily plan questions (SCHEDULE)."""

    def __init__(
        self,
        schedule_store: ScheduleStore,
        llm: LLMService,
        prompt_mgr: PromptManager,
    ) -> None:
        """Initialize ScheduleAgent with injected dependencies.

        Args:
            schedule_store: Schedule store.
            llm: Centralized LLM service.
            prompt_mgr: Centralized prompt manager.
        """
        self.schedule_store = schedule_store
        self.llm = llm
        self.prompt_mgr = prompt_mgr

    def answer(self, transcript: str) -> AgentResponse:
        """Answer schedule/appointment query.

        Args:
            transcript: Transcribed user question.

        Returns:
            AgentResponse containing answer_text, confidence, reasoning.
        """
        now = datetime.now()
        current_dt_str = now.strftime("%Y-%m-%d %H:%M (%A)")

        appointments = self.schedule_store.get_appointments()
        routines = self.schedule_store.get_routines(day_of_week=now.strftime("%A").lower())
        custom_reminders = self.schedule_store.get_custom_reminders()

        # Compute confidence deterministically
        confidence = self._compute_schedule_confidence(appointments, routines)

        app_str = "\n".join([f"- {a['title']} at {a['datetime']} ({a.get('location', '')})" for a in appointments]) if appointments else "None"
        routine_str = "\n".join([f"- {r['name']} ({r['start_time']}): steps={r['steps']}" for r in routines]) if routines else "None"
        custom_str = "\n".join([f"- {c['message']} at {c['trigger_time']}" for c in custom_reminders]) if custom_reminders else "None"

        prompt = self.prompt_mgr.get_prompt(
            "schedule_agent.txt",
            current_datetime=current_dt_str,
            appointments=app_str,
            routines=routine_str,
            custom_reminders=custom_str,
            transcript=transcript,
            confidence=confidence.value,
        )

        res_json = self.llm.generate_json(prompt=prompt, temperature=0.3)
        answer_text = res_json.get("answer_text")
        reasoning = res_json.get("reasoning")

        if not answer_text:
            if appointments:
                first_app = appointments[0]
                answer_text = f"You have an appointment: {first_app['title']} at {first_app['datetime']}."
                reasoning = "Retrieved direct appointment match from schedule store."
            elif routines:
                first_rt = routines[0]
                answer_text = f"Your {first_rt['name']} starts around {first_rt['start_time']}."
                reasoning = "Retrieved routine block from schedule store."
            else:
                answer_text = "You don't have any appointments listed on your schedule today."
                reasoning = "No appointments or routines found in schedule store."

        logger.info("ScheduleAgent: answer='%s', confidence=%s", answer_text, confidence.value)
        return AgentResponse(
            answer_text=answer_text,
            confidence=confidence,
            reasoning=reasoning or f"Confidence {confidence.value} calculated from schedule entries.",
        )

    def _compute_schedule_confidence(
        self, appointments: List[Dict[str, Any]], routines: List[Dict[str, Any]]
    ) -> Confidence:
        """Compute deterministic confidence level for schedule queries.

        Args:
            appointments: List of appointments.
            routines: List of routine blocks.

        Returns:
            Confidence Enum.
        """
        if appointments:
            return Confidence.HIGH
        elif routines:
            return Confidence.HIGH
        return Confidence.LOW

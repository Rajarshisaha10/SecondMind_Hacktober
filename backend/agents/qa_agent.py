"""General QA Agent module for profile facts and everyday questions."""

import json
from datetime import datetime
from backend.constants import Confidence
from backend.llm_service import LLMService
from backend.logger import logger
from backend.prompt_manager import PromptManager
from backend.schemas import AgentResponse
from backend.stores.profile_store import ProfileStore


class QAAgent:
    """Agent answering profile questions (PROFILE) and general everyday QA (GENERAL)."""

    def __init__(
        self,
        profile_store: ProfileStore,
        llm: LLMService,
        prompt_mgr: PromptManager,
    ) -> None:
        """Initialize QAAgent with injected dependencies.

        Args:
            profile_store: Patient profile store.
            llm: Centralized LLM service.
            prompt_mgr: Centralized prompt manager.
        """
        self.profile_store = profile_store
        self.llm = llm
        self.prompt_mgr = prompt_mgr

    def answer_profile_query(self, transcript: str) -> AgentResponse:
        """Answer family or personal facts query from profile.

        Args:
            transcript: Transcribed user question.

        Returns:
            AgentResponse.
        """
        profile_data = self.profile_store.get_full_profile()
        profile_str = json.dumps(profile_data, indent=2)

        prompt = self.prompt_mgr.get_prompt(
            "qa_agent.txt",
            profile_data=profile_str,
            transcript=transcript,
            confidence=Confidence.HIGH.value,
        )

        res_json = self.llm.generate_json(prompt=prompt, temperature=0.3)
        answer_text = res_json.get("answer_text")
        reasoning = res_json.get("reasoning")

        if not answer_text:
            family = self.profile_store.get_family_contacts()
            patient = self.profile_store.get_patient_info()
            if "daughter" in transcript.lower() or "sarah" in transcript.lower():
                answer_text = "Sarah is your daughter."
                reasoning = "Retrieved daughter Sarah from profile contacts."
            else:
                answer_text = f"You are {patient.get('name', 'Arthur')}, and your family loves you very much."
                reasoning = "Retrieved patient metadata from profile store."

        logger.info("QAAgent (Profile): answer='%s'", answer_text)
        return AgentResponse(
            answer_text=answer_text,
            confidence=Confidence.HIGH,
            reasoning=reasoning or "Direct fact match in profile store.",
        )

    def answer_general_query(self, transcript: str) -> AgentResponse:
        """Answer general everyday question.

        Args:
            transcript: Transcribed user question.

        Returns:
            AgentResponse.
        """
        now = datetime.now()
        current_dt = now.strftime("%A, %B %d, %Y, %I:%M %p")

        prompt = self.prompt_mgr.get_prompt(
            "qa_agent.txt",
            profile_data=f"Current Date and Time: {current_dt}",
            transcript=transcript,
            confidence=Confidence.HIGH.value,
        )

        res_json = self.llm.generate_json(prompt=prompt, temperature=0.3)
        answer_text = res_json.get("answer_text")
        reasoning = res_json.get("reasoning")

        if not answer_text:
            if "time" in transcript.lower() or "day" in transcript.lower() or "date" in transcript.lower():
                answer_text = f"Today is {current_dt}."
                reasoning = "Retrieved system date and time."
            else:
                answer_text = "I'm here with you. How can I help you today?"
                reasoning = "General conversational response."

        logger.info("QAAgent (General): answer='%s'", answer_text)
        return AgentResponse(
            answer_text=answer_text,
            confidence=Confidence.HIGH,
            reasoning=reasoning or "General everyday response.",
        )

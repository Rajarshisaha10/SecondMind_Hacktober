"""Emergency Agent module for distress calls and contact surfacing."""

from backend.constants import Confidence, EventType
from backend.llm_service import LLMService
from backend.prompt_manager import PromptManager
from backend.schemas import AgentResponse
from backend.stores.memory_store import MemoryStore
from backend.stores.profile_store import ProfileStore


class EmergencyAgent:
    """Agent handling emergency distress intents (EMERGENCY)."""

    def __init__(
        self,
        profile_store: ProfileStore,
        memory_store: MemoryStore,
        llm: LLMService,
        prompt_mgr: PromptManager,
    ) -> None:
        """Initialize EmergencyAgent with injected dependencies.

        Args:
            profile_store: Profile store containing family contacts.
            memory_store: Memory store to record help_request event.
            llm: Centralized LLM service.
            prompt_mgr: Centralized prompt manager.
        """
        self.profile_store = profile_store
        self.memory_store = memory_store
        self.llm = llm
        self.prompt_mgr = prompt_mgr

    def answer(self, transcript: str) -> AgentResponse:
        """Answer emergency call and log help_request event to MemoryStore.

        Args:
            transcript: Transcribed user question.

        Returns:
            AgentResponse.
        """
        contacts = self.profile_store.get_family_contacts()
        contact_str = "555-0199"
        if contacts:
            c = contacts[0]
            contact_str = f"{c.get('name')} ({c.get('phone')})"

        # Log help_request event
        self.memory_store.add_observation(
            object_name="emergency_contact",
            location="wearable_device",
            action="help_requested",
            event_type=EventType.HELP_REQUEST,
            raw_description=f"Emergency distress call: '{transcript}'",
        )

        return AgentResponse(
            answer_text=f"I found your emergency contact: {contact_str}. I'm here with you.",
            confidence=Confidence.HIGH,
            reasoning="Surfaced emergency contact from profile.",
        )

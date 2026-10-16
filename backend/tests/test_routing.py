"""Unit test suite for OrchestratorAgent intent classification and routing across all six intent categories."""

import tempfile
import pytest
from pathlib import Path
from backend.constants import Intent
from backend.llm_service import LLMService
from backend.prompt_manager import PromptManager
from backend.stores.memory_store import MemoryStore
from backend.stores.profile_store import ProfileStore
from backend.stores.schedule_store import ScheduleStore
from backend.stores.session_store import SessionStore
from backend.agents.memory_agent import MemoryAgent
from backend.agents.schedule_agent import ScheduleAgent
from backend.agents.qa_agent import QAAgent
from backend.agents.emergency_agent import EmergencyAgent
from backend.agents.orchestrator import OrchestratorAgent


@pytest.fixture
def orchestrator() -> OrchestratorAgent:
    """Fixture initializing OrchestratorAgent with all required stores and sub-agents."""
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "test_routing.db"
    memory_store = MemoryStore(db_path=db_path)
    profile_store = ProfileStore()
    schedule_store = ScheduleStore()
    session_store = SessionStore()
    prompt_mgr = PromptManager()
    llm = LLMService()

    memory_agent = MemoryAgent(memory_store, profile_store, llm, prompt_mgr)
    schedule_agent = ScheduleAgent(schedule_store, llm, prompt_mgr)
    qa_agent = QAAgent(profile_store, llm, prompt_mgr)
    emergency_agent = EmergencyAgent(profile_store, memory_store, llm, prompt_mgr)

    return OrchestratorAgent(
        llm=llm,
        prompt_mgr=prompt_mgr,
        session_store=session_store,
        memory_agent=memory_agent,
        schedule_agent=schedule_agent,
        qa_agent=qa_agent,
        emergency_agent=emergency_agent,
    )


def test_emergency_routing(orchestrator: OrchestratorAgent) -> None:
    """Verify emergency distress phrases route to EMERGENCY intent."""
    response, resolved, intent = orchestrator.dispatch("I need help, call my daughter!")
    assert intent == Intent.EMERGENCY
    assert "emergency contact" in response.answer_text.lower() or "555" in response.answer_text


def test_intent_classification(orchestrator: OrchestratorAgent) -> None:
    """Verify orchestrator classifies sample questions into expected categories."""
    assert orchestrator.classify_intent("Where are my reading glasses?") in [Intent.MEMORY, Intent.GENERAL]
    assert orchestrator.classify_intent("Have I taken my morning medication?") in [Intent.MEDICATION, Intent.GENERAL]
    assert orchestrator.classify_intent("Do I have any doctor appointments today?") in [Intent.SCHEDULE, Intent.GENERAL]
    assert orchestrator.classify_intent("Who is Sarah?") in [Intent.PROFILE, Intent.GENERAL]
    assert orchestrator.classify_intent("What is the capital of France?") in [Intent.GENERAL]
    assert orchestrator.classify_intent("I need help right now") == Intent.EMERGENCY

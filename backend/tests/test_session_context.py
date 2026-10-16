"""Unit test suite for session context resolution and multi-turn pronoun tracking."""

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
    """Fixture initializing OrchestratorAgent."""
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "test_session.db"
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


def test_session_store_context_update(orchestrator: OrchestratorAgent) -> None:
    """Verify orchestrator updates session context attributes."""
    device_id = "test_dev_01"
    orchestrator.session_store.update_session(
        device_id=device_id,
        subject="reading glasses",
        intent=Intent.MEMORY,
    )
    sess = orchestrator.session_store.get_session(device_id)
    assert sess["last_subject"] == "reading glasses"
    assert sess["last_intent"] == Intent.MEMORY.value


def test_followup_resolution_without_context(orchestrator: OrchestratorAgent) -> None:
    """Verify standalone question without context returns unchanged."""
    res = orchestrator.resolve_session_context("Where are my reading glasses?", "new_dev")
    assert res == "Where are my reading glasses?"


def test_multiturn_pronoun_resolution_dispatch(orchestrator: OrchestratorAgent) -> None:
    """Verify dispatch extracts subject and resolves pronoun in subsequent turn."""
    device_id = "test_dev_multi"
    # Turn 1
    _, resolved_1, intent_1 = orchestrator.dispatch("Where are my reading glasses?", device_id=device_id)
    assert intent_1 == Intent.MEMORY
    sess = orchestrator.session_store.get_session(device_id)
    assert sess["last_subject"] == "reading glasses"

    # Turn 2 with pronoun "them"
    _, resolved_2, _ = orchestrator.dispatch("Did I move them after lunch?", device_id=device_id)
    assert "reading glasses" in resolved_2.lower()


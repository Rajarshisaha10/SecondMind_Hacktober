"""Reminder Service module managing proactive reminder checks and acknowledgements."""

from backend.schemas import ReminderAckResponse, ReminderCheckResponse
from backend.agents.reminder_agent import ReminderAgent
from backend.constants import ReminderType


class ReminderService:
    """Service handling proactive reminder polling and acknowledgement API endpoints."""

    def __init__(self, reminder_agent: ReminderAgent) -> None:
        """Initialize ReminderService with injected ReminderAgent.

        Args:
            reminder_agent: ReminderAgent background timer instance.
        """
        self.reminder_agent = reminder_agent

    def start_agent(self) -> None:
        """Start the background reminder loop agent thread."""
        self.reminder_agent.start()

    def stop_agent(self) -> None:
        """Stop the background reminder loop agent thread."""
        self.reminder_agent.stop()

    def check_reminder(self) -> ReminderCheckResponse:
        """Check if a proactive reminder is pending in the single-slot queue.

        Returns:
            ReminderCheckResponse object.
        """
        pending = self.reminder_agent.get_pending_reminder()
        if not pending:
            return ReminderCheckResponse(pending=False)

        rem_type_val = pending.get("reminder_type")
        rem_type = None
        for rt in ReminderType:
            if rt.value == rem_type_val:
                rem_type = rt
                break

        return ReminderCheckResponse(
            pending=True,
            audio_url=pending.get("audio_url"),
            message=pending.get("message"),
            reminder_type=rem_type,
        )

    def acknowledge_reminder(self) -> ReminderAckResponse:
        """Acknowledge and clear the current pending reminder slot.

        Returns:
            ReminderAckResponse object.
        """
        cleared = self.reminder_agent.acknowledge_reminder()
        return ReminderAckResponse(
            status="acknowledged" if cleared else "no_pending_reminder",
            cleared=cleared,
        )

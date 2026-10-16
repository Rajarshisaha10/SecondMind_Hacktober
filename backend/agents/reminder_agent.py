"""Reminder Agent background timer module for all proactive reminder categories.

Monitors medication schedules against observed MemoryStore events and system clock.
Queues proactive spoken reminders into a single pending slot and caps nudges to 2 per event.
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.config import settings
from backend.constants import EventType, ReminderType
from backend.logger import logger
from backend.stores.memory_store import MemoryStore
from backend.stores.profile_store import ProfileStore
from backend.stores.schedule_store import ScheduleStore
from backend.tts import TTSWrapper


class ReminderAgent:
    """Proactive background timer agent checking schedules and queuing reminders."""

    def __init__(
        self,
        profile_store: ProfileStore,
        schedule_store: ScheduleStore,
        memory_store: MemoryStore,
        tts: Optional[TTSWrapper] = None,
    ) -> None:
        """Initialize ReminderAgent with injected dependencies.

        Args:
            profile_store: Patient profile store.
            schedule_store: Schedule store.
            memory_store: Observation memory store.
            tts: Optional TTS wrapper instance.
        """
        self.profile_store = profile_store
        self.schedule_store = schedule_store
        self.memory_store = memory_store
        self.tts = tts or TTSWrapper()

        self._pending_reminder: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Tracking fired nudges: key = event_identifier, value = nudge_count
        self._fired_nudges: Dict[str, int] = {}

    def get_pending_reminder(self) -> Optional[Dict[str, Any]]:
        """Get currently queued pending reminder.

        Returns:
            Dict representing pending reminder or None if queue is empty.
        """
        with self._lock:
            return self._pending_reminder

    def acknowledge_reminder(self) -> bool:
        """Clear the pending reminder slot upon device acknowledgement.

        Returns:
            True if a reminder was cleared, False otherwise.
        """
        with self._lock:
            if self._pending_reminder is not None:
                rem_type = self._pending_reminder.get("reminder_type")
                self._pending_reminder = None
                logger.info("ReminderAgent: Cleared pending reminder slot (%s).", rem_type)
                return True
            return False

    def start(self) -> None:
        """Start background reminder checking thread."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("ReminderAgent: Proactive background timer thread started.")

    def stop(self) -> None:
        """Stop background thread loop."""
        self._running = False

    def _run_loop(self) -> None:
        """Background loop executing check every 30 seconds."""
        while self._running:
            try:
                self.check_reminders()
            except Exception as e:
                logger.error("ReminderAgent: Error in check_reminders loop: %s", e)
            time.sleep(30)

    def check_reminders(self) -> None:
        """Check medication, appointment, routine, and custom reminders against memory log."""
        with self._lock:
            if self._pending_reminder is not None:
                # Slot is occupied, wait for device ack
                return

        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        # 1. Medication Reminders Check
        self._check_medications(now, today_str)

        # 2. Appointment Reminders Check
        with self._lock:
            if self._pending_reminder is None:
                self._check_appointments(now, today_str)

        # 3. Custom Reminders Check
        with self._lock:
            if self._pending_reminder is None:
                self._check_custom_reminders(now, today_str)

    def _check_medications(self, now: datetime, today_str: str) -> None:
        """Check scheduled medication dose times against memory store observations.

        Args:
            now: Current datetime object.
            today_str: YYYY-MM-DD date string.
        """
        medications = self.profile_store.get_medications()
        med_events = self.memory_store.query_medication_events(since_iso=today_str, limit=10)

        for med in medications:
            name = med.get("name", "medication")
            schedule_times = med.get("schedule_times", [])
            grace_min = med.get("grace_window_minutes", 30)

            for st in schedule_times:
                try:
                    hour, minute = map(int, st.split(":"))
                    dose_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    dose_window_end = dose_time + timedelta(minutes=grace_min)
                    nudge_key = f"med_{name}_{today_str}_{st}"

                    # If current time is past dose time and within grace window
                    if dose_time <= now <= dose_window_end + timedelta(minutes=15):
                        # Check if a medication_taken event exists for today
                        taken = False
                        for evt in med_events:
                            evt_ts = evt.get("timestamp", "")
                            if evt_ts.startswith(today_str):
                                taken = True
                                break

                        current_nudges = self._fired_nudges.get(nudge_key, 0)
                        if not taken and current_nudges < 2:
                            message = f"It is time for your {name} medication."
                            audio_path = self.tts.synthesize_to_file(message)

                            with self._lock:
                                self._pending_reminder = {
                                    "pending": True,
                                    "reminder_type": ReminderType.MEDICATION.value,
                                    "message": message,
                                    "audio_url": f"/audio/{Path(audio_path).name}",
                                }

                            self._fired_nudges[nudge_key] = current_nudges + 1
                            logger.info(
                                "ReminderAgent: Queued medication reminder '%s' (nudge %d/2)",
                                message,
                                current_nudges + 1,
                            )
                            return
                except Exception as e:
                    logger.error("ReminderAgent: Error checking medication dose time '%s': %s", st, e)

    def _check_appointments(self, now: datetime, today_str: str) -> None:
        """Check appointment lead time windows.

        Args:
            now: Current datetime object.
            today_str: YYYY-MM-DD date string.
        """
        appointments = self.schedule_store.get_appointments()
        for app in appointments:
            app_id = app.get("id", "app")
            dt_str = app.get("datetime", "")
            title = app.get("title", "Appointment")
            lead_mins = app.get("reminder_lead_minutes", [30, 10])

            try:
                app_dt = datetime.fromisoformat(dt_str)
                for lead in lead_mins:
                    nudge_key = f"app_{app_id}_{lead}"
                    trigger_start = app_dt - timedelta(minutes=lead)
                    trigger_end = trigger_start + timedelta(minutes=5)

                    if trigger_start <= now <= trigger_end:
                        current_nudges = self._fired_nudges.get(nudge_key, 0)
                        if current_nudges < 1:
                            message = f"Reminder: Your appointment '{title}' is in {lead} minutes."
                            audio_path = self.tts.synthesize_to_file(message)

                            self._pending_reminder = {
                                "pending": True,
                                "reminder_type": ReminderType.APPOINTMENT.value,
                                "message": message,
                                "audio_url": f"/audio/{Path(audio_path).name}",
                            }
                            self._fired_nudges[nudge_key] = 1
                            logger.info("ReminderAgent: Queued appointment reminder '%s'", message)
                            return
            except Exception:
                pass

    def _check_custom_reminders(self, now: datetime, today_str: str) -> None:
        """Check custom caregiver reminders.

        Args:
            now: Current datetime object.
            today_str: YYYY-MM-DD date string.
        """
        custom_rems = self.schedule_store.get_custom_reminders()
        current_hm = now.strftime("%H:%M")

        for rem in custom_rems:
            rem_id = rem.get("id", "custom")
            msg = rem.get("message", "Reminder")
            trig = rem.get("trigger_time", "")
            nudge_key = f"custom_{rem_id}_{today_str}"

            if current_hm == trig and self._fired_nudges.get(nudge_key, 0) < 1:
                audio_path = self.tts.synthesize_to_file(msg)
                self._pending_reminder = {
                    "pending": True,
                    "reminder_type": ReminderType.CUSTOM.value,
                    "message": msg,
                    "audio_url": f"/audio/{Path(audio_path).name}",
                }
                self._fired_nudges[nudge_key] = 1
                logger.info("ReminderAgent: Queued custom reminder '%s'", msg)
                return

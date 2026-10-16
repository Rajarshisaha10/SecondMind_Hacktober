"""Schedule Store module loading schedule JSON data into memory.

Maintains caregiver-entered upcoming appointments, daily routines, and custom reminders.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.config import settings
from backend.logger import logger


class ScheduleStore:
    """JSON-backed store for schedule data."""

    def __init__(self, schedule_path: Optional[Path] = None) -> None:
        """Initialize schedule store by loading schedule_setup.json.

        Args:
            schedule_path: Optional custom path to schedule setup JSON file.
        """
        self.schedule_path = schedule_path or settings.SCHEDULE_PATH
        self._data: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        """Reload schedule JSON data from disk."""
        if self.schedule_path.exists():
            try:
                with open(self.schedule_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info("ScheduleStore: Loaded schedule from '%s'", self.schedule_path)
                return
            except Exception as e:
                logger.warning("ScheduleStore: Error reading '%s': %s", self.schedule_path, e)

        # Default fallback structure
        self._data = {
            "appointments": [],
            "routines": [],
            "custom_reminders": [],
        }

    def get_appointments(self, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get upcoming appointments, optionally filtered by YYYY-MM-DD date prefix.

        Args:
            date_str: Optional YYYY-MM-DD string prefix.

        Returns:
            List of appointment dicts.
        """
        appointments = self._data.get("appointments", [])
        if date_str:
            return [
                app for app in appointments if str(app.get("datetime", "")).startswith(date_str)
            ]
        return appointments

    def get_routines(self, day_of_week: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get daily routines, optionally filtered by day of week (e.g. 'monday').

        Args:
            day_of_week: Optional day name string.

        Returns:
            List of routine block dicts.
        """
        routines = self._data.get("routines", [])
        if day_of_week:
            dow = day_of_week.lower()
            return [
                r for r in routines if dow in [d.lower() for d in r.get("days", [])]
            ]
        return routines

    def get_custom_reminders(self) -> List[Dict[str, Any]]:
        """Get list of custom caregiver reminders.

        Returns:
            List of custom reminder dicts.
        """
        return self._data.get("custom_reminders", [])

    def save_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Save updated schedule data to disk and refresh in-memory store.

        Args:
            schedule_data: Dictionary matching schedule setup schema.
        """
        self._data = schedule_data
        try:
            self.schedule_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.schedule_path, "w", encoding="utf-8") as f:
                json.dump(schedule_data, f, indent=2)
            logger.info("ScheduleStore: Saved updated schedule to '%s'", self.schedule_path)
        except Exception as e:
            logger.error("ScheduleStore: Failed to save schedule to '%s': %s", self.schedule_path, e)

    def get_full_schedule(self) -> Dict[str, Any]:
        """Get entire schedule dictionary.

        Returns:
            Dict containing appointments, routines, custom_reminders.
        """
        return self._data

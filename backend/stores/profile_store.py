"""Profile Store module loading patient profile JSON data into memory.

Maintains patient personal facts, medical conditions, medication schedule times,
and family contact information provided by the caregiver.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.config import settings
from backend.logger import logger


class ProfileStore:
    """JSON-backed store for patient profile data."""

    def __init__(self, profile_path: Optional[Path] = None) -> None:
        """Initialize profile store by loading profile_setup.json.

        Args:
            profile_path: Optional custom path to profile setup JSON file.
        """
        self.profile_path = profile_path or settings.PROFILE_PATH
        self._data: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        """Reload profile JSON data from disk."""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info("ProfileStore: Loaded setup from '%s'", self.profile_path)
                return
            except Exception as e:
                logger.warning("ProfileStore: Error reading '%s': %s", self.profile_path, e)

        # Default fallback structure
        self._data = {
            "patient": {
                "name": "Arthur Pendelton",
                "age": 76,
                "conditions": ["Mild Cognitive Impairment"],
                "medications": [
                    {
                        "name": "Donepezil",
                        "schedule_times": ["08:00"],
                        "grace_window_minutes": 30,
                    }
                ],
            },
            "family": [
                {
                    "name": "Sarah Pendelton",
                    "relationship": "Daughter",
                    "phone": "555-0199",
                }
            ],
        }

    def get_patient_info(self) -> Dict[str, Any]:
        """Get patient metadata (name, age, conditions).

        Returns:
            Dict containing patient information.
        """
        return self._data.get("patient", {})

    def get_medications(self) -> List[Dict[str, Any]]:
        """Get patient medication schedule list.

        Returns:
            List of medication dicts with schedule_times and grace_window_minutes.
        """
        patient = self._data.get("patient", {})
        return patient.get("medications", [])

    def get_family_contacts(self) -> List[Dict[str, Any]]:
        """Get family contacts list.

        Returns:
            List of family contact dicts (name, relationship, phone).
        """
        return self._data.get("family", [])

    def save_profile(self, profile_data: Dict[str, Any]) -> None:
        """Save updated profile data to disk and refresh in-memory store.

        Args:
            profile_data: Dictionary matching profile setup schema.
        """
        self._data = profile_data
        try:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)
            logger.info("ProfileStore: Saved updated profile to '%s'", self.profile_path)
        except Exception as e:
            logger.error("ProfileStore: Failed to save profile to '%s': %s", self.profile_path, e)

    def get_full_profile(self) -> Dict[str, Any]:
        """Get full raw profile dictionary.

        Returns:
            Dict containing entire profile setup.
        """
        return self._data

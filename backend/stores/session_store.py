"""Session Store module maintaining rolling short-term conversation context.

Provides thread-safe in-memory session context for resolving follow-up questions,
pronouns ("them", "it"), and chained time references within a session window.
"""

import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
from backend.config import settings
from backend.constants import Intent
from backend.logger import logger


class SessionStore:
    """Thread-safe in-memory session context store."""

    def __init__(self, ttl_minutes: int = settings.SESSION_TTL_MINUTES) -> None:
        """Initialize session store with configurable TTL.

        Args:
            ttl_minutes: Session inactivity expiration threshold in minutes.
        """
        self.ttl_minutes = ttl_minutes
        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get_session(self, device_id: str = settings.DEFAULT_DEVICE_ID) -> Dict[str, Any]:
        """Retrieve active session context for a device, resetting if expired.

        Args:
            device_id: Unique device identifier string.

        Returns:
            Dict containing session attributes (last_subject, last_resolved_time_ref, last_intent).
        """
        with self._lock:
            session = self._sessions.get(device_id)
            if not session:
                return self._create_empty_session(device_id)

            # Check expiration
            updated_at = session.get("updated_at")
            if updated_at:
                now = datetime.now(timezone.utc)
                if now - updated_at > timedelta(minutes=self.ttl_minutes):
                    logger.info("SessionStore: Session for device '%s' expired, resetting.", device_id)
                    return self._create_empty_session(device_id)

            return session

    def update_session(
        self,
        device_id: str = settings.DEFAULT_DEVICE_ID,
        subject: Optional[str] = None,
        time_ref: Optional[str] = None,
        intent: Optional[Intent | str] = None,
    ) -> Dict[str, Any]:
        """Update session context with new values from the current turn.

        Args:
            device_id: Device identifier.
            subject: Resolved subject entity (e.g. 'reading glasses').
            time_ref: Resolved timestamp reference.
            intent: Classified intent enum or string.

        Returns:
            Updated session dictionary.
        """
        with self._lock:
            session = self._sessions.get(device_id) or self._create_empty_session(device_id)

            if subject is not None:
                session["last_subject"] = subject
            if time_ref is not None:
                session["last_resolved_time_ref"] = time_ref
            if intent is not None:
                intent_str = intent.value if isinstance(intent, Intent) else str(intent)
                session["last_intent"] = intent_str

            session["updated_at"] = datetime.now(timezone.utc)
            self._sessions[device_id] = session
            logger.info("SessionStore: Updated session for device '%s': subject='%s', intent='%s'",
                        device_id, session.get("last_subject"), session.get("last_intent"))
            return session

    def clear_session(self, device_id: str = settings.DEFAULT_DEVICE_ID) -> None:
        """Clear session state for a device.

        Args:
            device_id: Device identifier.
        """
        with self._lock:
            if device_id in self._sessions:
                del self._sessions[device_id]
                logger.info("SessionStore: Cleared session for device '%s'", device_id)

    def _create_empty_session(self, device_id: str) -> Dict[str, Any]:
        """Create and return an empty session dictionary.

        Args:
            device_id: Device identifier.

        Returns:
            Dict representing a fresh session state.
        """
        session = {
            "device_id": device_id,
            "last_subject": None,
            "last_resolved_time_ref": None,
            "last_intent": None,
            "updated_at": datetime.now(timezone.utc),
        }
        self._sessions[device_id] = session
        return session

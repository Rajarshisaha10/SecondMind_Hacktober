"""Memory Store module using standard library sqlite3.

Provides an append-only observation log for object locations, medication-taken events,
and help requests passively logged by the perception agent or emergency system.
"""

import uuid
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.config import settings
from backend.constants import EventType
from backend.logger import logger


class MemoryStore:
    """SQLite repository for room memory observations."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize SQLite database connection and ensure schema exists.

        Args:
            db_path: Optional custom path to SQLite database file.
        """
        self.db_path = db_path or settings.DATABASE_PATH
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Create, yield, and automatically close SQLite connection.

        Yields:
            sqlite3.Connection object configured with row factory.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create observations table if it does not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    object TEXT NOT NULL,
                    location TEXT NOT NULL,
                    action TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    raw_description TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_event_type ON observations(event_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON observations(timestamp)"
            )
            conn.commit()

    def add_observation(
        self,
        object_name: str,
        location: str,
        action: str,
        event_type: EventType | str = EventType.OBJECT_OBSERVATION,
        raw_description: str = "",
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a new observation entry to the store.

        Args:
            object_name: Name of the observed object or entity.
            location: Room location (e.g. kitchen counter).
            action: Action performed (e.g. placed, picked up).
            event_type: EventType enum or string classification.
            raw_description: Optional detailed textual description.
            timestamp: Optional ISO formatted timestamp string. Defaults to now.

        Returns:
            Dict representing the added observation row.
        """
        obs_id = str(uuid.uuid4())
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        evt_type_str = (
            event_type.value if isinstance(event_type, EventType) else str(event_type)
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO observations (id, timestamp, object, location, action, event_type, raw_description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (obs_id, ts, object_name, location, action, evt_type_str, raw_description),
            )
            conn.commit()

        logger.info(
            "MemoryStore: Added observation '%s' (%s) at '%s'",
            object_name,
            evt_type_str,
            location,
        )

        return {
            "id": obs_id,
            "timestamp": ts,
            "object": object_name,
            "location": location,
            "action": action,
            "event_type": evt_type_str,
            "raw_description": raw_description,
        }

    def query_objects(
        self, object_name: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query recent object observations, optionally filtering by object name.

        Args:
            object_name: Optional object name search string.
            limit: Maximum number of rows to return.

        Returns:
            List of matching observation dicts ordered by timestamp descending.
        """
        with self._get_connection() as conn:
            if object_name:
                cursor = conn.execute(
                    """
                    SELECT * FROM observations
                    WHERE LOWER(object) LIKE ? AND event_type = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (f"%{object_name.lower()}%", EventType.OBJECT_OBSERVATION.value, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM observations
                    WHERE event_type = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (EventType.OBJECT_OBSERVATION.value, limit),
                )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def query_medication_events(
        self, since_iso: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query medication_taken observations.

        Args:
            since_iso: Optional ISO timestamp to search after.
            limit: Maximum number of rows to return.

        Returns:
            List of matching medication_taken observation dicts.
        """
        with self._get_connection() as conn:
            if since_iso:
                cursor = conn.execute(
                    """
                    SELECT * FROM observations
                    WHERE event_type = ? AND timestamp >= ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (EventType.MEDICATION_TAKEN.value, since_iso, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM observations
                    WHERE event_type = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (EventType.MEDICATION_TAKEN.value, limit),
                )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def query_all(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Query all recent observations regardless of type.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of observation dicts.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM observations ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
            return [dict(r) for r in cursor.fetchall()]

    def seed_demo_data(self) -> None:
        """Seed default observations if table is currently empty."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM observations")
            if cursor.fetchone()["cnt"] == 0:
                now_str = datetime.now().strftime("%Y-%m-%d")
                self.add_observation(
                    object_name="reading glasses",
                    location="kitchen counter",
                    action="placed",
                    event_type=EventType.OBJECT_OBSERVATION,
                    raw_description="A pair of reading glasses was set down near the fruit bowl.",
                    timestamp=f"{now_str}T09:15:00",
                )
                self.add_observation(
                    object_name="medication bottle",
                    location="kitchen counter",
                    action="picked up",
                    event_type=EventType.MEDICATION_TAKEN,
                    raw_description="A pill bottle was picked up and opened.",
                    timestamp=f"{now_str}T08:10:00",
                )
                self.add_observation(
                    object_name="house keys",
                    location="dining table",
                    action="placed",
                    event_type=EventType.OBJECT_OBSERVATION,
                    raw_description="House keys placed on dining table next to newspaper.",
                    timestamp=f"{now_str}T10:30:00",
                )
                logger.info("MemoryStore: Seeded default demo observations.")

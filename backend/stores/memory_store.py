"""Memory Store module using standard library sqlite3.

Provides an append-only observation log for object locations, medication-taken events,
and help requests passively logged by the perception agent or emergency system.
"""

import json
import uuid
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.config import settings
from backend.constants import EventType
from backend.logger import logger
from backend.time_utils import (
    format_observation_timestamp,
    get_configured_timezone,
    to_iso_with_tz,
)


class MemoryStore:
    """SQLite repository for room memory observations."""

    def __init__(self, db_path: Optional[Path] = None, reality_checksum: Optional[Any] = None) -> None:
        """Initialize SQLite database connection and ensure schema exists.

        Args:
            db_path: Optional custom path to SQLite database file.
            reality_checksum: Optional RealityChecksumService instance.
        """
        self.db_path = db_path or settings.DATABASE_PATH
        self.reality_checksum = reality_checksum
        self._init_db()

    def get_reality_checksum(self):
        """Lazy loader for RealityChecksumService."""
        if self.reality_checksum is None:
            try:
                from backend.services.reality_checksum_service import RealityChecksumService
                self.reality_checksum = RealityChecksumService(db_path=self.db_path, memory_store=self)
            except Exception:
                pass
        return self.reality_checksum

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
        """Create observations table if it does not exist and apply safe migrations."""
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
                    raw_description TEXT,
                    registered_object_id TEXT,
                    bounding_box TEXT,
                    similarity_score REAL,
                    evidence_source TEXT DEFAULT 'perception',
                    verification_status TEXT DEFAULT 'unverified'
                )
                """
            )
            # Safe schema migration for pre-existing databases
            existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(observations)").fetchall()}
            new_columns = [
                ("registered_object_id", "TEXT"),
                ("bounding_box", "TEXT"),
                ("similarity_score", "REAL"),
                ("evidence_source", "TEXT DEFAULT 'perception'"),
                ("verification_status", "TEXT DEFAULT 'unverified'"),
                ("observed_at", "TEXT"),
                ("created_at", "TEXT"),
                ("checksum_status", "TEXT DEFAULT 'active'"),
                ("superseded_by", "TEXT"),
            ]
            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    try:
                        conn.execute(f"ALTER TABLE observations ADD COLUMN {col_name} {col_type}")
                    except Exception:
                        pass

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_event_type ON observations(event_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON observations(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_reg_obj ON observations(registered_object_id)"
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
        registered_object_id: Optional[str] = None,
        bounding_box: Optional[Any] = None,
        similarity_score: Optional[float] = None,
        evidence_source: str = "perception",
        verification_status: str = "unverified",
        observed_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a new observation entry to the store.

        Args:
            object_name: Name of the observed object or entity.
            location: Room location (e.g. kitchen counter).
            action: Action performed (e.g. placed, picked up, visually observed).
            event_type: EventType enum or string classification.
            raw_description: Optional detailed textual description.
            timestamp: Optional ISO formatted timestamp string. Defaults to now.
            registered_object_id: Optional ID of registered reference object.
            bounding_box: Optional bounding box dict or JSON string.
            similarity_score: Optional embedding similarity score.
            evidence_source: Evidence modality ('perception', 'visual_embedding', 'manual').
            verification_status: Verification state ('strong_match', 'provisional', 'unverified').
            observed_at: Specific timestamp when object was visually observed.
            created_at: Specific timestamp when DB record was written.

        Returns:
            Dict representing the added observation row.
        """
        obs_id = str(uuid.uuid4())
        now_iso = to_iso_with_tz(datetime.now(get_configured_timezone())) or datetime.now(timezone.utc).isoformat()
        if observed_at is not None:
            obs_at = observed_at
        elif timestamp is not None:
            obs_at = timestamp
        else:
            obs_at = now_iso
        created_at_val = created_at or now_iso
        ts = obs_at
        formatted_time = format_observation_timestamp(obs_at)

        evt_type_str = (
            event_type.value if isinstance(event_type, EventType) else str(event_type)
        )
        bbox_str = json.dumps(bounding_box) if isinstance(bounding_box, dict) else (bounding_box or None)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO observations (
                    id, timestamp, object, location, action, event_type, raw_description,
                    registered_object_id, bounding_box, similarity_score, evidence_source, verification_status,
                    observed_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    obs_id,
                    ts,
                    object_name,
                    location,
                    action,
                    evt_type_str,
                    raw_description,
                    registered_object_id,
                    bbox_str,
                    similarity_score,
                    evidence_source,
                    verification_status,
                    obs_at,
                    created_at_val,
                ),
            )
            conn.commit()

        logger.info(
            "MemoryStore: Added observation '%s' (%s) at '%s' [source=%s, sim=%s, observed_at=%s]",
            object_name,
            evt_type_str,
            location,
            evidence_source,
            similarity_score,
            obs_at,
        )

        ret = {
            "id": obs_id,
            "timestamp": ts,
            "observed_at": obs_at,
            "created_at": created_at_val,
            "formatted_time": formatted_time,
            "object": object_name,
            "location": location,
            "action": action,
            "event_type": evt_type_str,
            "raw_description": raw_description,
            "registered_object_id": registered_object_id,
            "bounding_box": bounding_box,
            "similarity_score": similarity_score,
            "evidence_source": evidence_source,
            "verification_status": verification_status,
            "checksum_status": "active",
            "superseded_by": None,
        }

        # Automatically run Reality Checksum conflict detection
        checksum_svc = self.get_reality_checksum()
        if checksum_svc:
            try:
                ret["reality_checksum"] = checksum_svc.process_observation(ret)
            except Exception as e:
                logger.warning("MemoryStore: RealityChecksum conflict check failed: %s", e)

        return ret

    def add_visual_observation(
        self,
        object_name: str,
        location: str,
        registered_object_id: Optional[str] = None,
        bounding_box: Optional[Dict[str, Any]] = None,
        similarity_score: Optional[float] = None,
        verification_status: str = "strong_match",
        timestamp: Optional[str] = None,
        raw_description: str = "",
        observed_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience method to record a successful visual search match into observations."""
        obs_time = observed_at or timestamp
        desc = (
            raw_description
            or f"Visual embedding match for '{object_name}' (similarity={similarity_score:.3f}) at {location}."
        )
        return self.add_observation(
            object_name=object_name,
            location=location,
            action="visually observed",
            event_type=EventType.OBJECT_OBSERVATION,
            raw_description=desc,
            timestamp=obs_time,
            observed_at=obs_time,
            registered_object_id=registered_object_id,
            bounding_box=bounding_box,
            similarity_score=similarity_score,
            evidence_source="visual_embedding",
            verification_status=verification_status,
        )


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
            return [self._format_row(r) for r in rows]

    @staticmethod
    def _format_row(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary and deserialize bounding_box JSON."""
        d = dict(row)
        if d.get("bounding_box") and isinstance(d["bounding_box"], str):
            try:
                d["bounding_box"] = json.loads(d["bounding_box"])
            except Exception:
                pass
        obs_at = d.get("observed_at") or d.get("timestamp")
        d["observed_at"] = obs_at
        if not d.get("created_at"):
            d["created_at"] = d.get("timestamp")
        d["formatted_time"] = format_observation_timestamp(obs_at)
        d["checksum_status"] = d.get("checksum_status") or "active"
        d["is_outdated"] = d["checksum_status"] == "superseded"
        return d

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
            return [self._format_row(r) for r in rows]

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
            return [self._format_row(r) for r in cursor.fetchall()]

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

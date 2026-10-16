# ===========================================================================
# [NEW FILE - FEATURE 3: REALITY CHECKSUM]
# Path: backend/services/reality_checksum_service.py
# Summary: Detects conflicts between new observations and stored memories,
# verifies discrepancies, tracks object relocations, supersedes outdated
# information, and manages verified reality states.
# ===========================================================================

"""Reality Checksum Service for SecondMind.

Detects conflicts between new observations and stored memories, verifies discrepancies,
tracks relocations, supersedes outdated information, and prevents low-confidence
or temporal anomalies from corrupting verified object locations.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.logger import logger
from backend.time_utils import (
    format_observation_timestamp,
    get_configured_timezone,
    parse_timestamp,
    to_iso_with_tz,
)


class ConflictType(str, Enum):
    """Classification of detected memory discrepancies."""
    LOCATION_CHANGE = "location_change"
    STATIONARY_CONFIRMED = "stationary_confirmed"
    AMBIGUOUS_DISCREPANCY = "ambiguous_discrepancy"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    STATE_CHANGE = "state_change"


class ResolutionStatus(str, Enum):
    """Resolution state for detected discrepancies."""
    RESOLVED_UPDATED = "resolved_updated"
    VERIFIED_CURRENT = "verified_current"
    FLAGGED_UNVERIFIED = "flagged_unverified"
    MANUALLY_RESOLVED = "manually_resolved"
    REJECTED = "rejected"


class RealityChecksumService:
    """Service detecting and resolving conflicts between incoming observations and stored memories."""

    def __init__(self, db_path: Optional[Path] = None, memory_store: Optional[Any] = None) -> None:
        """Initialize RealityChecksumService with SQLite connection.

        Args:
            db_path: Optional custom path to SQLite database.
            memory_store: Optional reference to MemoryStore to mark superseded rows.
        """
        self.db_path = db_path or settings.DATABASE_PATH
        self.memory_store = memory_store
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Create and yield SQLite connection with Row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create reality conflicts and object states tracking tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            # 1. Detected conflicts / discrepancies log
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reality_conflicts (
                    id TEXT PRIMARY KEY,
                    object_name TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    previous_location TEXT,
                    new_location TEXT,
                    previous_observed_at TEXT,
                    new_observed_at TEXT,
                    previous_observation_id TEXT,
                    new_observation_id TEXT,
                    confidence_level TEXT,
                    resolution_status TEXT NOT NULL,
                    resolution_notes TEXT,
                    detected_at TEXT NOT NULL,
                    verified_by TEXT
                )
                """
            )

            # 2. Latest verified reality state for each tracked object
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reality_object_states (
                    object_name TEXT PRIMARY KEY,
                    current_location TEXT NOT NULL,
                    previous_location TEXT,
                    last_observed_at TEXT NOT NULL,
                    last_observation_id TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    evidence_source TEXT DEFAULT 'perception',
                    similarity_score REAL,
                    discrepancy_count INTEGER DEFAULT 0,
                    has_active_conflict INTEGER DEFAULT 0,
                    last_conflict_notes TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )

            # Indexes for fast querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_obj ON reality_conflicts(object_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_status ON reality_conflicts(resolution_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_time ON reality_conflicts(detected_at)")
            conn.commit()

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize object name for deterministic matching."""
        if not name:
            return ""
        s = name.lower().strip()
        for prefix in ["a pair of ", "pair of ", "my ", "the ", "a ", "an "]:
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
        return s

    def process_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Verify an incoming observation against stored memories, detect conflicts, and update states.

        Args:
            obs: Observation dict containing object, location, observed_at, id, etc.

        Returns:
            Dict describing conflict status, resolution, and updated reality state.
        """
        raw_name = obs.get("object", "")
        norm_name = self._normalize_name(raw_name)
        if not norm_name:
            return {"conflict_detected": False, "status": "skipped_no_object"}

        new_loc = (obs.get("location") or "").strip()
        new_obs_id = obs.get("id") or str(uuid.uuid4())
        new_ts_str = obs.get("observed_at") or obs.get("timestamp")
        new_dt = parse_timestamp(new_ts_str, default_tz=get_configured_timezone())
        now_iso = to_iso_with_tz(datetime.now(get_configured_timezone()))
        new_iso = to_iso_with_tz(new_dt) if new_dt else now_iso

        new_verif = obs.get("verification_status") or "unverified"
        new_source = obs.get("evidence_source") or "perception"
        new_sim = obs.get("similarity_score")

        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM reality_object_states WHERE object_name = ?", (norm_name,)
            )
            state_row = cur.fetchone()

        # -------------------------------------------------------------------
        # 1. First observation of this object -> Initial Registration
        # -------------------------------------------------------------------
        if not state_row:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO reality_object_states (
                        object_name, current_location, previous_location, last_observed_at,
                        last_observation_id, verification_status, evidence_source,
                        similarity_score, discrepancy_count, has_active_conflict,
                        last_conflict_notes, updated_at
                    ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, 0, 0, 'Initial observation registered.', ?)
                    """,
                    (
                        norm_name,
                        new_loc,
                        new_iso,
                        new_obs_id,
                        new_verif,
                        new_source,
                        new_sim,
                        now_iso,
                    ),
                )
                conn.commit()

            logger.info("RealityChecksum: Registered initial state for '%s' at '%s'", norm_name, new_loc)
            return {
                "conflict_detected": False,
                "conflict_type": ConflictType.STATIONARY_CONFIRMED.value,
                "resolution_status": ResolutionStatus.VERIFIED_CURRENT.value,
                "object_name": norm_name,
                "current_location": new_loc,
                "previous_location": None,
                "observed_at": new_iso,
                "notes": f"Initial observation registered at {new_loc}.",
            }

        # Existing state exists: inspect for discrepancies
        prev_loc = state_row["current_location"]
        prev_obs_id = state_row["last_observation_id"]
        prev_ts_str = state_row["last_observed_at"]
        prev_dt = parse_timestamp(prev_ts_str, default_tz=get_configured_timezone())
        prev_verif = state_row["verification_status"]
        discrepancy_count = state_row["discrepancy_count"]

        # -------------------------------------------------------------------
        # 2. Check for Temporal Anomaly (New observation is strictly older)
        # -------------------------------------------------------------------
        if prev_dt and new_dt and new_dt < prev_dt:
            conflict_id = str(uuid.uuid4())
            notes = (
                f"Temporal anomaly: Observation timestamp ({new_ts_str}) is older than "
                f"existing verified observation ({prev_ts_str}). Outdated sighting recorded as historical."
            )
            self._record_conflict(
                conflict_id=conflict_id,
                object_name=norm_name,
                conflict_type=ConflictType.TEMPORAL_ANOMALY,
                prev_loc=prev_loc,
                new_loc=new_loc,
                prev_ts=prev_ts_str,
                new_ts=new_iso,
                prev_id=prev_obs_id,
                new_id=new_obs_id,
                resolution_status=ResolutionStatus.REJECTED,
                notes=notes,
                confidence_level="low",
            )
            logger.warning("RealityChecksum: %s", notes)
            return {
                "conflict_detected": True,
                "conflict_type": ConflictType.TEMPORAL_ANOMALY.value,
                "resolution_status": ResolutionStatus.REJECTED.value,
                "object_name": norm_name,
                "current_location": prev_loc,
                "discrepancy_location": new_loc,
                "notes": notes,
                "conflict_id": conflict_id,
            }

        # -------------------------------------------------------------------
        # 3. Check for Checksum Match / Stationary Confirmation
        # -------------------------------------------------------------------
        if prev_loc.lower().strip() == new_loc.lower().strip():
            # Same location -> Verified stationary, extend recency
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE reality_object_states
                    SET last_observed_at = ?,
                        last_observation_id = ?,
                        verification_status = ?,
                        evidence_source = ?,
                        similarity_score = ?,
                        has_active_conflict = 0,
                        updated_at = ?
                    WHERE object_name = ?
                    """,
                    (new_iso, new_obs_id, new_verif, new_source, new_sim, now_iso, norm_name),
                )
                conn.commit()

            logger.info("RealityChecksum: Verified stationary '%s' at '%s'", norm_name, new_loc)
            return {
                "conflict_detected": False,
                "conflict_type": ConflictType.STATIONARY_CONFIRMED.value,
                "resolution_status": ResolutionStatus.VERIFIED_CURRENT.value,
                "object_name": norm_name,
                "current_location": new_loc,
                "previous_location": prev_loc,
                "observed_at": new_iso,
                "notes": f"Location verified unchanged at {new_loc}.",
            }

        # -------------------------------------------------------------------
        # 4. Location Mismatch -> Discrepancy Detected!
        # -------------------------------------------------------------------
        # Check if the new observation is ambiguous / weak against a strong prior
        if new_verif in ["ambiguous_match", "weak_match"] and prev_verif == "strong_match":
            conflict_id = str(uuid.uuid4())
            notes = (
                f"Ambiguous discrepancy: Weak visual sighting ({new_verif}, sim={new_sim}) "
                f"at '{new_loc}' contradicts verified location '{prev_loc}'. Flagged for verification."
            )
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE reality_object_states
                    SET discrepancy_count = discrepancy_count + 1,
                        has_active_conflict = 1,
                        last_conflict_notes = ?,
                        updated_at = ?
                    WHERE object_name = ?
                    """,
                    (notes, now_iso, norm_name),
                )
                conn.commit()

            self._record_conflict(
                conflict_id=conflict_id,
                object_name=norm_name,
                conflict_type=ConflictType.AMBIGUOUS_DISCREPANCY,
                prev_loc=prev_loc,
                new_loc=new_loc,
                prev_ts=prev_ts_str,
                new_ts=new_iso,
                prev_id=prev_obs_id,
                new_id=new_obs_id,
                resolution_status=ResolutionStatus.FLAGGED_UNVERIFIED,
                notes=notes,
                confidence_level="low",
            )
            logger.warning("RealityChecksum: %s", notes)
            return {
                "conflict_detected": True,
                "conflict_type": ConflictType.AMBIGUOUS_DISCREPANCY.value,
                "resolution_status": ResolutionStatus.FLAGGED_UNVERIFIED.value,
                "object_name": norm_name,
                "current_location": prev_loc,
                "discrepancy_location": new_loc,
                "notes": notes,
                "conflict_id": conflict_id,
            }

        # -------------------------------------------------------------------
        # 5. Verified Location Change / Relocation
        # -------------------------------------------------------------------
        # The new observation is valid, newer, and verified -> Supersede outdated memory
        conflict_id = str(uuid.uuid4())
        notes = (
            f"Relocation confirmed: '{norm_name}' moved from '{prev_loc}' to '{new_loc}' "
            f"(observed at {new_ts_str}). Outdated memory superseded."
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE reality_object_states
                SET previous_location = ?,
                    current_location = ?,
                    last_observed_at = ?,
                    last_observation_id = ?,
                    verification_status = ?,
                    evidence_source = ?,
                    similarity_score = ?,
                    has_active_conflict = 0,
                    last_conflict_notes = ?,
                    updated_at = ?
                WHERE object_name = ?
                """,
                (
                    prev_loc,
                    new_loc,
                    new_iso,
                    new_obs_id,
                    new_verif,
                    new_source,
                    new_sim,
                    notes,
                    now_iso,
                    norm_name,
                ),
            )
            conn.commit()

        # Mark previous observation in MemoryStore as superseded
        self._mark_observation_superseded(prev_obs_id, superseded_by_id=new_obs_id)

        self._record_conflict(
            conflict_id=conflict_id,
            object_name=norm_name,
            conflict_type=ConflictType.LOCATION_CHANGE,
            prev_loc=prev_loc,
            new_loc=new_loc,
            prev_ts=prev_ts_str,
            new_ts=new_iso,
            prev_id=prev_obs_id,
            new_id=new_obs_id,
            resolution_status=ResolutionStatus.RESOLVED_UPDATED,
            notes=notes,
            confidence_level="high" if new_verif == "strong_match" else "medium",
        )

        logger.info("RealityChecksum: %s", notes)
        return {
            "conflict_detected": True,
            "conflict_type": ConflictType.LOCATION_CHANGE.value,
            "resolution_status": ResolutionStatus.RESOLVED_UPDATED.value,
            "object_name": norm_name,
            "current_location": new_loc,
            "previous_location": prev_loc,
            "observed_at": new_iso,
            "notes": notes,
            "conflict_id": conflict_id,
        }

    def _record_conflict(
        self,
        conflict_id: str,
        object_name: str,
        conflict_type: ConflictType,
        prev_loc: Optional[str],
        new_loc: str,
        prev_ts: Optional[str],
        new_ts: str,
        prev_id: Optional[str],
        new_id: Optional[str],
        resolution_status: ResolutionStatus,
        notes: str,
        confidence_level: str = "medium",
        verified_by: Optional[str] = None,
    ) -> None:
        """Insert a conflict entry into the reality_conflicts table."""
        now_iso = to_iso_with_tz(datetime.now(get_configured_timezone()))
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO reality_conflicts (
                    id, object_name, conflict_type, previous_location, new_location,
                    previous_observed_at, new_observed_at, previous_observation_id,
                    new_observation_id, confidence_level, resolution_status,
                    resolution_notes, detected_at, verified_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conflict_id,
                    object_name,
                    conflict_type.value,
                    prev_loc,
                    new_loc,
                    prev_ts,
                    new_ts,
                    prev_id,
                    new_id,
                    confidence_level,
                    resolution_status.value,
                    notes,
                    now_iso,
                    verified_by,
                ),
            )
            conn.commit()

    def _mark_observation_superseded(self, obs_id: Optional[str], superseded_by_id: Optional[str]) -> None:
        """Mark an older observation in MemoryStore as superseded to preserve history while flagging outdatedness."""
        if not obs_id:
            return
        try:
            with self._get_connection() as conn:
                # Check if columns exist
                cols = {r["name"] for r in conn.execute("PRAGMA table_info(observations)").fetchall()}
                if "checksum_status" in cols and "superseded_by" in cols:
                    conn.execute(
                        "UPDATE observations SET checksum_status = 'superseded', superseded_by = ? WHERE id = ?",
                        (superseded_by_id, obs_id),
                    )
                    conn.commit()
        except Exception as e:
            logger.debug("Could not mark observation %s superseded: %s", obs_id, e)

    def get_object_state(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve verified reality state for an object."""
        norm = self._normalize_name(object_name)
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM reality_object_states WHERE object_name = ? OR object_name LIKE ?",
                (norm, f"%{norm}%"),
            )
            row = cur.fetchone()
            if row:
                d = dict(row)
                d["formatted_time"] = format_observation_timestamp(d.get("last_observed_at"))
                return d
        return None

    def get_all_states(self) -> List[Dict[str, Any]]:
        """List current reality states for all registered and observed objects."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM reality_object_states ORDER BY updated_at DESC")
            results = []
            for r in cur.fetchall():
                d = dict(r)
                d["formatted_time"] = format_observation_timestamp(d.get("last_observed_at"))
                results.append(d)
            return results

    def get_conflicts(
        self,
        limit: int = 50,
        object_name: Optional[str] = None,
        active_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve conflict history or open discrepancies."""
        query = "SELECT * FROM reality_conflicts"
        params: List[Any] = []
        clauses: List[str] = []

        if object_name:
            norm = self._normalize_name(object_name)
            clauses.append("(object_name = ? OR object_name LIKE ?)")
            params.extend([norm, f"%{norm}%"])

        if active_only:
            clauses.append("resolution_status = ?")
            params.append(ResolutionStatus.FLAGGED_UNVERIFIED.value)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cur = conn.execute(query, tuple(params))
            results = []
            for r in cur.fetchall():
                d = dict(r)
                d["formatted_detected_at"] = format_observation_timestamp(d.get("detected_at"))
                d["formatted_previous_time"] = format_observation_timestamp(d.get("previous_observed_at"))
                d["formatted_new_time"] = format_observation_timestamp(d.get("new_observed_at"))
                results.append(d)
            return results

    def resolve_discrepancy(
        self,
        conflict_id: str,
        verified_location: str,
        verified_by: str = "caregiver",
    ) -> Dict[str, Any]:
        """Manually verify or resolve a flagged discrepancy."""
        now_iso = to_iso_with_tz(datetime.now(get_configured_timezone()))
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM reality_conflicts WHERE id = ?", (conflict_id,))
            conflict = cur.fetchone()
            if not conflict:
                raise ValueError(f"Conflict with ID '{conflict_id}' not found.")

            obj_name = conflict["object_name"]
            notes = f"Manually verified by {verified_by}: Location confirmed as '{verified_location}'."

            # Update conflict record
            conn.execute(
                """
                UPDATE reality_conflicts
                SET resolution_status = ?,
                    resolution_notes = ?,
                    verified_by = ?
                WHERE id = ?
                """,
                (ResolutionStatus.MANUALLY_RESOLVED.value, notes, verified_by, conflict_id),
            )

            # Update object state
            conn.execute(
                """
                UPDATE reality_object_states
                SET current_location = ?,
                    has_active_conflict = 0,
                    last_conflict_notes = ?,
                    updated_at = ?
                WHERE object_name = ?
                """,
                (verified_location, notes, now_iso, obj_name),
            )
            conn.commit()

        logger.info("RealityChecksum: Resolved conflict %s for '%s' to '%s'", conflict_id, obj_name, verified_location)
        return {
            "status": "resolved",
            "conflict_id": conflict_id,
            "object_name": obj_name,
            "verified_location": verified_location,
            "verified_by": verified_by,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary metrics of monitored objects, active discrepancies, and recent relocations."""
        with self._get_connection() as conn:
            total_objects = conn.execute("SELECT COUNT(*) FROM reality_object_states").fetchone()[0]
            active_conflicts = conn.execute(
                "SELECT COUNT(*) FROM reality_conflicts WHERE resolution_status = ?",
                (ResolutionStatus.FLAGGED_UNVERIFIED.value,),
            ).fetchone()[0]
            total_relocations = conn.execute(
                "SELECT COUNT(*) FROM reality_conflicts WHERE conflict_type = ?",
                (ConflictType.LOCATION_CHANGE.value,),
            ).fetchone()[0]

            return {
                "total_monitored_objects": total_objects,
                "active_unverified_conflicts": active_conflicts,
                "total_relocations_detected": total_relocations,
            }

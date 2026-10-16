# ===========================================================================
# [NEW FILE - BACKUP FOR TEAMMATE & ANTIGRAVITY CONTEXT]
# Feature: Feature 1: Image Embeddings & Object Registry
# Relative Path: backend/stores/object_store.py
# Summary: SQLite store for registered objects, reference image paths, 576-dim MobileNetV3 visual embeddings (BLOB storage), and visual match audit logging.
# Integration: Part of SecondMind Image Embeddings & Accurate Timestamps features.
# If your teammate's code fails or lacks this file, drop this directly into place.
# ===========================================================================

"""Object registry and reference-image embedding repository using SQLite.

Manages registered objects, binary serialized embedding vectors, reference metadata,
and historical visual detection matches in second_mind.db.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime, timezone
import sqlite3
import numpy as np

from backend.config import settings
from backend.logger import logger
from backend.time_utils import (
    format_observation_timestamp,
    get_configured_timezone,
    to_iso_with_tz,
)


class ObjectStore:
    """SQLite repository for registered physical objects and reference image embeddings."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or settings.DATABASE_PATH
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Yield SQLite connection configured with row factory and foreign keys."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables and indexes for object registry and reference embeddings."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS registered_objects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_images (
                    id TEXT PRIMARY KEY,
                    object_id TEXT NOT NULL,
                    encoder_model TEXT NOT NULL,
                    encoder_version TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimensions INTEGER NOT NULL,
                    image_path TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (object_id) REFERENCES registered_objects(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS visual_matches (
                    id TEXT PRIMARY KEY,
                    frame_timestamp TEXT NOT NULL,
                    registered_object_id TEXT NOT NULL,
                    object_name TEXT NOT NULL,
                    bounding_box TEXT NOT NULL,
                    location_name TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    local_matches INTEGER DEFAULT 0,
                    match_status TEXT NOT NULL,
                    diagnostics TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ref_object_id ON reference_images(object_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vis_matches_time ON visual_matches(frame_timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vis_matches_obj ON visual_matches(registered_object_id)")
            conn.commit()

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """Safely serialize 1D float32 numpy vector to binary bytes."""
        arr = np.ascontiguousarray(embedding, dtype=np.float32)
        return arr.tobytes()

    @staticmethod
    def deserialize_embedding(blob: bytes, expected_dim: Optional[int] = None) -> np.ndarray:
        """Safely deserialize binary bytes back to 1D float32 numpy vector."""
        arr = np.frombuffer(blob, dtype=np.float32)
        if expected_dim is not None and len(arr) != expected_dim:
            raise ValueError(f"Corrupted embedding: expected {expected_dim} dims, got {len(arr)}")
        return arr.copy()

    def add_object(
        self,
        name: str,
        category: str = "personal",
        description: str = "",
        object_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new important object."""
        oid = object_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO registered_objects (id, name, category, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (oid, name.strip(), category.strip(), description.strip(), now, now),
            )
            conn.commit()

        logger.info("ObjectStore: Registered object '%s' (%s, id=%s)", name, category, oid)
        return {
            "id": oid,
            "name": name.strip(),
            "category": category.strip(),
            "description": description.strip(),
            "created_at": now,
            "updated_at": now,
            "references": [],
        }

    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a registered object with its associated reference image metadata."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM registered_objects WHERE id = ?", (object_id,))
            row = cursor.fetchone()
            if not row:
                return None
            obj_data = dict(row)

            # Fetch references
            ref_cursor = conn.execute(
                """
                SELECT id, object_id, encoder_model, encoder_version, dimensions, image_path, metadata, created_at
                FROM reference_images WHERE object_id = ? ORDER BY created_at ASC
                """,
                (object_id,),
            )
            refs = []
            for r in ref_cursor.fetchall():
                r_dict = dict(r)
                if r_dict.get("metadata"):
                    try:
                        r_dict["metadata"] = json.loads(r_dict["metadata"])
                    except Exception:
                        pass
                refs.append(r_dict)
            obj_data["references"] = refs
            return obj_data

    def list_objects(self) -> List[Dict[str, Any]]:
        """List all registered objects with summary reference counts."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM registered_objects ORDER BY created_at ASC")
            objects = []
            for row in cursor.fetchall():
                obj_data = dict(row)
                ref_count_cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM reference_images WHERE object_id = ?",
                    (obj_data["id"],),
                )
                obj_data["reference_count"] = ref_count_cursor.fetchone()["cnt"]
                objects.append(obj_data)
            return objects

    def delete_object(self, object_id: str) -> bool:
        """Delete registered object and cascade-remove references and local image files."""
        with self._get_connection() as conn:
            # Query image paths first to remove files
            cursor = conn.execute("SELECT image_path FROM reference_images WHERE object_id = ?", (object_id,))
            file_paths = [r["image_path"] for r in cursor.fetchall() if r["image_path"]]

            del_cursor = conn.execute("DELETE FROM registered_objects WHERE id = ?", (object_id,))
            conn.commit()
            deleted = del_cursor.rowcount > 0

        if deleted:
            for fp in file_paths:
                try:
                    p = Path(fp)
                    if p.exists() and p.is_file():
                        p.unlink()
                except Exception as e:
                    logger.warning("ObjectStore: Could not delete image file '%s': %s", fp, e)
            logger.info("ObjectStore: Deleted object '%s' and associated references", object_id)

        return deleted

    def add_reference_image(
        self,
        object_id: str,
        encoder_model: str,
        encoder_version: str,
        embedding: np.ndarray,
        image_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reference_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save a new reference image embedding for a registered object."""
        ref_id = reference_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        blob = self.serialize_embedding(embedding)
        meta_json = json.dumps(metadata or {})
        dims = int(len(embedding))

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO reference_images (
                    id, object_id, encoder_model, encoder_version, embedding, dimensions, image_path, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ref_id, object_id, encoder_model, encoder_version, blob, dims, image_path, meta_json, now),
            )
            # Update updated_at in parent object
            conn.execute("UPDATE registered_objects SET updated_at = ? WHERE id = ?", (now, object_id))
            conn.commit()

        logger.info(
            "ObjectStore: Added reference %s (model=%s, dims=%d) for object %s",
            ref_id,
            encoder_model,
            dims,
            object_id,
        )
        return {
            "id": ref_id,
            "object_id": object_id,
            "encoder_model": encoder_model,
            "encoder_version": encoder_version,
            "dimensions": dims,
            "image_path": image_path,
            "metadata": metadata or {},
            "created_at": now,
        }

    def get_all_active_references(self) -> List[Dict[str, Any]]:
        """Retrieve all reference embeddings across all objects with deserialized vectors."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT r.id as reference_id, r.object_id, o.name as object_name, o.category,
                       r.encoder_model, r.dimensions, r.embedding, r.image_path, r.metadata
                FROM reference_images r
                JOIN registered_objects o ON r.object_id = o.id
                ORDER BY r.created_at ASC
                """
            )
            results = []
            for row in cursor.fetchall():
                r_dict = dict(row)
                embedding_vec = self.deserialize_embedding(r_dict.pop("embedding"), expected_dim=r_dict["dimensions"])
                r_dict["embedding"] = embedding_vec
                if r_dict.get("metadata"):
                    try:
                        r_dict["metadata"] = json.loads(r_dict["metadata"])
                    except Exception:
                        pass
                results.append(r_dict)
            return results

    def delete_reference_image(self, reference_id: str) -> bool:
        """Delete an individual reference image and its local file."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT image_path FROM reference_images WHERE id = ?", (reference_id,))
            row = cursor.fetchone()
            if not row:
                return False
            img_path = row["image_path"]

            del_cursor = conn.execute("DELETE FROM reference_images WHERE id = ?", (reference_id,))
            conn.commit()
            deleted = del_cursor.rowcount > 0

        if deleted and img_path:
            try:
                p = Path(img_path)
                if p.exists() and p.is_file():
                    p.unlink()
            except Exception as e:
                logger.warning("ObjectStore: Could not delete image file '%s': %s", img_path, e)

        return deleted

    def record_visual_match(
        self,
        frame_timestamp: str,
        registered_object_id: str,
        object_name: str,
        bounding_box: Dict[str, Any],
        location_name: str,
        similarity_score: float,
        match_status: str,
        local_matches: int = 0,
        diagnostics: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a visual search match event for diagnostics and audit logging."""
        mid = str(uuid.uuid4())
        now = created_at or to_iso_with_tz(datetime.now(get_configured_timezone())) or datetime.now(timezone.utc).isoformat()
        observed_at = to_iso_with_tz(frame_timestamp) or frame_timestamp
        formatted_time = format_observation_timestamp(frame_timestamp)
        bbox_json = json.dumps(bounding_box)
        diag_json = json.dumps(diagnostics or {})

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO visual_matches (
                    id, frame_timestamp, registered_object_id, object_name, bounding_box,
                    location_name, similarity_score, local_matches, match_status, diagnostics, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid,
                    frame_timestamp,
                    registered_object_id,
                    object_name,
                    bbox_json,
                    location_name,
                    float(similarity_score),
                    int(local_matches),
                    match_status,
                    diag_json,
                    now,
                ),
            )
            conn.commit()

        return {
            "id": mid,
            "frame_timestamp": frame_timestamp,
            "observed_at": observed_at,
            "formatted_time": formatted_time,
            "registered_object_id": registered_object_id,
            "object_name": object_name,
            "bounding_box": bounding_box,
            "location_name": location_name,
            "similarity_score": similarity_score,
            "local_matches": local_matches,
            "match_status": match_status,
            "diagnostics": diagnostics or {},
            "created_at": now,
        }

    def get_recent_visual_matches(
        self, limit: int = 50, object_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query recent visual matches."""
        with self._get_connection() as conn:
            if object_id:
                cursor = conn.execute(
                    """
                    SELECT * FROM visual_matches WHERE registered_object_id = ?
                    ORDER BY frame_timestamp DESC LIMIT ?
                    """,
                    (object_id, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM visual_matches ORDER BY frame_timestamp DESC LIMIT ?",
                    (limit,),
                )
            matches = []
            for row in cursor.fetchall():
                m_dict = dict(row)
                try:
                    m_dict["bounding_box"] = json.loads(m_dict["bounding_box"])
                except Exception:
                    pass
                try:
                    m_dict["diagnostics"] = json.loads(m_dict["diagnostics"])
                except Exception:
                    pass
                f_ts = m_dict.get("frame_timestamp")
                m_dict["observed_at"] = to_iso_with_tz(f_ts) or f_ts
                m_dict["formatted_time"] = format_observation_timestamp(f_ts)
                matches.append(m_dict)
            return matches

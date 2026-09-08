"""
TrueSight — result storage.

Backs GET /result/{id}: analysis is synchronous (the prototype's scale
doesn't need a job queue), but results are still persisted so a client
can re-fetch them later without re-running inference. Uses plain
sqlite3 (stdlib) rather than the SQLAlchemy ORM listed in the tech
stack — a single results table with JSON payloads doesn't need an ORM's
relational modeling, and stdlib sqlite3 keeps this dependency-free.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("backend/truesight_results.db")


def _get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            payload TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_result(result_id: str, payload: dict, db_path: Path = DEFAULT_DB_PATH) -> None:
    conn = _get_connection(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO results (id, payload) VALUES (?, ?)",
            (result_id, json.dumps(payload)),
        )
        conn.commit()
    finally:
        conn.close()


def load_result(result_id: str, db_path: Path = DEFAULT_DB_PATH) -> dict | None:
    conn = _get_connection(db_path)
    try:
        row = conn.execute("SELECT payload FROM results WHERE id = ?", (result_id,)).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        conn.close()

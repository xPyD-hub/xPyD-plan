"""Benchmark session manager with SQLite-backed storage.

Organize benchmark files into named sessions with metadata,
enabling logical grouping of related experiments.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from pydantic import BaseModel, Field

from xpyd_plan.bench_adapter import load_benchmark_auto


class SessionEntry(BaseModel):
    """A benchmark file within a session."""

    id: int | None = Field(None, description="Auto-assigned row ID")
    session_id: int = Field(..., description="Parent session ID")
    benchmark_path: str = Field(..., description="Path to benchmark JSON file")
    added_at: float = Field(..., description="Unix timestamp when added")
    num_requests: int = Field(..., ge=0, description="Number of requests in benchmark")
    measured_qps: float = Field(..., ge=0, description="Measured QPS")
    num_prefill: int = Field(..., ge=1, description="Prefill instance count")
    num_decode: int = Field(..., ge=1, description="Decode instance count")


class Session(BaseModel):
    """A named benchmark session."""

    id: int | None = Field(None, description="Auto-assigned session ID")
    name: str = Field(..., description="Session name")
    description: str = Field("", description="Optional description")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")
    created_at: float = Field(..., description="Unix timestamp when created")
    entries: list[SessionEntry] = Field(default_factory=list, description="Benchmark entries")


class SessionReport(BaseModel):
    """Summary report of sessions."""

    sessions: list[Session] = Field(default_factory=list)
    total_sessions: int = Field(0, description="Total number of sessions")
    total_benchmarks: int = Field(0, description="Total benchmark files across all sessions")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS session_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    benchmark_path TEXT NOT NULL,
    added_at REAL NOT NULL,
    num_requests INTEGER NOT NULL,
    measured_qps REAL NOT NULL,
    num_prefill INTEGER NOT NULL,
    num_decode INTEGER NOT NULL
);
"""


class SessionManager:
    """Manage benchmark sessions with SQLite storage."""

    def __init__(self, db_path: str | Path = "xpyd-plan-sessions.db") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def create(
        self,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> Session:
        """Create a new session.

        Args:
            name: Unique session name.
            description: Optional description.
            tags: Optional tags for filtering.

        Returns:
            The created Session.

        Raises:
            ValueError: If a session with the same name already exists.
        """
        tags = tags or []
        now = time.time()
        try:
            cur = self._conn.execute(
                "INSERT INTO sessions (name, description, tags, created_at) VALUES (?, ?, ?, ?)",
                (name, description, json.dumps(tags), now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Session '{name}' already exists")
        return Session(
            id=cur.lastrowid,
            name=name,
            description=description,
            tags=tags,
            created_at=now,
        )

    def add(self, session_name: str, benchmark_path: str | Path) -> SessionEntry:
        """Add a benchmark file to a session.

        Args:
            session_name: Name of the session.
            benchmark_path: Path to benchmark JSON file.

        Returns:
            The created SessionEntry.

        Raises:
            ValueError: If session not found or file already in session.
        """
        session_id = self._get_session_id(session_name)
        path_str = str(Path(benchmark_path).resolve())

        # Check for duplicate
        row = self._conn.execute(
            "SELECT id FROM session_entries WHERE session_id = ? AND benchmark_path = ?",
            (session_id, path_str),
        ).fetchone()
        if row is not None:
            raise ValueError(f"Benchmark '{path_str}' already in session '{session_name}'")

        # Load and extract metadata
        data = load_benchmark_auto(path_str)
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO session_entries "
            "(session_id, benchmark_path, added_at, num_requests, measured_qps, "
            "num_prefill, num_decode) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                path_str,
                now,
                len(data.requests),
                data.metadata.measured_qps,
                data.metadata.num_prefill_instances,
                data.metadata.num_decode_instances,
            ),
        )
        self._conn.commit()
        return SessionEntry(
            id=cur.lastrowid,
            session_id=session_id,
            benchmark_path=path_str,
            added_at=now,
            num_requests=len(data.requests),
            measured_qps=data.metadata.measured_qps,
            num_prefill=data.metadata.num_prefill_instances,
            num_decode=data.metadata.num_decode_instances,
        )

    def remove(self, session_name: str, benchmark_path: str | Path) -> None:
        """Remove a benchmark file from a session.

        Args:
            session_name: Name of the session.
            benchmark_path: Path to benchmark JSON file.

        Raises:
            ValueError: If session or entry not found.
        """
        session_id = self._get_session_id(session_name)
        path_str = str(Path(benchmark_path).resolve())
        cur = self._conn.execute(
            "DELETE FROM session_entries WHERE session_id = ? AND benchmark_path = ?",
            (session_id, path_str),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise ValueError(f"Benchmark '{path_str}' not found in session '{session_name}'")

    def list_sessions(self) -> list[Session]:
        """List all sessions with their entries.

        Returns:
            List of Session objects.
        """
        rows = self._conn.execute(
            "SELECT id, name, description, tags, created_at FROM sessions ORDER BY created_at"
        ).fetchall()
        sessions = []
        for row in rows:
            session_id, name, description, tags_json, created_at = row
            entries = self._get_entries(session_id)
            sessions.append(
                Session(
                    id=session_id,
                    name=name,
                    description=description,
                    tags=json.loads(tags_json),
                    created_at=created_at,
                    entries=entries,
                )
            )
        return sessions

    def show(self, session_name: str) -> Session:
        """Show a single session with its entries.

        Args:
            session_name: Name of the session.

        Returns:
            Session with entries populated.

        Raises:
            ValueError: If session not found.
        """
        session_id = self._get_session_id(session_name)
        row = self._conn.execute(
            "SELECT id, name, description, tags, created_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        entries = self._get_entries(session_id)
        return Session(
            id=row[0],
            name=row[1],
            description=row[2],
            tags=json.loads(row[3]),
            created_at=row[4],
            entries=entries,
        )

    def delete(self, session_name: str) -> None:
        """Delete a session and all its entries.

        Args:
            session_name: Name of the session.

        Raises:
            ValueError: If session not found.
        """
        session_id = self._get_session_id(session_name)
        self._conn.execute("DELETE FROM session_entries WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    def _get_session_id(self, name: str) -> int:
        """Get session ID by name, raise if not found."""
        row = self._conn.execute(
            "SELECT id FROM sessions WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Session '{name}' not found")
        return row[0]

    def _get_entries(self, session_id: int) -> list[SessionEntry]:
        """Get all entries for a session."""
        rows = self._conn.execute(
            "SELECT id, session_id, benchmark_path, added_at, num_requests, "
            "measured_qps, num_prefill, num_decode "
            "FROM session_entries WHERE session_id = ? ORDER BY added_at",
            (session_id,),
        ).fetchall()
        return [
            SessionEntry(
                id=r[0],
                session_id=r[1],
                benchmark_path=r[2],
                added_at=r[3],
                num_requests=r[4],
                measured_qps=r[5],
                num_prefill=r[6],
                num_decode=r[7],
            )
            for r in rows
        ]


def manage_session(
    action: str,
    db_path: str | Path = "xpyd-plan-sessions.db",
    name: str | None = None,
    description: str = "",
    tags: list[str] | None = None,
    benchmark_path: str | Path | None = None,
) -> Session | list[Session] | SessionReport | None:
    """Programmatic API for session management.

    Args:
        action: One of 'create', 'add', 'remove', 'list', 'show', 'delete'.
        db_path: Path to SQLite database.
        name: Session name (required for all actions except 'list').
        description: Session description (for 'create').
        tags: Session tags (for 'create').
        benchmark_path: Benchmark file path (for 'add' and 'remove').

    Returns:
        Depends on action:
        - create: Session
        - add: Session (updated)
        - remove: None
        - list: SessionReport
        - show: Session
        - delete: None
    """
    mgr = SessionManager(db_path=db_path)
    try:
        if action == "create":
            if name is None:
                raise ValueError("name is required for 'create'")
            return mgr.create(name=name, description=description, tags=tags)
        elif action == "add":
            if name is None or benchmark_path is None:
                raise ValueError("name and benchmark_path are required for 'add'")
            mgr.add(session_name=name, benchmark_path=benchmark_path)
            return mgr.show(name)
        elif action == "remove":
            if name is None or benchmark_path is None:
                raise ValueError("name and benchmark_path are required for 'remove'")
            mgr.remove(session_name=name, benchmark_path=benchmark_path)
            return None
        elif action == "list":
            sessions = mgr.list_sessions()
            total_benchmarks = sum(len(s.entries) for s in sessions)
            return SessionReport(
                sessions=sessions,
                total_sessions=len(sessions),
                total_benchmarks=total_benchmarks,
            )
        elif action == "show":
            if name is None:
                raise ValueError("name is required for 'show'")
            return mgr.show(name)
        elif action == "delete":
            if name is None:
                raise ValueError("name is required for 'delete'")
            mgr.delete(name)
            return None
        else:
            raise ValueError(f"Unknown action: {action}")
    finally:
        mgr.close()

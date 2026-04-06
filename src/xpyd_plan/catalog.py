"""Benchmark Dataset Catalog — SQLite-backed local index for benchmark files."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class CatalogEntry(BaseModel):
    """A single indexed benchmark file."""

    id: int = 0
    file_path: str
    file_hash: str
    gpu_type: str = ""
    model_name: str = ""
    prefill_instances: int = 0
    decode_instances: int = 0
    total_instances: int = 0
    pd_ratio: str = ""
    measured_qps: float = 0.0
    request_count: int = 0
    date_added: str = ""
    notes: str = ""


class CatalogQuery(BaseModel):
    """Query filters for catalog search."""

    gpu_type: Optional[str] = None
    model_name: Optional[str] = None
    min_qps: Optional[float] = None
    max_qps: Optional[float] = None
    pd_ratio: Optional[str] = None
    min_instances: Optional[int] = None
    max_instances: Optional[int] = None
    added_after: Optional[str] = None
    added_before: Optional[str] = None


class CatalogReport(BaseModel):
    """Result of a catalog operation."""

    entries: list[CatalogEntry] = Field(default_factory=list)
    total_count: int = 0
    message: str = ""


class DatasetCatalog:
    """SQLite-backed catalog for indexing benchmark files."""

    def __init__(self, db_path: str = "~/.xpyd-plan/catalog.db") -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                gpu_type TEXT DEFAULT '',
                model_name TEXT DEFAULT '',
                prefill_instances INTEGER DEFAULT 0,
                decode_instances INTEGER DEFAULT 0,
                total_instances INTEGER DEFAULT 0,
                pd_ratio TEXT DEFAULT '',
                measured_qps REAL DEFAULT 0.0,
                request_count INTEGER DEFAULT 0,
                date_added TEXT NOT NULL,
                notes TEXT DEFAULT ''
            )
        """)
        # Indexes for common queries
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_gpu_type ON catalog(gpu_type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pd_ratio ON catalog(pd_ratio)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_measured_qps ON catalog(measured_qps)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_date_added ON catalog(date_added)"
        )
        self._conn.commit()

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _extract_metadata(path: Path) -> dict:
        """Extract metadata from a benchmark JSON file."""
        with open(path) as f:
            data = json.load(f)

        config = data.get("config", data.get("cluster_config", {}))
        metadata = data.get("metadata", {})
        requests = data.get("requests", [])

        prefill = config.get("num_prefill_instances", 0)
        decode = config.get("num_decode_instances", 0)
        total = config.get("total_instances", prefill + decode)

        if prefill > 0 and decode > 0:
            pd_ratio = f"{prefill}:{decode}"
        else:
            pd_ratio = ""

        return {
            "gpu_type": metadata.get("gpu_type", config.get("gpu_type", "")),
            "model_name": metadata.get("model_name", config.get("model_name", "")),
            "prefill_instances": prefill,
            "decode_instances": decode,
            "total_instances": total,
            "pd_ratio": pd_ratio,
            "measured_qps": float(data.get("measured_qps", 0.0)),
            "request_count": len(requests),
        }

    def add(self, file_path: str, notes: str = "") -> CatalogEntry:
        """Add a benchmark file to the catalog. Raises ValueError on duplicate."""
        path = Path(file_path).resolve()
        if not path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        fhash = self._file_hash(path)

        # Check duplicate
        row = self._conn.execute(
            "SELECT id FROM catalog WHERE file_hash = ?", (fhash,)
        ).fetchone()
        if row:
            msg = f"Duplicate file (hash={fhash[:12]}...) already in catalog as id={row['id']}"
            raise ValueError(msg)

        meta = self._extract_metadata(path)
        now = datetime.now(timezone.utc).isoformat()

        cursor = self._conn.execute(
            """INSERT INTO catalog
               (file_path, file_hash, gpu_type, model_name,
                prefill_instances, decode_instances, total_instances,
                pd_ratio, measured_qps, request_count, date_added, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(path),
                fhash,
                meta["gpu_type"],
                meta["model_name"],
                meta["prefill_instances"],
                meta["decode_instances"],
                meta["total_instances"],
                meta["pd_ratio"],
                meta["measured_qps"],
                meta["request_count"],
                now,
                notes,
            ),
        )
        self._conn.commit()

        return CatalogEntry(
            id=cursor.lastrowid or 0,
            file_path=str(path),
            file_hash=fhash,
            date_added=now,
            notes=notes,
            **meta,
        )

    def remove(self, entry_id: int) -> bool:
        """Remove an entry by ID. Returns True if removed."""
        cursor = self._conn.execute("DELETE FROM catalog WHERE id = ?", (entry_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def get(self, entry_id: int) -> Optional[CatalogEntry]:
        """Get a single entry by ID."""
        row = self._conn.execute(
            "SELECT * FROM catalog WHERE id = ?", (entry_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_entry(row)

    def list_all(self) -> CatalogReport:
        """List all entries."""
        rows = self._conn.execute(
            "SELECT * FROM catalog ORDER BY date_added DESC"
        ).fetchall()
        entries = [self._row_to_entry(r) for r in rows]
        return CatalogReport(
            entries=entries, total_count=len(entries), message="All catalog entries"
        )

    def search(self, query: CatalogQuery) -> CatalogReport:
        """Search with filters."""
        conditions: list[str] = []
        params: list = []

        if query.gpu_type:
            conditions.append("gpu_type = ?")
            params.append(query.gpu_type)
        if query.model_name:
            conditions.append("model_name = ?")
            params.append(query.model_name)
        if query.min_qps is not None:
            conditions.append("measured_qps >= ?")
            params.append(query.min_qps)
        if query.max_qps is not None:
            conditions.append("measured_qps <= ?")
            params.append(query.max_qps)
        if query.pd_ratio:
            conditions.append("pd_ratio = ?")
            params.append(query.pd_ratio)
        if query.min_instances is not None:
            conditions.append("total_instances >= ?")
            params.append(query.min_instances)
        if query.max_instances is not None:
            conditions.append("total_instances <= ?")
            params.append(query.max_instances)
        if query.added_after:
            conditions.append("date_added >= ?")
            params.append(query.added_after)
        if query.added_before:
            conditions.append("date_added <= ?")
            params.append(query.added_before)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM catalog WHERE {where} ORDER BY date_added DESC"  # noqa: S608
        rows = self._conn.execute(sql, params).fetchall()
        entries = [self._row_to_entry(r) for r in rows]
        return CatalogReport(
            entries=entries,
            total_count=len(entries),
            message=f"Found {len(entries)} matching entries",
        )

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> CatalogEntry:
        return CatalogEntry(
            id=row["id"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            gpu_type=row["gpu_type"],
            model_name=row["model_name"],
            prefill_instances=row["prefill_instances"],
            decode_instances=row["decode_instances"],
            total_instances=row["total_instances"],
            pd_ratio=row["pd_ratio"],
            measured_qps=row["measured_qps"],
            request_count=row["request_count"],
            date_added=row["date_added"],
            notes=row["notes"],
        )


def manage_catalog(
    action: str,
    db_path: str = "~/.xpyd-plan/catalog.db",
    file_path: str = "",
    entry_id: int = 0,
    query: Optional[CatalogQuery] = None,
    notes: str = "",
) -> CatalogReport:
    """Programmatic API for catalog management."""
    catalog = DatasetCatalog(db_path=db_path)
    try:
        if action == "add":
            entry = catalog.add(file_path, notes=notes)
            return CatalogReport(
                entries=[entry], total_count=1, message="Added to catalog"
            )
        elif action == "list":
            return catalog.list_all()
        elif action == "search":
            return catalog.search(query or CatalogQuery())
        elif action == "remove":
            removed = catalog.remove(entry_id)
            msg = f"Removed entry {entry_id}" if removed else f"Entry {entry_id} not found"
            return CatalogReport(entries=[], total_count=0, message=msg)
        elif action == "show":
            entry = catalog.get(entry_id)
            if entry:
                return CatalogReport(
                    entries=[entry], total_count=1, message=f"Entry {entry_id}"
                )
            return CatalogReport(
                entries=[], total_count=0, message=f"Entry {entry_id} not found"
            )
        else:
            msg = f"Unknown action: {action}"
            raise ValueError(msg)
    finally:
        catalog.close()

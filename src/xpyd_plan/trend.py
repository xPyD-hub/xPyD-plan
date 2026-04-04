"""Historical trend tracking with SQLite-backed storage.

Persist benchmark analysis results over time and detect gradual
performance degradation across runs.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class TrendEntry(BaseModel):
    """A single historical benchmark entry."""

    id: int | None = Field(None, description="Auto-assigned row ID")
    label: str = Field(..., description="Run label/tag (e.g. 'v1.2.0', 'nightly-2026-04-04')")
    timestamp: float = Field(..., description="Unix timestamp when entry was added")
    measured_qps: float = Field(..., description="Measured QPS")
    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    ttft_p50_ms: float = Field(..., ge=0)
    ttft_p95_ms: float = Field(..., ge=0)
    ttft_p99_ms: float = Field(..., ge=0)
    tpot_p50_ms: float = Field(..., ge=0)
    tpot_p95_ms: float = Field(..., ge=0)
    tpot_p99_ms: float = Field(..., ge=0)
    total_latency_p50_ms: float = Field(..., ge=0)
    total_latency_p95_ms: float = Field(..., ge=0)
    total_latency_p99_ms: float = Field(..., ge=0)
    num_requests: int = Field(..., ge=1)


class DegradationAlert(BaseModel):
    """Alert for a metric showing degradation trend."""

    metric: str = Field(..., description="Metric name")
    slope_per_run: float = Field(..., description="Average increase per run (ms)")
    total_change_pct: float = Field(
        ..., description="Total relative change from first to last entry"
    )
    is_degrading: bool = Field(..., description="True if degradation exceeds threshold")


class TrendReport(BaseModel):
    """Report from trend analysis."""

    entries: list[TrendEntry] = Field(default_factory=list)
    lookback_count: int = Field(..., description="Number of entries analyzed")
    alerts: list[DegradationAlert] = Field(default_factory=list)
    has_degradation: bool = Field(False, description="True if any metric is degrading")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trend_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    timestamp REAL NOT NULL,
    measured_qps REAL NOT NULL,
    num_prefill INTEGER NOT NULL,
    num_decode INTEGER NOT NULL,
    ttft_p50_ms REAL NOT NULL,
    ttft_p95_ms REAL NOT NULL,
    ttft_p99_ms REAL NOT NULL,
    tpot_p50_ms REAL NOT NULL,
    tpot_p95_ms REAL NOT NULL,
    tpot_p99_ms REAL NOT NULL,
    total_latency_p50_ms REAL NOT NULL,
    total_latency_p95_ms REAL NOT NULL,
    total_latency_p99_ms REAL NOT NULL,
    num_requests INTEGER NOT NULL
);
"""

_LATENCY_METRICS = [
    "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
    "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms",
    "total_latency_p50_ms", "total_latency_p95_ms", "total_latency_p99_ms",
]


class TrendTracker:
    """Track benchmark performance trends over time.

    Args:
        db_path: Path to SQLite database file. Use ":memory:" for in-memory.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def add(self, data: BenchmarkData, label: str, timestamp: float | None = None) -> TrendEntry:
        """Add a benchmark result to the trend database.

        Args:
            data: Benchmark data to record.
            label: Run label/tag.
            timestamp: Unix timestamp (default: current time).

        Returns:
            The created TrendEntry with assigned ID.
        """
        if timestamp is None:
            timestamp = time.time()

        ttft = [r.ttft_ms for r in data.requests]
        tpot = [r.tpot_ms for r in data.requests]
        total = [r.total_latency_ms for r in data.requests]

        entry = TrendEntry(
            label=label,
            timestamp=timestamp,
            measured_qps=data.metadata.measured_qps,
            num_prefill=data.metadata.num_prefill_instances,
            num_decode=data.metadata.num_decode_instances,
            ttft_p50_ms=float(np.percentile(ttft, 50)),
            ttft_p95_ms=float(np.percentile(ttft, 95)),
            ttft_p99_ms=float(np.percentile(ttft, 99)),
            tpot_p50_ms=float(np.percentile(tpot, 50)),
            tpot_p95_ms=float(np.percentile(tpot, 95)),
            tpot_p99_ms=float(np.percentile(tpot, 99)),
            total_latency_p50_ms=float(np.percentile(total, 50)),
            total_latency_p95_ms=float(np.percentile(total, 95)),
            total_latency_p99_ms=float(np.percentile(total, 99)),
            num_requests=len(data.requests),
        )

        cursor = self._conn.execute(
            """INSERT INTO trend_entries
               (label, timestamp, measured_qps, num_prefill, num_decode,
                ttft_p50_ms, ttft_p95_ms, ttft_p99_ms,
                tpot_p50_ms, tpot_p95_ms, tpot_p99_ms,
                total_latency_p50_ms, total_latency_p95_ms, total_latency_p99_ms,
                num_requests)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.label, entry.timestamp, entry.measured_qps,
                entry.num_prefill, entry.num_decode,
                entry.ttft_p50_ms, entry.ttft_p95_ms, entry.ttft_p99_ms,
                entry.tpot_p50_ms, entry.tpot_p95_ms, entry.tpot_p99_ms,
                entry.total_latency_p50_ms, entry.total_latency_p95_ms,
                entry.total_latency_p99_ms, entry.num_requests,
            ),
        )
        self._conn.commit()
        entry.id = cursor.lastrowid
        return entry

    def list_entries(self, limit: int | None = None) -> list[TrendEntry]:
        """List entries ordered by timestamp ascending.

        Args:
            limit: Max entries to return (most recent). None = all.
        """
        if limit is not None:
            rows = self._conn.execute(
                "SELECT * FROM trend_entries ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = self._conn.execute(
                "SELECT * FROM trend_entries ORDER BY timestamp ASC"
            ).fetchall()
        return [TrendEntry(**dict(row)) for row in rows]

    def check(
        self,
        lookback: int = 10,
        threshold: float = 0.1,
    ) -> TrendReport:
        """Check for performance degradation trends.

        Uses linear regression on the most recent `lookback` entries.
        A metric is flagged as degrading if the total relative increase
        from first to last predicted value exceeds `threshold`.

        Args:
            lookback: Number of most recent entries to analyze.
            threshold: Relative change threshold to flag degradation (default 0.1 = 10%).

        Returns:
            TrendReport with degradation alerts.
        """
        entries = self.list_entries(limit=lookback)
        if len(entries) < 2:
            return TrendReport(
                entries=entries,
                lookback_count=len(entries),
                alerts=[],
                has_degradation=False,
            )

        alerts: list[DegradationAlert] = []
        n = len(entries)
        x = np.arange(n, dtype=float)

        for metric in _LATENCY_METRICS:
            values = np.array([getattr(e, metric) for e in entries])
            # Linear regression: y = mx + b
            coeffs = np.polyfit(x, values, 1)
            slope = coeffs[0]
            first_val = coeffs[1]  # predicted value at x=0
            last_val = coeffs[0] * (n - 1) + coeffs[1]  # predicted at x=n-1

            if first_val > 0:
                total_change_pct = (last_val - first_val) / first_val
            else:
                total_change_pct = 0.0

            is_degrading = total_change_pct > threshold

            alerts.append(
                DegradationAlert(
                    metric=metric,
                    slope_per_run=float(slope),
                    total_change_pct=float(total_change_pct),
                    is_degrading=is_degrading,
                )
            )

        has_degradation = any(a.is_degrading for a in alerts)

        return TrendReport(
            entries=entries,
            lookback_count=n,
            alerts=alerts,
            has_degradation=has_degradation,
        )

    def count(self) -> int:
        """Return total number of entries."""
        row = self._conn.execute("SELECT COUNT(*) FROM trend_entries").fetchone()
        return row[0]


def track_trend(
    benchmark_path: str | Path,
    label: str,
    db_path: str | Path = "xpyd-plan-trend.db",
    lookback: int = 10,
    threshold: float = 0.1,
) -> TrendReport:
    """Programmatic API: add a benchmark and check for trends.

    Args:
        benchmark_path: Path to benchmark JSON file.
        label: Run label/tag.
        db_path: Path to SQLite database.
        lookback: Number of recent entries to analyze.
        threshold: Degradation threshold.

    Returns:
        TrendReport after adding the new entry.
    """
    raw = json.loads(Path(benchmark_path).read_text())
    data = BenchmarkData(**raw)
    tracker = TrendTracker(db_path=db_path)
    try:
        tracker.add(data, label=label)
        return tracker.check(lookback=lookback, threshold=threshold)
    finally:
        tracker.close()

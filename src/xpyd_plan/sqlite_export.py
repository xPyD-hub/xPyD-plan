"""Benchmark Export to SQLite — export benchmark data to SQLite databases.

Convert benchmark request-level data and analysis summaries to SQLite
for ad-hoc SQL queries, dashboarding, and BI tool integration.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class SQLiteExportConfig(BaseModel):
    """Configuration for SQLite export."""

    append: bool = Field(False, description="Append to existing database instead of overwriting")
    benchmark_id: str | None = Field(
        None, description="Custom benchmark ID (auto-generated if not provided)"
    )
    include_summary: bool = Field(True, description="Include analysis summary table")
    create_indexes: bool = Field(True, description="Create indexes on key columns")


class SQLiteExportReport(BaseModel):
    """Result of a SQLite export operation."""

    output_path: str = Field(..., description="Path to the exported SQLite file")
    total_requests: int = Field(..., ge=0, description="Number of request rows exported")
    total_benchmarks: int = Field(..., ge=1, description="Number of benchmark files processed")
    benchmark_ids: list[str] = Field(..., description="Benchmark IDs written")
    tables_created: list[str] = Field(..., description="Tables in the database")
    indexes_created: int = Field(..., ge=0, description="Number of indexes created")
    file_size_bytes: int = Field(..., ge=0, description="Size of the exported file in bytes")
    appended: bool = Field(False, description="Whether data was appended to existing DB")


_REQUESTS_DDL = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    ttft_ms REAL NOT NULL,
    tpot_ms REAL NOT NULL,
    total_latency_ms REAL NOT NULL,
    timestamp REAL
)
"""

_METADATA_DDL = """
CREATE TABLE IF NOT EXISTS metadata (
    benchmark_id TEXT PRIMARY KEY,
    num_prefill_instances INTEGER NOT NULL,
    num_decode_instances INTEGER NOT NULL,
    total_instances INTEGER NOT NULL,
    measured_qps REAL NOT NULL,
    request_count INTEGER NOT NULL
)
"""

_SUMMARY_DDL = """
CREATE TABLE IF NOT EXISTS analysis_summary (
    benchmark_id TEXT PRIMARY KEY,
    ttft_p50_ms REAL,
    ttft_p95_ms REAL,
    ttft_p99_ms REAL,
    tpot_p50_ms REAL,
    tpot_p95_ms REAL,
    tpot_p99_ms REAL,
    total_latency_p50_ms REAL,
    total_latency_p95_ms REAL,
    total_latency_p99_ms REAL,
    mean_prompt_tokens REAL,
    mean_output_tokens REAL
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_requests_benchmark_id ON requests(benchmark_id)",
    "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_requests_prompt_tokens ON requests(prompt_tokens)",
    "CREATE INDEX IF NOT EXISTS idx_requests_output_tokens ON requests(output_tokens)",
    "CREATE INDEX IF NOT EXISTS idx_requests_ttft_ms ON requests(ttft_ms)",
    "CREATE INDEX IF NOT EXISTS idx_requests_tpot_ms ON requests(tpot_ms)",
]


def _percentile(vals: list[float], p: float) -> float:
    """Compute percentile using linear interpolation."""
    if not vals:
        return 0.0
    sorted_vals = sorted(vals)
    k = (len(sorted_vals) - 1) * p / 100.0
    f_idx = int(k)
    c_idx = min(f_idx + 1, len(sorted_vals) - 1)
    d = k - f_idx
    return round(sorted_vals[f_idx] + d * (sorted_vals[c_idx] - sorted_vals[f_idx]), 2)


class SQLiteExporter:
    """Export benchmark data to SQLite database."""

    def __init__(self, config: SQLiteExportConfig | None = None) -> None:
        self.config = config or SQLiteExportConfig()

    def export(
        self,
        benchmarks: list[BenchmarkData],
        output_path: str | Path,
        source_tags: list[str] | None = None,
    ) -> SQLiteExportReport:
        """Export benchmark data to a SQLite database.

        Args:
            benchmarks: One or more benchmark datasets.
            output_path: Path for the output SQLite file.
            source_tags: Optional labels for each benchmark (used as benchmark_id).

        Returns:
            SQLiteExportReport with export metadata.

        Raises:
            ValueError: If no benchmarks are provided.
        """
        if not benchmarks:
            raise ValueError("At least one benchmark dataset is required")

        output_path = Path(output_path)
        appended = False

        if output_path.exists() and not self.config.append:
            output_path.unlink()
        elif output_path.exists() and self.config.append:
            appended = True

        conn = sqlite3.connect(str(output_path))
        try:
            conn.execute(_REQUESTS_DDL)
            conn.execute(_METADATA_DDL)
            if self.config.include_summary:
                conn.execute(_SUMMARY_DDL)

            index_count = 0
            if self.config.create_indexes:
                for idx_sql in _INDEXES:
                    conn.execute(idx_sql)
                    index_count += 1

            total_requests = 0
            benchmark_ids: list[str] = []

            for i, bench in enumerate(benchmarks):
                bid = (
                    source_tags[i]
                    if source_tags and i < len(source_tags)
                    else self.config.benchmark_id or f"bench_{i}"
                )
                benchmark_ids.append(bid)

                # Insert requests
                rows = [
                    (
                        bid,
                        req.request_id,
                        req.prompt_tokens,
                        req.output_tokens,
                        req.ttft_ms,
                        req.tpot_ms,
                        req.total_latency_ms,
                        req.timestamp,
                    )
                    for req in bench.requests
                ]
                conn.executemany(
                    "INSERT INTO requests "
                    "(benchmark_id, request_id, prompt_tokens, output_tokens, "
                    "ttft_ms, tpot_ms, total_latency_ms, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                total_requests += len(rows)

                # Insert metadata
                conn.execute(
                    "INSERT OR REPLACE INTO metadata "
                    "(benchmark_id, num_prefill_instances, num_decode_instances, "
                    "total_instances, measured_qps, request_count) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        bid,
                        bench.metadata.num_prefill_instances,
                        bench.metadata.num_decode_instances,
                        bench.metadata.total_instances,
                        bench.metadata.measured_qps,
                        len(bench.requests),
                    ),
                )

                # Insert summary
                if self.config.include_summary:
                    import statistics

                    ttfts = [r.ttft_ms for r in bench.requests]
                    tpots = [r.tpot_ms for r in bench.requests]
                    totals = [r.total_latency_ms for r in bench.requests]
                    prompts = [r.prompt_tokens for r in bench.requests]
                    outputs = [r.output_tokens for r in bench.requests]

                    conn.execute(
                        "INSERT OR REPLACE INTO analysis_summary "
                        "(benchmark_id, ttft_p50_ms, ttft_p95_ms, ttft_p99_ms, "
                        "tpot_p50_ms, tpot_p95_ms, tpot_p99_ms, "
                        "total_latency_p50_ms, total_latency_p95_ms, total_latency_p99_ms, "
                        "mean_prompt_tokens, mean_output_tokens) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            bid,
                            _percentile(ttfts, 50),
                            _percentile(ttfts, 95),
                            _percentile(ttfts, 99),
                            _percentile(tpots, 50),
                            _percentile(tpots, 95),
                            _percentile(tpots, 99),
                            _percentile(totals, 50),
                            _percentile(totals, 95),
                            _percentile(totals, 99),
                            round(statistics.mean(prompts), 1) if prompts else 0.0,
                            round(statistics.mean(outputs), 1) if outputs else 0.0,
                        ),
                    )

            conn.commit()

            # Gather table list
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            ]
        finally:
            conn.close()

        file_size = output_path.stat().st_size

        return SQLiteExportReport(
            output_path=str(output_path),
            total_requests=total_requests,
            total_benchmarks=len(benchmarks),
            benchmark_ids=benchmark_ids,
            tables_created=tables,
            indexes_created=index_count,
            file_size_bytes=file_size,
            appended=appended,
        )


def export_to_sqlite(
    benchmarks: list[BenchmarkData],
    output_path: str | Path,
    *,
    append: bool = False,
    benchmark_id: str | None = None,
    include_summary: bool = True,
    create_indexes: bool = True,
    source_tags: list[str] | None = None,
) -> dict:
    """Programmatic API for SQLite export.

    Args:
        benchmarks: One or more benchmark datasets.
        output_path: Path for the output SQLite file.
        append: Append to existing database.
        benchmark_id: Custom benchmark ID.
        include_summary: Include analysis summary table.
        create_indexes: Create indexes on key columns.
        source_tags: Optional labels for each benchmark file.

    Returns:
        Dict with export result data.
    """
    config = SQLiteExportConfig(
        append=append,
        benchmark_id=benchmark_id,
        include_summary=include_summary,
        create_indexes=create_indexes,
    )
    exporter = SQLiteExporter(config)
    result = exporter.export(benchmarks, output_path, source_tags=source_tags)
    return result.model_dump()

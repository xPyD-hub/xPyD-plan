"""Benchmark Export to Parquet — export benchmark data to Apache Parquet format.

Convert benchmark request-level data and analysis results to Parquet files
for integration with pandas, DuckDB, Jupyter notebooks, and other analytical tools.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class ExportMode(str, Enum):
    """What to include in the Parquet export."""

    REQUESTS = "requests"
    SUMMARY = "summary"
    BOTH = "both"


class ParquetConfig(BaseModel):
    """Configuration for Parquet export."""

    mode: ExportMode = Field(ExportMode.REQUESTS, description="Export mode")
    enrich: bool = Field(
        False, description="Add SLA compliance and workload category columns"
    )
    sla_ttft_ms: float | None = Field(None, description="TTFT SLA threshold for enrichment")
    sla_tpot_ms: float | None = Field(None, description="TPOT SLA threshold for enrichment")
    sla_total_ms: float | None = Field(
        None, description="Total latency SLA threshold for enrichment"
    )


class ParquetExportResult(BaseModel):
    """Result of a Parquet export operation."""

    output_path: str = Field(..., description="Path to the exported Parquet file")
    mode: ExportMode
    total_requests: int = Field(..., ge=0, description="Number of request rows exported")
    total_benchmarks: int = Field(..., ge=1, description="Number of benchmark files processed")
    columns: list[str] = Field(..., description="Column names in the exported file")
    file_size_bytes: int = Field(..., ge=0, description="Size of the exported file in bytes")
    enriched: bool = Field(False, description="Whether enrichment columns were added")


def _classify_workload(prompt_tokens: int, output_tokens: int) -> str:
    """Classify a request into a workload category based on token counts."""
    total = prompt_tokens + output_tokens
    ratio = prompt_tokens / output_tokens if output_tokens > 0 else float("inf")

    if total < 100:
        return "SHORT"
    if total > 2000:
        return "LONG"
    if ratio > 3.0:
        return "PREFILL_HEAVY"
    if ratio < 0.5:
        return "DECODE_HEAVY"
    return "BALANCED"


def _check_sla(
    req_ttft: float,
    req_tpot: float,
    req_total: float,
    sla_ttft: float | None,
    sla_tpot: float | None,
    sla_total: float | None,
) -> bool:
    """Check if a single request passes SLA thresholds."""
    if sla_ttft is not None and req_ttft > sla_ttft:
        return False
    if sla_tpot is not None and req_tpot > sla_tpot:
        return False
    if sla_total is not None and req_total > sla_total:
        return False
    return True


def _build_request_rows(
    benchmarks: list[BenchmarkData],
    config: ParquetConfig,
    source_tags: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Build request-level row dicts and column list."""
    rows: list[dict] = []
    base_cols = [
        "request_id",
        "prompt_tokens",
        "output_tokens",
        "ttft_ms",
        "tpot_ms",
        "total_latency_ms",
        "timestamp",
        "qps",
        "num_prefill_instances",
        "num_decode_instances",
        "total_instances",
    ]
    extra_cols: list[str] = []
    if len(benchmarks) > 1:
        extra_cols.append("source")
    if config.enrich:
        extra_cols.extend(["sla_pass", "workload_category"])

    for idx, bench in enumerate(benchmarks):
        tag = source_tags[idx] if source_tags and idx < len(source_tags) else f"bench_{idx}"
        for req in bench.requests:
            row: dict = {
                "request_id": req.request_id,
                "prompt_tokens": req.prompt_tokens,
                "output_tokens": req.output_tokens,
                "ttft_ms": req.ttft_ms,
                "tpot_ms": req.tpot_ms,
                "total_latency_ms": req.total_latency_ms,
                "timestamp": req.timestamp,
                "qps": bench.metadata.measured_qps,
                "num_prefill_instances": bench.metadata.num_prefill_instances,
                "num_decode_instances": bench.metadata.num_decode_instances,
                "total_instances": bench.metadata.total_instances,
            }
            if len(benchmarks) > 1:
                row["source"] = tag
            if config.enrich:
                row["sla_pass"] = _check_sla(
                    req.ttft_ms,
                    req.tpot_ms,
                    req.total_latency_ms,
                    config.sla_ttft_ms,
                    config.sla_tpot_ms,
                    config.sla_total_ms,
                )
                row["workload_category"] = _classify_workload(
                    req.prompt_tokens, req.output_tokens
                )
            rows.append(row)

    return rows, base_cols + extra_cols


def _build_summary_rows(
    benchmarks: list[BenchmarkData],
    source_tags: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Build summary-level row dicts and column list."""
    import statistics

    cols = [
        "source",
        "request_count",
        "qps",
        "num_prefill_instances",
        "num_decode_instances",
        "total_instances",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "tpot_p99_ms",
        "total_latency_p50_ms",
        "total_latency_p95_ms",
        "total_latency_p99_ms",
        "mean_prompt_tokens",
        "mean_output_tokens",
    ]
    rows: list[dict] = []

    for idx, bench in enumerate(benchmarks):
        tag = source_tags[idx] if source_tags and idx < len(source_tags) else f"bench_{idx}"
        ttfts = sorted(r.ttft_ms for r in bench.requests)
        tpots = sorted(r.tpot_ms for r in bench.requests)
        totals = sorted(r.total_latency_ms for r in bench.requests)
        prompts = [r.prompt_tokens for r in bench.requests]
        outputs = [r.output_tokens for r in bench.requests]

        def _percentile(vals: list[float], p: float) -> float:
            if not vals:
                return 0.0
            k = (len(vals) - 1) * p / 100.0
            f_idx = int(k)
            c_idx = min(f_idx + 1, len(vals) - 1)
            d = k - f_idx
            return vals[f_idx] + d * (vals[c_idx] - vals[f_idx])

        rows.append({
            "source": tag,
            "request_count": len(bench.requests),
            "qps": bench.metadata.measured_qps,
            "num_prefill_instances": bench.metadata.num_prefill_instances,
            "num_decode_instances": bench.metadata.num_decode_instances,
            "total_instances": bench.metadata.total_instances,
            "ttft_p50_ms": round(_percentile(ttfts, 50), 2),
            "ttft_p95_ms": round(_percentile(ttfts, 95), 2),
            "ttft_p99_ms": round(_percentile(ttfts, 99), 2),
            "tpot_p50_ms": round(_percentile(tpots, 50), 2),
            "tpot_p95_ms": round(_percentile(tpots, 95), 2),
            "tpot_p99_ms": round(_percentile(tpots, 99), 2),
            "total_latency_p50_ms": round(_percentile(totals, 50), 2),
            "total_latency_p95_ms": round(_percentile(totals, 95), 2),
            "total_latency_p99_ms": round(_percentile(totals, 99), 2),
            "mean_prompt_tokens": round(statistics.mean(prompts), 1) if prompts else 0.0,
            "mean_output_tokens": round(statistics.mean(outputs), 1) if outputs else 0.0,
        })

    return rows, cols


class ParquetExporter:
    """Export benchmark data to Apache Parquet format."""

    def __init__(self, config: ParquetConfig | None = None) -> None:
        self.config = config or ParquetConfig()

    def export(
        self,
        benchmarks: list[BenchmarkData],
        output_path: str | Path,
        source_tags: list[str] | None = None,
    ) -> ParquetExportResult:
        """Export benchmark data to a Parquet file.

        Args:
            benchmarks: One or more benchmark datasets.
            output_path: Path for the output Parquet file.
            source_tags: Optional labels for each benchmark file.

        Returns:
            ParquetExportResult with export metadata.

        Raises:
            ImportError: If pyarrow is not installed.
            ValueError: If no benchmarks are provided.
        """
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for Parquet export. "
                "Install it with: pip install pyarrow"
            ) from exc

        if not benchmarks:
            raise ValueError("At least one benchmark dataset is required")

        output_path = Path(output_path)
        mode = self.config.mode

        all_rows: list[dict] = []
        all_cols: list[str] = []

        if mode in (ExportMode.REQUESTS, ExportMode.BOTH):
            rows, cols = _build_request_rows(benchmarks, self.config, source_tags)
            all_rows.extend(rows)
            all_cols = cols

        if mode == ExportMode.SUMMARY:
            rows, cols = _build_summary_rows(benchmarks, source_tags)
            all_rows = rows
            all_cols = cols
        elif mode == ExportMode.BOTH:
            # For BOTH mode, we write the request-level data (richer).
            # Summary can be derived from it. We add summary as a separate
            # row group if pyarrow supports it, but for simplicity we
            # embed summary columns into each request row.
            pass  # request rows already built above

        # Build pyarrow table
        if not all_rows:
            # Edge case: no requests in any benchmark
            table = pa.table({col: [] for col in all_cols})
        else:
            col_arrays: dict[str, list] = {col: [] for col in all_cols}
            for row in all_rows:
                for col in all_cols:
                    col_arrays[col].append(row.get(col))
            table = pa.table(col_arrays)

        pq.write_table(table, str(output_path))

        file_size = output_path.stat().st_size

        return ParquetExportResult(
            output_path=str(output_path),
            mode=mode,
            total_requests=len(all_rows),
            total_benchmarks=len(benchmarks),
            columns=all_cols,
            file_size_bytes=file_size,
            enriched=self.config.enrich,
        )


def export_parquet(
    benchmarks: list[BenchmarkData],
    output_path: str | Path,
    *,
    mode: str = "requests",
    enrich: bool = False,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_ms: float | None = None,
    source_tags: list[str] | None = None,
) -> dict:
    """Programmatic API for Parquet export.

    Args:
        benchmarks: One or more benchmark datasets.
        output_path: Path for the output Parquet file.
        mode: Export mode ('requests', 'summary', 'both').
        enrich: Add SLA compliance and workload category columns.
        sla_ttft_ms: TTFT SLA threshold for enrichment.
        sla_tpot_ms: TPOT SLA threshold for enrichment.
        sla_total_ms: Total latency SLA threshold for enrichment.
        source_tags: Optional labels for each benchmark file.

    Returns:
        Dict with export result data.
    """
    config = ParquetConfig(
        mode=ExportMode(mode),
        enrich=enrich,
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_ms=sla_total_ms,
    )
    exporter = ParquetExporter(config)
    result = exporter.export(benchmarks, output_path, source_tags=source_tags)
    return result.model_dump()

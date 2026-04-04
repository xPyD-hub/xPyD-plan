"""Latency heatmap data generation from benchmark results."""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class AggregationMetric(str, Enum):
    """Aggregation method for latency values within a cell."""

    MEAN = "mean"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"


class LatencyField(str, Enum):
    """Which latency field to analyze."""

    TTFT = "ttft_ms"
    TPOT = "tpot_ms"
    TOTAL = "total_latency_ms"


class HeatmapCell(BaseModel):
    """Single cell in the heatmap grid."""

    prompt_bin_start: int = Field(..., description="Start of prompt token bin (inclusive)")
    prompt_bin_end: int = Field(..., description="End of prompt token bin (exclusive)")
    output_bin_start: int = Field(..., description="Start of output token bin (inclusive)")
    output_bin_end: int = Field(..., description="End of output token bin (exclusive)")
    value: float = Field(..., description="Aggregated latency value")
    count: int = Field(..., ge=0, description="Number of requests in this cell")
    is_hotspot: bool = Field(False, description="Whether this cell exceeds the SLA threshold")


class HeatmapConfig(BaseModel):
    """Configuration for heatmap generation."""

    prompt_bins: int = Field(10, ge=1, description="Number of bins for prompt tokens")
    output_bins: int = Field(10, ge=1, description="Number of bins for output tokens")
    metric: AggregationMetric = Field(AggregationMetric.P95, description="Aggregation metric")
    field: LatencyField = Field(LatencyField.TOTAL, description="Latency field to analyze")
    sla_threshold_ms: float | None = Field(
        None, ge=0, description="SLA threshold for hotspot detection"
    )


class HeatmapGrid(BaseModel):
    """2D grid of heatmap cells."""

    cells: list[HeatmapCell] = Field(..., description="All cells in row-major order")
    rows: int = Field(..., ge=1, description="Number of rows (prompt bins)")
    cols: int = Field(..., ge=1, description="Number of columns (output bins)")


class HeatmapReport(BaseModel):
    """Complete heatmap analysis report."""

    config: HeatmapConfig = Field(..., description="Configuration used")
    grid: HeatmapGrid = Field(..., description="The heatmap grid")
    total_requests: int = Field(..., ge=0, description="Total requests analyzed")
    hotspot_count: int = Field(0, ge=0, description="Number of hotspot cells")
    min_value: float | None = Field(None, description="Minimum cell value")
    max_value: float | None = Field(None, description="Maximum cell value")
    recommendation: str = Field(..., description="Human-readable summary")


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (p / 100.0) * (len(sorted_vals) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _aggregate(values: list[float], metric: AggregationMetric) -> float:
    """Aggregate a list of values using the specified metric."""
    if not values:
        return 0.0
    if metric == AggregationMetric.MEAN:
        return sum(values) / len(values)
    if metric == AggregationMetric.P50:
        return _percentile(values, 50)
    if metric == AggregationMetric.P95:
        return _percentile(values, 95)
    # P99
    return _percentile(values, 99)


class HeatmapGenerator:
    """Generate 2D latency heatmap data from benchmark results."""

    def generate(
        self,
        data: BenchmarkData,
        config: HeatmapConfig | None = None,
    ) -> HeatmapReport:
        """Generate a heatmap report from benchmark data.

        Args:
            data: Loaded benchmark data.
            config: Optional configuration; uses defaults if not provided.

        Returns:
            HeatmapReport with the 2D grid and hotspot analysis.
        """
        if config is None:
            config = HeatmapConfig()

        requests = data.requests
        if not requests:
            return HeatmapReport(
                config=config,
                grid=HeatmapGrid(cells=[], rows=config.prompt_bins, cols=config.output_bins),
                total_requests=0,
                hotspot_count=0,
                min_value=None,
                max_value=None,
                recommendation="No requests in benchmark data.",
            )

        # Determine ranges
        prompt_tokens = [r.prompt_tokens for r in requests]
        output_tokens = [r.output_tokens for r in requests]

        prompt_min, prompt_max = min(prompt_tokens), max(prompt_tokens)
        output_min, output_max = min(output_tokens), max(output_tokens)

        # Build bin edges
        prompt_edges = self._build_edges(prompt_min, prompt_max, config.prompt_bins)
        output_edges = self._build_edges(output_min, output_max, config.output_bins)

        # Bucket requests
        field_name = config.field.value
        buckets: dict[tuple[int, int], list[float]] = {}
        for i in range(len(prompt_edges) - 1):
            for j in range(len(output_edges) - 1):
                buckets[(i, j)] = []

        for req in requests:
            pi = self._find_bin(req.prompt_tokens, prompt_edges)
            oi = self._find_bin(req.output_tokens, output_edges)
            val = getattr(req, field_name)
            buckets[(pi, oi)].append(val)

        # Build cells
        cells: list[HeatmapCell] = []
        hotspot_count = 0
        all_values: list[float] = []

        for i in range(len(prompt_edges) - 1):
            for j in range(len(output_edges) - 1):
                vals = buckets[(i, j)]
                if vals:
                    agg_val = _aggregate(vals, config.metric)
                    is_hotspot = (
                        config.sla_threshold_ms is not None
                        and agg_val > config.sla_threshold_ms
                    )
                    if is_hotspot:
                        hotspot_count += 1
                    all_values.append(agg_val)
                else:
                    agg_val = 0.0
                    is_hotspot = False

                cells.append(HeatmapCell(
                    prompt_bin_start=prompt_edges[i],
                    prompt_bin_end=prompt_edges[i + 1],
                    output_bin_start=output_edges[j],
                    output_bin_end=output_edges[j + 1],
                    value=agg_val,
                    count=len(vals),
                    is_hotspot=is_hotspot,
                ))

        min_val = min(all_values) if all_values else None
        max_val = max(all_values) if all_values else None

        # Recommendation
        if hotspot_count > 0:
            recommendation = (
                f"{hotspot_count} hotspot(s) detected exceeding "
                f"{config.sla_threshold_ms}ms SLA threshold. "
                f"Investigate high-latency token ranges."
            )
        elif all_values:
            recommendation = (
                f"No hotspots detected. Latency range: "
                f"{min_val:.1f}ms – {max_val:.1f}ms."
            )
        else:
            recommendation = "No requests in benchmark data."

        return HeatmapReport(
            config=config,
            grid=HeatmapGrid(
                cells=cells,
                rows=len(prompt_edges) - 1,
                cols=len(output_edges) - 1,
            ),
            total_requests=len(requests),
            hotspot_count=hotspot_count,
            min_value=min_val,
            max_value=max_val,
            recommendation=recommendation,
        )

    @staticmethod
    def _build_edges(lo: int, hi: int, num_bins: int) -> list[int]:
        """Build bin edges. Returns num_bins+1 edge values."""
        if lo == hi:
            return [lo, lo + 1]
        step = (hi - lo) / num_bins
        edges = [int(lo + step * i) for i in range(num_bins)]
        edges.append(hi + 1)  # exclusive upper bound
        return edges

    @staticmethod
    def _find_bin(value: int, edges: list[int]) -> int:
        """Find bin index for a value given sorted edges."""
        for i in range(len(edges) - 1):
            if value < edges[i + 1]:
                return i
        return len(edges) - 2  # last bin


def generate_heatmap(
    benchmark_path: str | Path,
    prompt_bins: int = 10,
    output_bins: int = 10,
    metric: str = "p95",
    field: str = "total_latency_ms",
    sla_threshold_ms: float | None = None,
) -> dict:
    """Programmatic API for heatmap generation.

    Args:
        benchmark_path: Path to benchmark JSON file.
        prompt_bins: Number of prompt token bins.
        output_bins: Number of output token bins.
        metric: Aggregation metric (mean, p50, p95, p99).
        field: Latency field (ttft_ms, tpot_ms, total_latency_ms).
        sla_threshold_ms: Optional SLA threshold for hotspot detection.

    Returns:
        Dictionary representation of the HeatmapReport.
    """
    import json as json_mod

    path = Path(benchmark_path)
    raw = json_mod.loads(path.read_text())
    data = BenchmarkData.model_validate(raw)

    config = HeatmapConfig(
        prompt_bins=prompt_bins,
        output_bins=output_bins,
        metric=AggregationMetric(metric),
        field=LatencyField(field),
        sla_threshold_ms=sla_threshold_ms,
    )

    generator = HeatmapGenerator()
    report = generator.generate(data, config)
    return report.model_dump()

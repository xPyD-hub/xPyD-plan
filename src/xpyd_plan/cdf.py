"""Latency CDF (Cumulative Distribution Function) export.

Generate CDF data points for latency metrics, enabling distribution
visualization and multi-benchmark CDF overlay comparisons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.benchmark_models import BenchmarkData


class CDFPoint(BaseModel):
    """Single CDF data point."""

    latency_ms: float = Field(..., description="Latency value (ms)")
    cumulative_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Cumulative probability [0, 1]"
    )


class SLAMarker(BaseModel):
    """SLA threshold marker on a CDF curve."""

    threshold_ms: float = Field(..., description="SLA threshold (ms)")
    percentile_at_threshold: float = Field(
        ..., ge=0.0, le=100.0, description="Percentile of requests meeting threshold"
    )


class CDFCurve(BaseModel):
    """CDF curve for a single metric from a single source."""

    source: str = Field(..., description="Source benchmark file or label")
    metric: str = Field(..., description="Latency metric name")
    points: list[CDFPoint] = Field(..., description="CDF data points")
    sla_marker: SLAMarker | None = Field(
        None, description="SLA threshold marker (if configured)"
    )


class CDFReport(BaseModel):
    """Complete CDF export report."""

    curves: list[CDFCurve] = Field(..., description="CDF curves")


_METRIC_FIELDS = {
    "ttft": "ttft_ms",
    "tpot": "tpot_ms",
    "total_latency": "total_latency_ms",
}


def _extract_values(data: BenchmarkData, metric: str) -> np.ndarray:
    """Extract latency values for the given metric."""
    field = _METRIC_FIELDS.get(metric, metric)
    return np.array([getattr(r, field) for r in data.requests], dtype=float)


def _build_cdf_points(values: np.ndarray, n_points: int) -> list[CDFPoint]:
    """Build CDF data points via uniform quantile sampling."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n == 0:
        return []

    # Generate n_points evenly spaced quantiles from 0 to 1
    quantiles = np.linspace(0.0, 1.0, n_points)
    points: list[CDFPoint] = []
    for q in quantiles:
        idx = min(int(q * n), n - 1)
        points.append(
            CDFPoint(
                latency_ms=float(sorted_vals[idx]),
                cumulative_probability=float(q),
            )
        )
    return points


def _build_sla_marker(
    sorted_vals: np.ndarray, threshold_ms: float
) -> SLAMarker:
    """Compute the percentile at which the SLA threshold is crossed."""
    n = len(sorted_vals)
    if n == 0:
        return SLAMarker(threshold_ms=threshold_ms, percentile_at_threshold=0.0)
    count_below = int(np.searchsorted(sorted_vals, threshold_ms, side="right"))
    pct = (count_below / n) * 100.0
    return SLAMarker(threshold_ms=threshold_ms, percentile_at_threshold=pct)


class CDFGenerator:
    """Generate CDF data from benchmark data."""

    def __init__(self, data: BenchmarkData, source: str = "benchmark") -> None:
        self._data = data
        self._source = source

    def generate(
        self,
        metric: str = "total_latency",
        n_points: int = 100,
        sla_threshold_ms: float | None = None,
    ) -> CDFCurve:
        """Generate a CDF curve for a single metric."""
        values = _extract_values(self._data, metric)
        points = _build_cdf_points(values, n_points)

        sla_marker = None
        if sla_threshold_ms is not None:
            sorted_vals = np.sort(values)
            sla_marker = _build_sla_marker(sorted_vals, sla_threshold_ms)

        return CDFCurve(
            source=self._source,
            metric=metric,
            points=points,
            sla_marker=sla_marker,
        )


def generate_cdf(
    benchmarks: list[str | Path],
    metric: str = "total_latency",
    n_points: int = 100,
    sla_threshold_ms: float | None = None,
    labels: list[str] | None = None,
) -> CDFReport:
    """Programmatic API: generate CDF curves from one or more benchmark files.

    Args:
        benchmarks: Paths to benchmark JSON files.
        metric: Latency metric (ttft, tpot, total_latency).
        n_points: Number of CDF data points per curve.
        sla_threshold_ms: Optional SLA threshold for marker.
        labels: Optional per-file labels (defaults to filenames).

    Returns:
        CDFReport with one curve per benchmark file.
    """
    curves: list[CDFCurve] = []
    for i, path in enumerate(benchmarks):
        path = Path(path)
        label = labels[i] if labels and i < len(labels) else path.name
        data = load_benchmark_auto(str(path))
        gen = CDFGenerator(data, source=label)
        curve = gen.generate(
            metric=metric,
            n_points=n_points,
            sla_threshold_ms=sla_threshold_ms,
        )
        curves.append(curve)
    return CDFReport(curves=curves)

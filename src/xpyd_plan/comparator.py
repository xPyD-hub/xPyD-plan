"""Benchmark comparison and regression detection.

Compare two benchmark datasets (baseline vs current) to detect performance
regressions or improvements across latency and throughput metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class MetricDelta(BaseModel):
    """Delta between baseline and current for a single metric."""

    metric: str = Field(..., description="Metric name (e.g., ttft_p95_ms)")
    baseline: float = Field(..., description="Baseline value")
    current: float = Field(..., description="Current value")
    absolute_delta: float = Field(..., description="current - baseline")
    relative_delta: float = Field(
        ..., description="Relative change: (current - baseline) / baseline, or 0.0 if baseline is 0"
    )
    is_regression: bool = Field(
        False, description="True if metric degraded beyond threshold"
    )


class ComparisonResult(BaseModel):
    """Result of comparing two benchmark datasets."""

    baseline_qps: float
    current_qps: float
    qps_delta: MetricDelta
    latency_deltas: list[MetricDelta] = Field(default_factory=list)
    has_regression: bool = Field(False, description="True if any metric regressed")
    regression_count: int = Field(0, description="Number of regressed metrics")
    threshold: float = Field(0.1, description="Regression threshold used")


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile."""
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _compute_delta(
    metric: str,
    baseline: float,
    current: float,
    threshold: float,
    higher_is_worse: bool = True,
) -> MetricDelta:
    """Compute a metric delta and flag regression."""
    absolute = current - baseline
    if baseline > 0:
        relative = absolute / baseline
    else:
        relative = 0.0

    if higher_is_worse:
        is_regression = relative > threshold
    else:
        # For QPS: lower is worse → regression if it dropped beyond threshold
        is_regression = -relative > threshold

    return MetricDelta(
        metric=metric,
        baseline=baseline,
        current=current,
        absolute_delta=absolute,
        relative_delta=relative,
        is_regression=is_regression,
    )


class BenchmarkComparator:
    """Compare two benchmark datasets for regression detection.

    Args:
        threshold: Relative change threshold to flag as regression (default 0.1 = 10%).
    """

    def __init__(self, threshold: float = 0.1) -> None:
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        self.threshold = threshold

    def compare(self, baseline: BenchmarkData, current: BenchmarkData) -> ComparisonResult:
        """Compare baseline and current benchmark data.

        Args:
            baseline: The reference benchmark dataset.
            current: The new benchmark dataset to compare against baseline.

        Returns:
            ComparisonResult with per-metric deltas and regression flags.
        """
        # Extract latency lists
        b_ttft = [r.ttft_ms for r in baseline.requests]
        b_tpot = [r.tpot_ms for r in baseline.requests]
        b_total = [r.total_latency_ms for r in baseline.requests]
        c_ttft = [r.ttft_ms for r in current.requests]
        c_tpot = [r.tpot_ms for r in current.requests]
        c_total = [r.total_latency_ms for r in current.requests]

        # Latency metrics at P50, P95, P99
        latency_deltas: list[MetricDelta] = []
        for name, b_vals, c_vals in [
            ("ttft", b_ttft, c_ttft),
            ("tpot", b_tpot, c_tpot),
            ("total_latency", b_total, c_total),
        ]:
            for p in [50, 95, 99]:
                metric_name = f"{name}_p{p}_ms"
                b_val = _percentile(b_vals, p)
                c_val = _percentile(c_vals, p)
                delta = _compute_delta(metric_name, b_val, c_val, self.threshold)
                latency_deltas.append(delta)

        # QPS delta (higher is better)
        qps_delta = _compute_delta(
            "measured_qps",
            baseline.metadata.measured_qps,
            current.metadata.measured_qps,
            self.threshold,
            higher_is_worse=False,
        )

        all_deltas = latency_deltas + [qps_delta]
        regression_count = sum(1 for d in all_deltas if d.is_regression)

        return ComparisonResult(
            baseline_qps=baseline.metadata.measured_qps,
            current_qps=current.metadata.measured_qps,
            qps_delta=qps_delta,
            latency_deltas=latency_deltas,
            has_regression=regression_count > 0,
            regression_count=regression_count,
            threshold=self.threshold,
        )


def compare_benchmarks(
    baseline_path: str | Path,
    current_path: str | Path,
    threshold: float = 0.1,
) -> ComparisonResult:
    """Programmatic API: compare two benchmark files.

    Args:
        baseline_path: Path to baseline benchmark JSON file.
        current_path: Path to current benchmark JSON file.
        threshold: Regression threshold (default 0.1 = 10%).

    Returns:
        ComparisonResult with all metric deltas and regression flags.
    """
    baseline_data = json.loads(Path(baseline_path).read_text())
    current_data = json.loads(Path(current_path).read_text())
    baseline = BenchmarkData(**baseline_data)
    current = BenchmarkData(**current_data)
    comparator = BenchmarkComparator(threshold=threshold)
    return comparator.compare(baseline, current)

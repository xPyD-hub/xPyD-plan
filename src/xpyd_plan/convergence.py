"""Percentile convergence analysis for benchmark data."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class StabilityStatus(str, Enum):
    """Stability classification based on coefficient of variation."""

    STABLE = "stable"
    MARGINAL = "marginal"
    UNSTABLE = "unstable"


class ConvergencePoint(BaseModel):
    """Running percentile value at a specific sample size."""

    sample_size: int = Field(description="Number of requests in this window")
    sample_fraction: float = Field(description="Fraction of total requests (0.0–1.0)")
    p50: float = Field(description="P50 at this sample size")
    p95: float = Field(description="P95 at this sample size")
    p99: float = Field(description="P99 at this sample size")


class MetricConvergence(BaseModel):
    """Convergence analysis for a single latency metric."""

    field: str = Field(description="Latency field name")
    points: list[ConvergencePoint] = Field(description="Running percentile values")
    cv_p95: float = Field(description="Coefficient of variation of P95 across windows")
    cv_p99: float = Field(description="Coefficient of variation of P99 across windows")
    status: StabilityStatus = Field(description="Overall stability classification")
    min_stable_sample_size: Optional[int] = Field(
        default=None,
        description="Minimum sample size at which metric stabilizes (None if unstable)",
    )


class ConvergenceReport(BaseModel):
    """Complete convergence analysis report."""

    metrics: list[MetricConvergence] = Field(description="Per-metric convergence analysis")
    overall_status: StabilityStatus = Field(description="Worst status across all metrics")
    total_requests: int = Field(description="Total requests in benchmark")
    recommended_min_requests: Optional[int] = Field(
        default=None,
        description="Recommended minimum requests for stable percentiles",
    )


def _classify_stability(cv: float, threshold: float) -> StabilityStatus:
    """Classify stability based on coefficient of variation."""
    if cv <= threshold:
        return StabilityStatus.STABLE
    elif cv <= threshold * 2:
        return StabilityStatus.MARGINAL
    else:
        return StabilityStatus.UNSTABLE


def _worst_status(a: StabilityStatus, b: StabilityStatus) -> StabilityStatus:
    order = [StabilityStatus.STABLE, StabilityStatus.MARGINAL, StabilityStatus.UNSTABLE]
    return max(a, b, key=lambda s: order.index(s))


class ConvergenceAnalyzer:
    """Analyze whether benchmark percentile metrics have converged."""

    FIELDS = ["ttft_ms", "tpot_ms", "total_latency_ms"]

    def __init__(self, data: BenchmarkData) -> None:
        self._data = data

    def analyze(
        self,
        steps: int = 10,
        threshold: float = 0.05,
    ) -> ConvergenceReport:
        """Run convergence analysis.

        Args:
            steps: Number of cumulative windows (e.g. 10 → 10%, 20%, ..., 100%).
            threshold: CV threshold for STABLE classification (default 5%).
        """
        requests = self._data.requests
        n = len(requests)
        if n == 0:
            return ConvergenceReport(
                metrics=[],
                overall_status=StabilityStatus.UNSTABLE,
                total_requests=0,
                recommended_min_requests=None,
            )

        # Build arrays
        arrays: dict[str, np.ndarray] = {}
        for field in self.FIELDS:
            arrays[field] = np.array([getattr(r, field) for r in requests])

        # Window sizes
        window_sizes = [max(1, int(n * (i + 1) / steps)) for i in range(steps)]
        # Deduplicate while preserving order
        seen: set[int] = set()
        unique_sizes: list[int] = []
        for s in window_sizes:
            if s not in seen:
                seen.add(s)
                unique_sizes.append(s)
        window_sizes = unique_sizes

        metrics: list[MetricConvergence] = []
        overall = StabilityStatus.STABLE

        for field in self.FIELDS:
            arr = arrays[field]
            points: list[ConvergencePoint] = []

            for ws in window_sizes:
                subset = arr[:ws]
                points.append(
                    ConvergencePoint(
                        sample_size=ws,
                        sample_fraction=ws / n,
                        p50=float(np.percentile(subset, 50)),
                        p95=float(np.percentile(subset, 95)),
                        p99=float(np.percentile(subset, 99)),
                    )
                )

            # Compute CV over the last half of windows for stability
            tail_points = points[len(points) // 2 :]
            p95_values = [p.p95 for p in tail_points]
            p99_values = [p.p99 for p in tail_points]

            mean_p95 = np.mean(p95_values)
            mean_p99 = np.mean(p99_values)
            cv_p95 = float(np.std(p95_values) / mean_p95) if mean_p95 > 0 else 0.0
            cv_p99 = float(np.std(p99_values) / mean_p99) if mean_p99 > 0 else 0.0

            worst_cv = max(cv_p95, cv_p99)
            status = _classify_stability(worst_cv, threshold)

            # Find minimum stable sample size
            min_stable: Optional[int] = None
            if status != StabilityStatus.UNSTABLE and len(points) >= 3:
                # Walk forward: find the first point from which all subsequent CVs are stable
                for start_idx in range(len(points) - 2):
                    remaining = points[start_idx:]
                    r_p95 = [p.p95 for p in remaining]
                    r_p99 = [p.p99 for p in remaining]
                    m95 = np.mean(r_p95)
                    m99 = np.mean(r_p99)
                    c95 = float(np.std(r_p95) / m95) if m95 > 0 else 0.0
                    c99 = float(np.std(r_p99) / m99) if m99 > 0 else 0.0
                    if max(c95, c99) <= threshold:
                        min_stable = points[start_idx].sample_size
                        break

            mc = MetricConvergence(
                field=field,
                points=points,
                cv_p95=cv_p95,
                cv_p99=cv_p99,
                status=status,
                min_stable_sample_size=min_stable,
            )
            metrics.append(mc)
            overall = _worst_status(overall, status)

        # Recommended min requests: max of all min_stable values, or None
        min_stables = [
            m.min_stable_sample_size
            for m in metrics
            if m.min_stable_sample_size is not None
        ]
        recommended = max(min_stables) if min_stables else None

        return ConvergenceReport(
            metrics=metrics,
            overall_status=overall,
            total_requests=n,
            recommended_min_requests=recommended,
        )


def analyze_convergence(
    data: BenchmarkData,
    steps: int = 10,
    threshold: float = 0.05,
) -> dict:
    """Programmatic API for convergence analysis.

    Returns:
        dict representation of ConvergenceReport.
    """
    analyzer = ConvergenceAnalyzer(data)
    report = analyzer.analyze(steps=steps, threshold=threshold)
    return report.model_dump()

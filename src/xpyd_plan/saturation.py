"""Saturation point detection — find the QPS level where the system saturates.

Given multiple benchmark files at increasing QPS levels (same P:D configuration),
identify the QPS threshold where latency degrades beyond acceptable limits.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class SaturationPoint(BaseModel):
    """A single QPS data point with latency metrics."""

    measured_qps: float = Field(..., gt=0, description="Measured QPS for this benchmark")
    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    ttft_p95_ms: float = Field(..., ge=0)
    tpot_p95_ms: float = Field(..., ge=0)
    total_latency_p95_ms: float = Field(..., ge=0)
    ttft_p99_ms: float = Field(..., ge=0)
    tpot_p99_ms: float = Field(..., ge=0)
    total_latency_p99_ms: float = Field(..., ge=0)
    request_count: int = Field(..., ge=1)


class SaturationThreshold(BaseModel):
    """Detected saturation threshold for a specific metric."""

    metric: str = Field(..., description="Metric name (e.g., ttft_p95_ms)")
    saturated_qps: float = Field(
        ..., gt=0, description="QPS at which saturation was detected"
    )
    safe_qps: float = Field(
        ..., gt=0, description="Highest QPS before saturation"
    )
    latency_at_safe: float = Field(..., ge=0, description="Latency at safe QPS")
    latency_at_saturated: float = Field(..., ge=0, description="Latency at saturated QPS")
    increase_pct: float = Field(
        ..., description="Percentage increase from safe to saturated"
    )


class SaturationReport(BaseModel):
    """Complete saturation analysis report."""

    points: list[SaturationPoint] = Field(..., min_length=2)
    thresholds: list[SaturationThreshold] = Field(
        default_factory=list,
        description="Detected saturation thresholds per metric",
    )
    overall_safe_qps: float | None = Field(
        None,
        description="Conservative safe QPS (minimum across all metrics, None if no saturation)",
    )
    recommendation: str = Field(..., description="Human-readable recommendation")


class SaturationDetector:
    """Detect QPS saturation point from multiple benchmark runs.

    Saturation is detected when the latency increase between consecutive QPS
    levels exceeds a configurable threshold (default: 50% relative increase).
    """

    def __init__(self, increase_threshold: float = 0.5) -> None:
        """Initialize with saturation detection threshold.

        Args:
            increase_threshold: Relative latency increase between consecutive
                QPS points that triggers saturation detection. Default 0.5 (50%).
                Must be > 0.
        """
        if increase_threshold <= 0:
            msg = f"increase_threshold must be > 0, got {increase_threshold}"
            raise ValueError(msg)
        self.increase_threshold = increase_threshold

    def analyze(self, benchmarks: list[BenchmarkData]) -> SaturationReport:
        """Analyze saturation across multiple benchmark runs at different QPS levels.

        Args:
            benchmarks: List of benchmark datasets at different QPS levels.
                Must contain at least 2 benchmarks with distinct measured_qps.

        Returns:
            SaturationReport with detected saturation thresholds.

        Raises:
            ValueError: If fewer than 2 benchmarks or all have same QPS.
        """
        if len(benchmarks) < 2:
            msg = "Need at least 2 benchmark datasets for saturation analysis"
            raise ValueError(msg)

        # Sort by QPS
        sorted_benchmarks = sorted(benchmarks, key=lambda b: b.metadata.measured_qps)

        # Check distinct QPS values
        qps_values = {b.metadata.measured_qps for b in sorted_benchmarks}
        if len(qps_values) < 2:
            msg = "Need benchmarks with at least 2 distinct measured_qps values"
            raise ValueError(msg)

        # Build saturation points
        points: list[SaturationPoint] = []
        for bench in sorted_benchmarks:
            meta = bench.metadata
            ttfts = [r.ttft_ms for r in bench.requests]
            tpots = [r.tpot_ms for r in bench.requests]
            totals = [r.total_latency_ms for r in bench.requests]

            points.append(
                SaturationPoint(
                    measured_qps=meta.measured_qps,
                    num_prefill=meta.num_prefill_instances,
                    num_decode=meta.num_decode_instances,
                    ttft_p95_ms=round(float(np.percentile(ttfts, 95)), 2),
                    tpot_p95_ms=round(float(np.percentile(tpots, 95)), 2),
                    total_latency_p95_ms=round(float(np.percentile(totals, 95)), 2),
                    ttft_p99_ms=round(float(np.percentile(ttfts, 99)), 2),
                    tpot_p99_ms=round(float(np.percentile(tpots, 99)), 2),
                    total_latency_p99_ms=round(float(np.percentile(totals, 99)), 2),
                    request_count=len(bench.requests),
                )
            )

        # Detect saturation per metric
        metrics = [
            "ttft_p95_ms",
            "tpot_p95_ms",
            "total_latency_p95_ms",
            "ttft_p99_ms",
            "tpot_p99_ms",
            "total_latency_p99_ms",
        ]

        thresholds: list[SaturationThreshold] = []
        for metric in metrics:
            threshold = self._detect_metric_saturation(points, metric)
            if threshold is not None:
                thresholds.append(threshold)

        # Overall safe QPS
        overall_safe_qps: float | None = None
        if thresholds:
            overall_safe_qps = min(t.safe_qps for t in thresholds)

        recommendation = self._build_recommendation(points, thresholds, overall_safe_qps)

        return SaturationReport(
            points=points,
            thresholds=thresholds,
            overall_safe_qps=overall_safe_qps,
            recommendation=recommendation,
        )

    def _detect_metric_saturation(
        self, points: list[SaturationPoint], metric: str
    ) -> SaturationThreshold | None:
        """Detect saturation for a single metric.

        Returns the first QPS transition where latency increase exceeds threshold.
        """
        for i in range(1, len(points)):
            prev_val = getattr(points[i - 1], metric)
            curr_val = getattr(points[i], metric)

            if prev_val <= 0:
                continue

            increase_pct = (curr_val - prev_val) / prev_val

            if increase_pct >= self.increase_threshold:
                return SaturationThreshold(
                    metric=metric,
                    saturated_qps=points[i].measured_qps,
                    safe_qps=points[i - 1].measured_qps,
                    latency_at_safe=round(prev_val, 2),
                    latency_at_saturated=round(curr_val, 2),
                    increase_pct=round(increase_pct * 100, 1),
                )

        return None

    def _build_recommendation(
        self,
        points: list[SaturationPoint],
        thresholds: list[SaturationThreshold],
        overall_safe_qps: float | None,
    ) -> str:
        """Generate a human-readable recommendation."""
        if not thresholds:
            max_qps = points[-1].measured_qps
            return (
                f"No saturation detected up to {max_qps:.1f} QPS. "
                f"The system handles all tested load levels without significant "
                f"latency degradation."
            )

        saturated_metrics = [t.metric for t in thresholds]
        return (
            f"Saturation detected. Safe operating QPS: {overall_safe_qps:.1f}. "
            f"Metrics degrading: {', '.join(saturated_metrics)}. "
            f"Do not exceed {overall_safe_qps:.1f} QPS to maintain latency stability."
        )


def detect_saturation(
    benchmarks: list[BenchmarkData],
    increase_threshold: float = 0.5,
) -> dict:
    """Programmatic API for saturation detection.

    Args:
        benchmarks: List of benchmark datasets at different QPS levels.
        increase_threshold: Relative latency increase threshold for detection.

    Returns:
        Dictionary representation of SaturationReport.
    """
    detector = SaturationDetector(increase_threshold=increase_threshold)
    report = detector.analyze(benchmarks)
    return report.model_dump()

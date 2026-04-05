"""Latency jitter analysis — measures request-to-request latency variability."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class JitterClassification(str, Enum):
    """Classification of latency jitter severity."""

    STABLE = "stable"       # CV <= 0.3
    MODERATE = "moderate"   # 0.3 < CV <= 0.7
    HIGH = "high"           # CV > 0.7


class JitterStats(BaseModel):
    """Jitter statistics for a single latency metric."""

    consecutive_mean: float = Field(
        ..., description="Mean absolute difference between consecutive requests"
    )
    consecutive_p95: float = Field(
        ..., description="P95 of absolute differences between consecutive requests"
    )
    std: float = Field(..., description="Standard deviation of latencies")
    iqr: float = Field(..., description="Interquartile range (P75 - P25)")
    cv: float = Field(
        ..., description="Coefficient of variation (std / mean)"
    )
    classification: JitterClassification = Field(
        ..., description="Jitter severity classification"
    )


class MetricJitter(BaseModel):
    """Jitter analysis for a named metric."""

    metric: str = Field(..., description="Metric name (ttft_ms, tpot_ms, total_latency_ms)")
    sample_size: int = Field(..., ge=1, description="Number of data points")
    stats: JitterStats = Field(..., description="Jitter statistics")


class JitterReport(BaseModel):
    """Complete jitter analysis report."""

    metrics: list[MetricJitter] = Field(..., description="Per-metric jitter results")
    worst_metric: str = Field(..., description="Metric with highest CV")
    recommendation: str = Field(..., description="Human-readable summary")


def _classify_jitter(cv: float) -> JitterClassification:
    """Classify jitter based on coefficient of variation."""
    if cv <= 0.3:
        return JitterClassification.STABLE
    if cv <= 0.7:
        return JitterClassification.MODERATE
    return JitterClassification.HIGH


def _percentile(values: list[float], pct: float) -> float:
    """Compute percentile using linear interpolation."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    k = (pct / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)


def _compute_jitter(values: list[float]) -> JitterStats:
    """Compute jitter statistics for a list of latency values."""
    n = len(values)
    mean = sum(values) / n

    # Consecutive jitter
    diffs = [abs(values[i + 1] - values[i]) for i in range(n - 1)]
    consecutive_mean = sum(diffs) / len(diffs) if diffs else 0.0
    consecutive_p95 = _percentile(diffs, 95.0) if diffs else 0.0

    # Overall jitter
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    iqr = _percentile(values, 75.0) - _percentile(values, 25.0)
    cv = std / mean if mean > 0 else 0.0

    return JitterStats(
        consecutive_mean=round(consecutive_mean, 4),
        consecutive_p95=round(consecutive_p95, 4),
        std=round(std, 4),
        iqr=round(iqr, 4),
        cv=round(cv, 4),
        classification=_classify_jitter(cv),
    )


_METRICS = ["ttft_ms", "tpot_ms", "total_latency_ms"]


class JitterAnalyzer:
    """Analyze latency jitter across request streams."""

    def analyze(self, data: BenchmarkData) -> JitterReport:
        """Run jitter analysis on benchmark data.

        Args:
            data: Loaded benchmark data.

        Returns:
            JitterReport with per-metric jitter statistics.

        Raises:
            ValueError: If fewer than 2 requests in data.
        """
        requests = data.requests
        if len(requests) < 2:
            raise ValueError("Need at least 2 requests for jitter analysis")

        metric_results: list[MetricJitter] = []
        for metric in _METRICS:
            values = [getattr(r, metric) for r in requests]
            stats = _compute_jitter(values)
            metric_results.append(
                MetricJitter(metric=metric, sample_size=len(values), stats=stats)
            )

        worst = max(metric_results, key=lambda m: m.stats.cv)

        if worst.stats.classification == JitterClassification.HIGH:
            recommendation = (
                f"High jitter detected on {worst.metric} (CV={worst.stats.cv:.2f}). "
                f"Latency is highly unpredictable. Investigate system-level causes "
                f"(GC pauses, contention, cold starts)."
            )
        elif worst.stats.classification == JitterClassification.MODERATE:
            recommendation = (
                f"Moderate jitter on {worst.metric} (CV={worst.stats.cv:.2f}). "
                f"Latency variability is noticeable but may be acceptable for some SLAs."
            )
        else:
            recommendation = (
                f"Latency jitter is low across all metrics (worst CV={worst.stats.cv:.2f}). "
                f"Response times are predictable."
            )

        return JitterReport(
            metrics=metric_results,
            worst_metric=worst.metric,
            recommendation=recommendation,
        )


def analyze_jitter(data: BenchmarkData) -> dict:
    """Programmatic API for jitter analysis.

    Args:
        data: Benchmark data to analyze.

    Returns:
        Dictionary representation of the JitterReport.
    """
    analyzer = JitterAnalyzer()
    report = analyzer.analyze(data)
    return report.model_dump()

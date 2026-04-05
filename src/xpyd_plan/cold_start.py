"""Cold start detection — identify elevated latency in initial requests."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class ColdStartSeverity(str, Enum):
    """Severity of cold start effect."""

    NONE = "none"          # warmup P95 <= threshold * steady median
    MILD = "mild"          # ratio <= 2x threshold
    MODERATE = "moderate"  # ratio <= 4x threshold
    SEVERE = "severe"      # ratio > 4x threshold


class ColdStartWindow(BaseModel):
    """Cold start analysis for a single metric."""

    metric: str = Field(..., description="Metric name")
    warmup_p50: float = Field(..., description="P50 latency in warmup window")
    warmup_p95: float = Field(..., description="P95 latency in warmup window")
    steady_p50: float = Field(..., description="P50 latency in steady-state")
    steady_p95: float = Field(..., description="P95 latency in steady-state")
    ratio: float = Field(
        ..., description="warmup_p95 / steady_p50 (cold start ratio)"
    )
    stabilization_index: int = Field(
        ..., ge=0,
        description="Index of first request where running median stabilizes",
    )
    severity: ColdStartSeverity = Field(
        ..., description="Cold start severity classification"
    )


class ColdStartReport(BaseModel):
    """Complete cold start detection report."""

    total_requests: int = Field(..., description="Total requests analyzed")
    warmup_size: int = Field(..., description="Warmup window size used")
    threshold: float = Field(..., description="Detection threshold multiplier")
    metrics: list[ColdStartWindow] = Field(
        ..., description="Per-metric cold start results"
    )
    has_cold_start: bool = Field(
        ..., description="Whether any metric shows cold start"
    )
    worst_metric: str = Field(..., description="Metric with highest cold start ratio")
    recommendation: str = Field(..., description="Human-readable summary")


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


def _classify_severity(ratio: float, threshold: float) -> ColdStartSeverity:
    """Classify cold start severity based on ratio relative to threshold."""
    if ratio <= threshold:
        return ColdStartSeverity.NONE
    if ratio <= threshold * 2:
        return ColdStartSeverity.MILD
    if ratio <= threshold * 4:
        return ColdStartSeverity.MODERATE
    return ColdStartSeverity.SEVERE


def _find_stabilization(values: list[float], steady_median: float, threshold: float) -> int:
    """Find index where running median drops to within threshold of steady-state.

    Uses a sliding window of 5 requests. Returns 0 if already stable.
    """
    window_size = min(5, len(values))
    if window_size < 1:
        return 0

    for i in range(len(values) - window_size + 1):
        window = sorted(values[i : i + window_size])
        running_med = window[len(window) // 2]
        if steady_median > 0 and running_med / steady_median <= threshold:
            return i
    return len(values)


_METRICS = ["ttft_ms", "tpot_ms", "total_latency_ms"]


class ColdStartDetector:
    """Detect cold start effects in benchmark data."""

    def detect(
        self,
        data: BenchmarkData,
        warmup_window: int = 10,
        threshold: float = 2.0,
    ) -> ColdStartReport:
        """Run cold start detection.

        Args:
            data: Loaded benchmark data.
            warmup_window: Number of initial requests to consider as warmup.
            threshold: Multiplier — warmup P95 / steady P50 above this means cold start.

        Returns:
            ColdStartReport with per-metric analysis.

        Raises:
            ValueError: If fewer requests than warmup_window + 10.
        """
        requests = data.requests
        n = len(requests)
        min_required = warmup_window + 10
        if n < min_required:
            raise ValueError(
                f"Need at least {min_required} requests "
                f"(warmup_window={warmup_window} + 10 steady-state), got {n}"
            )

        metric_results: list[ColdStartWindow] = []
        for metric in _METRICS:
            all_values = [getattr(r, metric) for r in requests]
            warmup_values = all_values[:warmup_window]
            steady_values = all_values[warmup_window:]

            warmup_p50 = _percentile(warmup_values, 50.0)
            warmup_p95 = _percentile(warmup_values, 95.0)
            steady_p50 = _percentile(steady_values, 50.0)
            steady_p95 = _percentile(steady_values, 95.0)

            ratio = warmup_p95 / steady_p50 if steady_p50 > 0 else 0.0
            severity = _classify_severity(ratio, threshold)
            stab_idx = _find_stabilization(all_values, steady_p50, threshold)

            metric_results.append(
                ColdStartWindow(
                    metric=metric,
                    warmup_p50=round(warmup_p50, 4),
                    warmup_p95=round(warmup_p95, 4),
                    steady_p50=round(steady_p50, 4),
                    steady_p95=round(steady_p95, 4),
                    ratio=round(ratio, 4),
                    stabilization_index=stab_idx,
                    severity=severity,
                )
            )

        has_cold_start = any(
            m.severity != ColdStartSeverity.NONE for m in metric_results
        )
        worst = max(metric_results, key=lambda m: m.ratio)

        if not has_cold_start:
            recommendation = (
                "No significant cold start detected. "
                "Initial requests are within normal range."
            )
        elif worst.severity == ColdStartSeverity.SEVERE:
            recommendation = (
                f"Severe cold start on {worst.metric} "
                f"(warmup P95 is {worst.ratio:.1f}× steady-state P50). "
                f"Consider excluding first {worst.stabilization_index} requests "
                f"from analysis or adding a warmup phase to benchmarks."
            )
        else:
            recommendation = (
                f"Cold start detected on {worst.metric} "
                f"({worst.severity.value}, ratio={worst.ratio:.1f}×). "
                f"Stabilizes after ~{worst.stabilization_index} requests."
            )

        return ColdStartReport(
            total_requests=n,
            warmup_size=warmup_window,
            threshold=threshold,
            metrics=metric_results,
            has_cold_start=has_cold_start,
            worst_metric=worst.metric,
            recommendation=recommendation,
        )


def detect_cold_start(
    data: BenchmarkData,
    warmup_window: int = 10,
    threshold: float = 2.0,
) -> dict:
    """Programmatic API for cold start detection.

    Args:
        data: Benchmark data to analyze.
        warmup_window: Number of initial requests to treat as warmup.
        threshold: Detection threshold multiplier.

    Returns:
        Dictionary representation of ColdStartReport.
    """
    detector = ColdStartDetector()
    report = detector.detect(data, warmup_window=warmup_window, threshold=threshold)
    return report.model_dump()

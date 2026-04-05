"""SLA headroom calculator — measure safety margins between actual latency and SLA thresholds.

For each configured SLA metric (TTFT, TPOT, total latency), compute the absolute
and relative headroom at specified percentiles.  Identify the tightest metric
(smallest margin) and provide operational safety recommendations.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class SafetyLevel(str, Enum):
    """Safety level classification based on headroom."""

    CRITICAL = "critical"
    TIGHT = "tight"
    ADEQUATE = "adequate"
    COMFORTABLE = "comfortable"


class MetricHeadroom(BaseModel):
    """Headroom for a single metric at a given percentile."""

    metric: str = Field(..., description="Metric name (ttft, tpot, total_latency)")
    sla_threshold_ms: float = Field(..., description="SLA threshold in ms")
    actual_ms: float = Field(..., description="Actual latency at evaluated percentile")
    percentile: float = Field(..., description="Percentile evaluated")
    headroom_ms: float = Field(..., description="Absolute headroom (threshold - actual)")
    headroom_pct: float = Field(
        ..., description="Relative headroom as % of threshold"
    )
    passes_sla: bool = Field(..., description="Whether actual <= threshold")
    safety_level: SafetyLevel = Field(..., description="Safety classification")


class HeadroomReport(BaseModel):
    """Complete headroom analysis report."""

    metrics: list[MetricHeadroom] = Field(
        ..., description="Per-metric headroom details"
    )
    tightest_metric: str | None = Field(
        None, description="Metric with smallest relative headroom"
    )
    tightest_headroom_pct: float | None = Field(
        None, description="Smallest relative headroom %"
    )
    all_pass: bool = Field(..., description="All metrics within SLA")
    recommendation: str = Field(..., description="Operational recommendation")


def _percentile_value(values: list[float], pct: float) -> float:
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


def _classify_safety(headroom_pct: float, passes: bool) -> SafetyLevel:
    """Classify safety based on relative headroom."""
    if not passes:
        return SafetyLevel.CRITICAL
    if headroom_pct < 10.0:
        return SafetyLevel.TIGHT
    if headroom_pct < 30.0:
        return SafetyLevel.ADEQUATE
    return SafetyLevel.COMFORTABLE


class SLAHeadroomCalculator:
    """Calculate SLA headroom from benchmark data."""

    def calculate(
        self,
        data: BenchmarkData,
        *,
        sla_ttft_ms: float | None = None,
        sla_tpot_ms: float | None = None,
        sla_total_ms: float | None = None,
        percentile: float = 95.0,
    ) -> HeadroomReport:
        """Compute headroom for each configured SLA metric.

        Args:
            data: Benchmark data to analyze.
            sla_ttft_ms: TTFT SLA threshold in ms.
            sla_tpot_ms: TPOT SLA threshold in ms.
            sla_total_ms: Total latency SLA threshold in ms.
            percentile: Percentile to evaluate (default P95).

        Returns:
            HeadroomReport with per-metric margins and recommendations.
        """
        ttft_vals = [r.ttft_ms for r in data.requests]
        tpot_vals = [r.tpot_ms for r in data.requests]
        total_vals = [r.total_latency_ms for r in data.requests]

        metrics_config = []
        if sla_ttft_ms is not None:
            metrics_config.append(("ttft", sla_ttft_ms, ttft_vals))
        if sla_tpot_ms is not None:
            metrics_config.append(("tpot", sla_tpot_ms, tpot_vals))
        if sla_total_ms is not None:
            metrics_config.append(("total_latency", sla_total_ms, total_vals))

        if not metrics_config:
            return HeadroomReport(
                metrics=[],
                tightest_metric=None,
                tightest_headroom_pct=None,
                all_pass=True,
                recommendation="No SLA thresholds configured.",
            )

        results: list[MetricHeadroom] = []
        for name, threshold, values in metrics_config:
            actual = _percentile_value(values, percentile)
            headroom_ms = threshold - actual
            headroom_pct = (headroom_ms / threshold * 100.0) if threshold > 0 else 0.0
            passes = actual <= threshold
            safety = _classify_safety(headroom_pct, passes)
            results.append(
                MetricHeadroom(
                    metric=name,
                    sla_threshold_ms=threshold,
                    actual_ms=round(actual, 2),
                    percentile=percentile,
                    headroom_ms=round(headroom_ms, 2),
                    headroom_pct=round(headroom_pct, 2),
                    passes_sla=passes,
                    safety_level=safety,
                )
            )

        # Find tightest
        tightest = min(results, key=lambda m: m.headroom_pct)
        all_pass = all(m.passes_sla for m in results)

        # Generate recommendation
        if not all_pass:
            failing = [m.metric for m in results if not m.passes_sla]
            recommendation = (
                f"SLA VIOLATION on {', '.join(failing)}. "
                "Immediate action required: scale instances or relax SLA thresholds."
            )
        elif tightest.safety_level == SafetyLevel.TIGHT:
            recommendation = (
                f"Tight headroom on {tightest.metric} ({tightest.headroom_pct:.1f}%). "
                "Consider adding buffer capacity or relaxing thresholds."
            )
        elif tightest.safety_level == SafetyLevel.ADEQUATE:
            recommendation = (
                f"Adequate headroom. Tightest: {tightest.metric} "
                f"({tightest.headroom_pct:.1f}% margin)."
            )
        else:
            recommendation = (
                f"Comfortable headroom across all metrics. Tightest: "
                f"{tightest.metric} ({tightest.headroom_pct:.1f}% margin)."
            )

        return HeadroomReport(
            metrics=results,
            tightest_metric=tightest.metric,
            tightest_headroom_pct=tightest.headroom_pct,
            all_pass=all_pass,
            recommendation=recommendation,
        )


def analyze_sla_headroom(
    benchmark_path: str,
    *,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_ms: float | None = None,
    percentile: float = 95.0,
) -> dict:
    """Programmatic API for SLA headroom analysis.

    Args:
        benchmark_path: Path to benchmark JSON file.
        sla_ttft_ms: TTFT SLA threshold in ms.
        sla_tpot_ms: TPOT SLA threshold in ms.
        sla_total_ms: Total latency SLA threshold in ms.
        percentile: Percentile to evaluate.

    Returns:
        Dict representation of HeadroomReport.
    """
    from .bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    calc = SLAHeadroomCalculator()
    report = calc.calculate(
        data,
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_ms=sla_total_ms,
        percentile=percentile,
    )
    return report.model_dump()

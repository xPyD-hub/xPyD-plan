"""Benchmark Duration Advisor.

Analyze benchmark data to recommend minimum duration for reliable results.
Uses running percentile convergence to determine when metrics stabilize,
then advises whether the benchmark ran long enough and how long future
benchmarks should run.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class DurationVerdict(str, Enum):
    """Whether the benchmark ran long enough."""

    SUFFICIENT = "SUFFICIENT"
    MARGINAL = "MARGINAL"
    INSUFFICIENT = "INSUFFICIENT"
    TOO_SHORT = "TOO_SHORT"


class MetricStabilization(BaseModel):
    """Stabilization analysis for a single metric."""

    metric: str = Field(..., description="Metric name (e.g., ttft_ms)")
    stabilized: bool = Field(
        ..., description="Whether the metric stabilized within the benchmark"
    )
    stabilization_request_index: int | None = Field(
        None,
        description="Request index where metric stabilized (None if not stabilized)",
    )
    stabilization_time_s: float | None = Field(
        None,
        description="Elapsed time (seconds) when metric stabilized",
    )
    final_cv: float = Field(
        ...,
        description="Coefficient of variation of running P95 in last 20% of requests",
    )
    recommended_requests: int = Field(
        ...,
        description="Recommended minimum request count for this metric",
    )


class DurationAdvisorReport(BaseModel):
    """Result of benchmark duration analysis."""

    actual_duration_s: float = Field(..., description="Actual benchmark duration (s)")
    actual_request_count: int = Field(..., description="Total request count")
    recommended_duration_s: float = Field(
        ..., description="Recommended minimum benchmark duration (s)"
    )
    recommended_request_count: int = Field(
        ..., description="Recommended minimum request count"
    )
    verdict: DurationVerdict = Field(
        ..., description="Whether the benchmark was long enough"
    )
    metrics: list[MetricStabilization] = Field(
        ..., description="Per-metric stabilization details"
    )
    safety_multiplier: float = Field(
        default=1.5,
        description="Multiplier applied to stabilization point for recommendation",
    )


class DurationAdvisor:
    """Analyze benchmark duration adequacy.

    Computes running P95 over cumulative windows and checks for convergence.
    A metric is considered stabilized when the running P95 stays within
    a tolerance band (default ±5%) for a sustained tail window.

    Args:
        data: Benchmark data to analyze.
        percentile: Target percentile for convergence check (default 95).
        tolerance: Relative tolerance for stabilization (default 0.05 = 5%).
        window_steps: Number of cumulative windows to evaluate (default 20).
        safety_multiplier: Factor to multiply stabilization point for recommendation.
    """

    def __init__(
        self,
        data: BenchmarkData,
        *,
        percentile: float = 95.0,
        tolerance: float = 0.05,
        window_steps: int = 20,
        safety_multiplier: float = 1.5,
    ) -> None:
        self._data = data
        self._percentile = percentile
        self._tolerance = tolerance
        self._window_steps = max(5, window_steps)
        self._safety_multiplier = safety_multiplier

    def analyze(self) -> DurationAdvisorReport:
        """Run duration adequacy analysis."""
        requests = sorted(self._data.requests, key=lambda r: r.timestamp)
        n = len(requests)

        if n < 10:
            return DurationAdvisorReport(
                actual_duration_s=0.0,
                actual_request_count=n,
                recommended_duration_s=0.0,
                recommended_request_count=100,
                verdict=DurationVerdict.TOO_SHORT,
                metrics=[],
                safety_multiplier=self._safety_multiplier,
            )

        actual_duration = requests[-1].timestamp - requests[0].timestamp
        metric_names = ["ttft_ms", "tpot_ms", "total_latency_ms"]
        metric_results: list[MetricStabilization] = []

        for metric_name in metric_names:
            values = [getattr(r, metric_name) for r in requests]
            result = self._analyze_metric(
                metric_name, values, requests, actual_duration, n
            )
            metric_results.append(result)

        # Recommendation: max across all metrics
        rec_requests = max(m.recommended_requests for m in metric_results)
        if actual_duration > 0 and n > 0:
            qps = n / actual_duration
            rec_duration = rec_requests / qps if qps > 0 else 0.0
        else:
            rec_duration = 0.0

        # Verdict
        all_stabilized = all(m.stabilized for m in metric_results)
        any_stabilized = any(m.stabilized for m in metric_results)

        if all_stabilized and n >= rec_requests:
            verdict = DurationVerdict.SUFFICIENT
        elif all_stabilized:
            verdict = DurationVerdict.MARGINAL
        elif any_stabilized:
            verdict = DurationVerdict.MARGINAL
        else:
            verdict = DurationVerdict.INSUFFICIENT

        return DurationAdvisorReport(
            actual_duration_s=round(actual_duration, 2),
            actual_request_count=n,
            recommended_duration_s=round(rec_duration, 2),
            recommended_request_count=rec_requests,
            verdict=verdict,
            metrics=metric_results,
            safety_multiplier=self._safety_multiplier,
        )

    def _analyze_metric(
        self,
        metric_name: str,
        values: list[float],
        requests: list,
        actual_duration: float,
        n: int,
    ) -> MetricStabilization:
        """Analyze stabilization for one metric."""
        # Compute running percentile at each window step
        step_size = max(1, n // self._window_steps)
        running_pcts: list[tuple[int, float]] = []

        for i in range(1, self._window_steps + 1):
            end_idx = min(i * step_size, n)
            if end_idx < 5:
                continue
            pct_val = float(np.percentile(values[:end_idx], self._percentile))
            running_pcts.append((end_idx, pct_val))

        if len(running_pcts) < 3:
            return MetricStabilization(
                metric=metric_name,
                stabilized=False,
                stabilization_request_index=None,
                stabilization_time_s=None,
                final_cv=1.0,
                recommended_requests=n * 3,
            )

        # Check convergence: find first index where all subsequent windows
        # are within tolerance of the final value
        final_val = running_pcts[-1][1]
        if final_val == 0:
            # All zeros — trivially stable
            return MetricStabilization(
                metric=metric_name,
                stabilized=True,
                stabilization_request_index=running_pcts[0][0],
                stabilization_time_s=0.0,
                final_cv=0.0,
                recommended_requests=max(100, int(n * 0.5)),
            )

        stabilization_idx: int | None = None
        for start in range(len(running_pcts)):
            all_within = True
            for j in range(start, len(running_pcts)):
                rel_diff = abs(running_pcts[j][1] - final_val) / final_val
                if rel_diff > self._tolerance:
                    all_within = False
                    break
            if all_within:
                stabilization_idx = start
                break

        stabilized = stabilization_idx is not None

        # CV of last 20% of running percentiles
        tail_count = max(1, len(running_pcts) // 5)
        tail_vals = [p[1] for p in running_pcts[-tail_count:]]
        tail_mean = np.mean(tail_vals)
        tail_cv = float(np.std(tail_vals) / tail_mean) if tail_mean > 0 else 0.0

        if stabilized and stabilization_idx is not None:
            stab_request_idx = running_pcts[stabilization_idx][0]
            if actual_duration > 0:
                stab_time = actual_duration * (stab_request_idx / n)
            else:
                stab_time = 0.0
            rec_requests = max(
                100, int(math.ceil(stab_request_idx * self._safety_multiplier))
            )
        else:
            stab_request_idx = None
            stab_time = None
            # Not stabilized — recommend 3x current count
            rec_requests = n * 3

        return MetricStabilization(
            metric=metric_name,
            stabilized=stabilized,
            stabilization_request_index=stab_request_idx,
            stabilization_time_s=round(stab_time, 2) if stab_time is not None else None,
            final_cv=round(tail_cv, 4),
            recommended_requests=rec_requests,
        )


def advise_duration(
    data: BenchmarkData,
    *,
    percentile: float = 95.0,
    tolerance: float = 0.05,
    window_steps: int = 20,
    safety_multiplier: float = 1.5,
) -> dict:
    """Programmatic API for benchmark duration analysis.

    Args:
        data: Benchmark data to analyze.
        percentile: Target percentile (default 95).
        tolerance: Relative tolerance for stabilization (default 0.05).
        window_steps: Number of cumulative windows (default 20).
        safety_multiplier: Multiplier for safety margin (default 1.5).

    Returns:
        dict with analysis results.
    """
    advisor = DurationAdvisor(
        data,
        percentile=percentile,
        tolerance=tolerance,
        window_steps=window_steps,
        safety_multiplier=safety_multiplier,
    )
    report = advisor.analyze()
    return report.model_dump()

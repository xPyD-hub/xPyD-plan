"""Goodput analysis — effective throughput of SLA-compliant requests.

Goodput measures the rate of requests that actually meet SLA thresholds,
distinguishing useful work from raw throughput that includes SLA violations.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData, BenchmarkRequest


class FailureMetric(str, Enum):
    """Which SLA metric caused a request to fail."""

    TTFT = "ttft"
    TPOT = "tpot"
    TOTAL_LATENCY = "total_latency"


class FailureBreakdown(BaseModel):
    """Breakdown of SLA failures by metric."""

    ttft_failures: int = Field(0, ge=0, description="Requests failing TTFT SLA")
    tpot_failures: int = Field(0, ge=0, description="Requests failing TPOT SLA")
    total_latency_failures: int = Field(
        0, ge=0, description="Requests failing total latency SLA"
    )
    multi_metric_failures: int = Field(
        0, ge=0, description="Requests failing multiple SLA metrics"
    )


class GoodputWindow(BaseModel):
    """Goodput statistics for a single time window."""

    window_start: float = Field(..., description="Window start (epoch seconds)")
    window_end: float = Field(..., description="Window end (epoch seconds)")
    total_requests: int = Field(..., ge=0, description="Total requests in window")
    passing_requests: int = Field(..., ge=0, description="SLA-passing requests")
    goodput_ratio: float = Field(
        ..., ge=0, le=1, description="Fraction of SLA-passing requests"
    )
    raw_rps: float = Field(..., ge=0, description="Raw requests per second")
    goodput_rps: float = Field(..., ge=0, description="SLA-passing requests per second")


class GoodputGrade(str, Enum):
    """Overall goodput quality classification."""

    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"


class GoodputReport(BaseModel):
    """Complete goodput analysis report."""

    total_requests: int = Field(..., ge=0, description="Total requests analyzed")
    passing_requests: int = Field(
        ..., ge=0, description="Requests meeting all SLA thresholds"
    )
    failing_requests: int = Field(
        ..., ge=0, description="Requests violating at least one SLA threshold"
    )
    goodput_ratio: float = Field(
        ..., ge=0, le=1, description="Overall goodput ratio (passing / total)"
    )
    raw_qps: float = Field(..., ge=0, description="Raw measured QPS")
    goodput_qps: float = Field(
        ..., ge=0, description="Effective QPS (only SLA-passing requests)"
    )
    grade: GoodputGrade = Field(..., description="Goodput quality classification")
    failure_breakdown: FailureBreakdown = Field(
        ..., description="Breakdown of failures by SLA metric"
    )
    windows: list[GoodputWindow] = Field(
        default_factory=list, description="Per-window goodput tracking"
    )
    worst_window_goodput: float = Field(
        ..., ge=0, le=1, description="Worst goodput ratio across all windows"
    )
    sla_ttft_ms: float | None = Field(
        None, description="TTFT SLA threshold used (ms)"
    )
    sla_tpot_ms: float | None = Field(
        None, description="TPOT SLA threshold used (ms)"
    )
    sla_total_latency_ms: float | None = Field(
        None, description="Total latency SLA threshold used (ms)"
    )
    recommendation: str = Field(..., description="Human-readable recommendation")


def _grade_from_ratio(ratio: float) -> GoodputGrade:
    """Classify goodput ratio into a grade."""
    if ratio >= 0.99:
        return GoodputGrade.EXCELLENT
    if ratio >= 0.95:
        return GoodputGrade.GOOD
    if ratio >= 0.80:
        return GoodputGrade.FAIR
    return GoodputGrade.POOR


def _request_passes_sla(
    req: BenchmarkRequest,
    sla_ttft_ms: float | None,
    sla_tpot_ms: float | None,
    sla_total_latency_ms: float | None,
) -> tuple[bool, list[FailureMetric]]:
    """Check if a request meets all configured SLA thresholds.

    Returns (passes, list_of_failed_metrics).
    """
    failures: list[FailureMetric] = []
    if sla_ttft_ms is not None and req.ttft_ms > sla_ttft_ms:
        failures.append(FailureMetric.TTFT)
    if sla_tpot_ms is not None and req.tpot_ms > sla_tpot_ms:
        failures.append(FailureMetric.TPOT)
    if sla_total_latency_ms is not None and req.total_latency_ms > sla_total_latency_ms:
        failures.append(FailureMetric.TOTAL_LATENCY)
    return len(failures) == 0, failures


class GoodputAnalyzer:
    """Analyze effective throughput of SLA-compliant requests.

    Goodput = rate of requests that meet all configured SLA thresholds.
    """

    def __init__(
        self,
        sla_ttft_ms: float | None = None,
        sla_tpot_ms: float | None = None,
        sla_total_latency_ms: float | None = None,
        window_size: float = 5.0,
    ) -> None:
        if sla_ttft_ms is None and sla_tpot_ms is None and sla_total_latency_ms is None:
            raise ValueError(
                "At least one SLA threshold must be specified "
                "(sla_ttft_ms, sla_tpot_ms, or sla_total_latency_ms)"
            )
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._sla_ttft_ms = sla_ttft_ms
        self._sla_tpot_ms = sla_tpot_ms
        self._sla_total_latency_ms = sla_total_latency_ms
        self._window_size = window_size

    def analyze(self, data: BenchmarkData) -> GoodputReport:
        """Analyze goodput from benchmark data."""
        requests = data.requests
        total = len(requests)

        if total == 0:
            return GoodputReport(
                total_requests=0,
                passing_requests=0,
                failing_requests=0,
                goodput_ratio=1.0,
                raw_qps=0.0,
                goodput_qps=0.0,
                grade=GoodputGrade.EXCELLENT,
                failure_breakdown=FailureBreakdown(),
                windows=[],
                worst_window_goodput=1.0,
                sla_ttft_ms=self._sla_ttft_ms,
                sla_tpot_ms=self._sla_tpot_ms,
                sla_total_latency_ms=self._sla_total_latency_ms,
                recommendation="No requests to analyze.",
            )

        # Classify each request
        passing = 0
        ttft_fails = 0
        tpot_fails = 0
        total_lat_fails = 0
        multi_fails = 0
        pass_flags: list[bool] = []

        for req in requests:
            ok, failures = _request_passes_sla(
                req, self._sla_ttft_ms, self._sla_tpot_ms, self._sla_total_latency_ms
            )
            pass_flags.append(ok)
            if ok:
                passing += 1
            else:
                if FailureMetric.TTFT in failures:
                    ttft_fails += 1
                if FailureMetric.TPOT in failures:
                    tpot_fails += 1
                if FailureMetric.TOTAL_LATENCY in failures:
                    total_lat_fails += 1
                if len(failures) > 1:
                    multi_fails += 1

        failing = total - passing
        goodput_ratio = passing / total if total > 0 else 1.0
        raw_qps = data.metadata.measured_qps
        goodput_qps = raw_qps * goodput_ratio

        # Time-windowed analysis
        timestamps = np.array([r.timestamp for r in requests])
        t_min = float(np.min(timestamps))
        t_max = float(np.max(timestamps))
        duration = t_max - t_min

        windows: list[GoodputWindow] = []
        worst_window_goodput = 1.0

        if duration > 0:
            num_windows = max(1, math.ceil(duration / self._window_size))
            for i in range(num_windows):
                w_start = t_min + i * self._window_size
                w_end = w_start + self._window_size
                w_total = 0
                w_passing = 0
                for j, req in enumerate(requests):
                    if w_start <= req.timestamp < w_end:
                        w_total += 1
                        if pass_flags[j]:
                            w_passing += 1
                w_ratio = w_passing / w_total if w_total > 0 else 1.0
                w_raw_rps = w_total / self._window_size
                w_good_rps = w_passing / self._window_size
                windows.append(
                    GoodputWindow(
                        window_start=w_start,
                        window_end=w_end,
                        total_requests=w_total,
                        passing_requests=w_passing,
                        goodput_ratio=w_ratio,
                        raw_rps=w_raw_rps,
                        goodput_rps=w_good_rps,
                    )
                )
                if w_total > 0 and w_ratio < worst_window_goodput:
                    worst_window_goodput = w_ratio
        else:
            # All at same timestamp — single window
            w_ratio = goodput_ratio
            windows.append(
                GoodputWindow(
                    window_start=t_min,
                    window_end=t_min + self._window_size,
                    total_requests=total,
                    passing_requests=passing,
                    goodput_ratio=w_ratio,
                    raw_rps=total / self._window_size,
                    goodput_rps=passing / self._window_size,
                )
            )
            worst_window_goodput = w_ratio

        grade = _grade_from_ratio(goodput_ratio)

        # Recommendation
        if grade == GoodputGrade.EXCELLENT:
            rec = (
                f"Goodput ratio {goodput_ratio:.1%} is excellent. "
                f"Nearly all requests meet SLA thresholds."
            )
        elif grade == GoodputGrade.GOOD:
            rec = (
                f"Goodput ratio {goodput_ratio:.1%} is good. "
                f"{failing} of {total} requests violate SLA. "
                f"Minor tuning may improve compliance."
            )
        elif grade == GoodputGrade.FAIR:
            dominant = max(
                [("TTFT", ttft_fails), ("TPOT", tpot_fails), ("total_latency", total_lat_fails)],
                key=lambda x: x[1],
            )
            rec = (
                f"Goodput ratio {goodput_ratio:.1%} is fair. "
                f"{failing} of {total} requests fail SLA. "
                f"Dominant failure metric: {dominant[0]} ({dominant[1]} failures). "
                f"Consider scaling or relaxing SLA thresholds."
            )
        else:
            dominant = max(
                [("TTFT", ttft_fails), ("TPOT", tpot_fails), ("total_latency", total_lat_fails)],
                key=lambda x: x[1],
            )
            rec = (
                f"Goodput ratio {goodput_ratio:.1%} is poor. "
                f"{failing} of {total} requests fail SLA. "
                f"Dominant failure metric: {dominant[0]} ({dominant[1]} failures). "
                f"Significant intervention needed: scale instances, "
                f"optimize P:D ratio, or revise SLA targets."
            )

        return GoodputReport(
            total_requests=total,
            passing_requests=passing,
            failing_requests=failing,
            goodput_ratio=goodput_ratio,
            raw_qps=raw_qps,
            goodput_qps=goodput_qps,
            grade=grade,
            failure_breakdown=FailureBreakdown(
                ttft_failures=ttft_fails,
                tpot_failures=tpot_fails,
                total_latency_failures=total_lat_fails,
                multi_metric_failures=multi_fails,
            ),
            windows=windows,
            worst_window_goodput=worst_window_goodput,
            sla_ttft_ms=self._sla_ttft_ms,
            sla_tpot_ms=self._sla_tpot_ms,
            sla_total_latency_ms=self._sla_total_latency_ms,
            recommendation=rec,
        )


def analyze_goodput(
    benchmark_path: str,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_latency_ms: float | None = None,
    window_size: float = 5.0,
) -> dict:
    """Programmatic API for goodput analysis.

    Args:
        benchmark_path: Path to benchmark JSON file.
        sla_ttft_ms: TTFT SLA threshold in ms.
        sla_tpot_ms: TPOT SLA threshold in ms.
        sla_total_latency_ms: Total latency SLA threshold in ms.
        window_size: Time window size in seconds for windowed tracking.

    Returns:
        dict: GoodputReport as a dictionary.
    """
    from .bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    analyzer = GoodputAnalyzer(
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_latency_ms=sla_total_latency_ms,
        window_size=window_size,
    )
    report = analyzer.analyze(data)
    return report.model_dump()

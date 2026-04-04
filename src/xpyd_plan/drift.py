"""Distribution drift detection between benchmark runs using KS test."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class DriftSeverity(str, Enum):
    """Severity of distribution drift."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class DriftResult(BaseModel):
    """Drift detection result for a single metric."""

    metric: str = Field(..., description="Metric name (ttft, tpot, total_latency)")
    ks_statistic: float = Field(..., ge=0, le=1, description="KS test statistic (D)")
    p_value: float = Field(..., ge=0, le=1, description="KS test p-value")
    severity: DriftSeverity = Field(..., description="Drift severity classification")
    baseline_mean_ms: float = Field(..., ge=0, description="Baseline distribution mean")
    current_mean_ms: float = Field(..., ge=0, description="Current distribution mean")
    mean_shift_ms: float = Field(..., description="Mean shift (current - baseline)")


class DriftReport(BaseModel):
    """Complete drift detection report."""

    results: list[DriftResult] = Field(..., description="Per-metric drift results")
    overall_severity: DriftSeverity = Field(
        ..., description="Worst severity across all metrics"
    )
    drifted_metrics: list[str] = Field(
        ..., description="Metrics with severity > NONE"
    )
    recommendation: str = Field(..., description="Human-readable summary")


def _ks_two_sample(
    sample1: list[float], sample2: list[float]
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns (D statistic, approximate p-value).
    Uses the asymptotic Kolmogorov distribution approximation.
    """
    n1 = len(sample1)
    n2 = len(sample2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    sorted1 = sorted(sample1)
    sorted2 = sorted(sample2)

    # Merge and compute empirical CDFs
    all_values = sorted(set(sorted1 + sorted2))

    d_max = 0.0
    for v in all_values:
        # CDF of sample1 at v
        cdf1 = _bisect_right(sorted1, v) / n1
        cdf2 = _bisect_right(sorted2, v) / n2
        d_max = max(d_max, abs(cdf1 - cdf2))

    # Approximate p-value using Kolmogorov distribution
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d_max

    if lam == 0:
        p_value = 1.0
    else:
        # Kolmogorov survival function approximation
        p_value = 2.0 * sum(
            ((-1) ** (i - 1)) * math.exp(-2.0 * (i * lam) ** 2)
            for i in range(1, 101)
        )
        p_value = max(0.0, min(1.0, p_value))

    return round(d_max, 6), round(p_value, 6)


def _bisect_right(sorted_list: list[float], value: float) -> int:
    """Binary search: count of elements <= value."""
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _classify_severity(ks_d: float, p_value: float) -> DriftSeverity:
    """Classify drift severity from KS statistic and p-value."""
    if p_value > 0.05:
        return DriftSeverity.NONE
    if ks_d >= 0.4:
        return DriftSeverity.MAJOR
    if ks_d >= 0.2:
        return DriftSeverity.MODERATE
    return DriftSeverity.MINOR


_SEVERITY_ORDER = {
    DriftSeverity.NONE: 0,
    DriftSeverity.MINOR: 1,
    DriftSeverity.MODERATE: 2,
    DriftSeverity.MAJOR: 3,
}


class DriftDetector:
    """Detect distribution drift between two benchmark runs."""

    def detect(
        self, baseline: BenchmarkData, current: BenchmarkData
    ) -> DriftReport:
        """Run drift detection.

        Args:
            baseline: Reference benchmark data.
            current: New benchmark data to compare against baseline.

        Returns:
            DriftReport with per-metric KS test results.

        Raises:
            ValueError: If either dataset has fewer than 2 requests.
        """
        if len(baseline.requests) < 2:
            raise ValueError("Baseline must have at least 2 requests")
        if len(current.requests) < 2:
            raise ValueError("Current must have at least 2 requests")

        metrics = [
            ("ttft", "ttft_ms"),
            ("tpot", "tpot_ms"),
            ("total_latency", "total_latency_ms"),
        ]

        results: list[DriftResult] = []
        for metric_name, attr in metrics:
            baseline_vals = [getattr(r, attr) for r in baseline.requests]
            current_vals = [getattr(r, attr) for r in current.requests]

            ks_d, p_val = _ks_two_sample(baseline_vals, current_vals)
            severity = _classify_severity(ks_d, p_val)

            baseline_mean = sum(baseline_vals) / len(baseline_vals)
            current_mean = sum(current_vals) / len(current_vals)

            results.append(
                DriftResult(
                    metric=metric_name,
                    ks_statistic=ks_d,
                    p_value=p_val,
                    severity=severity,
                    baseline_mean_ms=round(baseline_mean, 3),
                    current_mean_ms=round(current_mean, 3),
                    mean_shift_ms=round(current_mean - baseline_mean, 3),
                )
            )

        drifted = [r.metric for r in results if r.severity != DriftSeverity.NONE]
        overall = max(results, key=lambda r: _SEVERITY_ORDER[r.severity]).severity

        recommendation = self._build_recommendation(results, drifted, overall)

        return DriftReport(
            results=results,
            overall_severity=overall,
            drifted_metrics=drifted,
            recommendation=recommendation,
        )

    def _build_recommendation(
        self,
        results: list[DriftResult],
        drifted: list[str],
        overall: DriftSeverity,
    ) -> str:
        """Build human-readable recommendation."""
        if overall == DriftSeverity.NONE:
            return (
                "No significant distribution drift detected. "
                "Latency distributions are statistically similar."
            )

        parts = [f"Distribution drift detected in: {', '.join(drifted)}."]

        for r in results:
            if r.severity != DriftSeverity.NONE:
                direction = "increased" if r.mean_shift_ms > 0 else "decreased"
                parts.append(
                    f"  {r.metric}: {r.severity.value} drift "
                    f"(D={r.ks_statistic:.3f}, p={r.p_value:.4f}), "
                    f"mean {direction} by {abs(r.mean_shift_ms):.1f}ms"
                )

        if overall == DriftSeverity.MAJOR:
            parts.append(
                "Action recommended: investigate root cause of major distribution shift."
            )
        elif overall == DriftSeverity.MODERATE:
            parts.append(
                "Consider investigating the distribution change and its impact on SLA."
            )

        return " ".join(parts)


def detect_drift(
    baseline: BenchmarkData,
    current: BenchmarkData,
) -> dict:
    """Programmatic API for drift detection.

    Args:
        baseline: Reference benchmark data.
        current: New benchmark data.

    Returns:
        Dict with drift detection results.
    """
    detector = DriftDetector()
    report = detector.detect(baseline, current)
    return report.model_dump()

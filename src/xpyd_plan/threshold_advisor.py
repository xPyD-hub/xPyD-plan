"""SLA Threshold Tuning Advisor — recommend optimal SLA targets from benchmark data.

Analyzes latency distributions and recommends SLA thresholds at various
pass-rate targets, identifying sweet spots where small threshold relaxation
yields large compliance gains.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class PassRateTarget(BaseModel):
    """A target pass rate for threshold computation."""

    pass_rate: float = Field(..., ge=0.0, le=1.0, description="Target pass rate (0-1)")


class ThresholdSuggestion(BaseModel):
    """A suggested threshold for a specific metric and pass rate."""

    metric: str = Field(..., description="Latency metric name (ttft_ms, tpot_ms, total_latency_ms)")
    pass_rate: float = Field(..., ge=0.0, le=1.0, description="Target pass rate")
    threshold_ms: float = Field(..., ge=0, description="Recommended threshold in ms")


class SweetSpot(BaseModel):
    """A sweet spot where small threshold change yields large compliance gain."""

    metric: str = Field(..., description="Latency metric name")
    threshold_ms: float = Field(..., ge=0, description="Threshold at the sweet spot")
    pass_rate_below: float = Field(
        ..., ge=0.0, le=1.0, description="Pass rate just below this threshold"
    )
    pass_rate_above: float = Field(
        ..., ge=0.0, le=1.0, description="Pass rate at/above this threshold"
    )
    gain: float = Field(..., description="Pass rate gain across the sweet spot")


class AdvisorReport(BaseModel):
    """Complete threshold advisor report."""

    total_requests: int = Field(..., ge=0)
    suggestions: list[ThresholdSuggestion] = Field(default_factory=list)
    sweet_spots: list[SweetSpot] = Field(default_factory=list)


class ThresholdAdvisor:
    """Analyze benchmark data and recommend SLA thresholds."""

    METRICS = ("ttft_ms", "tpot_ms", "total_latency_ms")

    def __init__(
        self,
        data: BenchmarkData,
        pass_rates: list[float] | None = None,
    ) -> None:
        self._data = data
        self._pass_rates = pass_rates or [0.90, 0.95, 0.99]
        # Validate pass rates
        for pr in self._pass_rates:
            if not 0.0 < pr <= 1.0:
                msg = f"pass_rate must be in (0, 1], got {pr}"
                raise ValueError(msg)

    def advise(self) -> AdvisorReport:
        """Generate threshold recommendations."""
        requests = self._data.requests
        if not requests:
            return AdvisorReport(total_requests=0)

        suggestions: list[ThresholdSuggestion] = []
        sweet_spots: list[SweetSpot] = []

        for metric in self.METRICS:
            values = np.array([getattr(r, metric) for r in requests])
            values.sort()

            # Compute thresholds for each pass rate
            for pr in self._pass_rates:
                # The threshold that achieves this pass rate is the percentile
                threshold = float(np.percentile(values, pr * 100))
                suggestions.append(
                    ThresholdSuggestion(
                        metric=metric,
                        pass_rate=pr,
                        threshold_ms=round(threshold, 2),
                    )
                )

            # Detect sweet spots: scan sorted values for large jumps in pass rate
            # relative to small threshold changes
            sweet_spots.extend(self._find_sweet_spots(metric, values))

        return AdvisorReport(
            total_requests=len(requests),
            suggestions=suggestions,
            sweet_spots=sweet_spots,
        )

    def _find_sweet_spots(
        self, metric: str, sorted_values: np.ndarray
    ) -> list[SweetSpot]:
        """Find inflection points where pass rate jumps significantly."""
        n = len(sorted_values)
        if n < 10:
            return []

        spots: list[SweetSpot] = []
        # Sample ~100 evenly spaced threshold points
        num_samples = min(100, n)
        indices = np.linspace(0, n - 1, num_samples, dtype=int)

        best_gain = 0.0
        best_spot: SweetSpot | None = None

        for i in range(1, len(indices)):
            idx_prev = indices[i - 1]
            idx_curr = indices[i]
            pr_prev = (idx_prev + 1) / n
            pr_curr = (idx_curr + 1) / n
            gain = pr_curr - pr_prev

            threshold_diff = sorted_values[idx_curr] - sorted_values[idx_prev]
            if threshold_diff <= 0:
                continue

            # Gain rate: pass-rate gain per ms of threshold relaxation
            gain_rate = gain / threshold_diff

            if gain >= 0.03 and gain_rate > 0.001:
                spot = SweetSpot(
                    metric=metric,
                    threshold_ms=round(float(sorted_values[idx_curr]), 2),
                    pass_rate_below=round(pr_prev, 4),
                    pass_rate_above=round(pr_curr, 4),
                    gain=round(gain, 4),
                )
                if gain > best_gain:
                    best_gain = gain
                    best_spot = spot

        if best_spot is not None:
            spots.append(best_spot)

        return spots


def advise_thresholds(
    data: BenchmarkData,
    pass_rates: list[float] | None = None,
) -> dict:
    """Programmatic API for threshold advising.

    Args:
        data: Benchmark data to analyze.
        pass_rates: Target pass rates (default: [0.90, 0.95, 0.99]).

    Returns:
        Dictionary with advisor report.
    """
    advisor = ThresholdAdvisor(data, pass_rates=pass_rates)
    report = advisor.advise()
    return report.model_dump()

"""Anomaly root cause analysis for SLA violations."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData, BenchmarkRequest


class FactorSignificance(str, Enum):
    """Statistical significance of a contributing factor."""

    HIGH = "high"  # p < 0.001
    MEDIUM = "medium"  # p < 0.01
    LOW = "low"  # p < 0.05
    NONE = "none"  # p >= 0.05


class CauseFactor(BaseModel):
    """A single contributing factor to SLA violations."""

    factor: str = Field(..., description="Factor name (e.g., prompt_tokens)")
    description: str = Field(..., description="Human-readable description")
    significance: FactorSignificance = Field(
        ..., description="Statistical significance level"
    )
    p_value: float = Field(..., ge=0, le=1, description="Mann-Whitney U p-value")
    effect_size: float = Field(
        ..., description="Cohen's d effect size (positive = higher in fail group)"
    )
    pass_mean: float = Field(..., description="Mean value in SLA-passing group")
    fail_mean: float = Field(..., description="Mean value in SLA-failing group")


class RootCauseReport(BaseModel):
    """Complete root cause analysis report."""

    total_requests: int = Field(..., ge=0, description="Total requests analyzed")
    passing_requests: int = Field(..., ge=0, description="Requests meeting SLA")
    failing_requests: int = Field(..., ge=0, description="Requests violating SLA")
    failure_rate: float = Field(
        ..., ge=0, le=1, description="Fraction of requests failing SLA"
    )
    sla_metric: str = Field(..., description="SLA metric used for segmentation")
    sla_threshold_ms: float = Field(..., gt=0, description="SLA threshold in ms")
    factors: list[CauseFactor] = Field(
        ..., description="Contributing factors ranked by significance"
    )
    recommendation: str = Field(..., description="Human-readable summary")


def _mann_whitney_u(
    sample1: list[float], sample2: list[float]
) -> tuple[float, float]:
    """Mann-Whitney U test.

    Returns (U statistic, approximate p-value using normal approximation).
    """
    n1 = len(sample1)
    n2 = len(sample2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Compute U statistic
    u1 = 0.0
    for x in sample1:
        for y in sample2:
            if x < y:
                u1 += 1.0
            elif x == y:
                u1 += 0.5

    # Normal approximation for p-value
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)

    if sigma == 0:
        return u1, 1.0

    z = (u1 - mu) / sigma
    # Two-tailed p-value using standard normal approximation
    p_value = 2.0 * _standard_normal_cdf(-abs(z))
    return round(u1, 6), round(min(1.0, max(0.0, p_value)), 6)


def _standard_normal_cdf(x: float) -> float:
    """Approximate CDF of standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size (group2 - group1)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1 = sum(group1) / n1
    mean2 = sum(group2) / n2

    var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0

    return round((mean2 - mean1) / pooled_std, 4)


def _classify_significance(p_value: float) -> FactorSignificance:
    """Classify significance from p-value."""
    if p_value < 0.001:
        return FactorSignificance.HIGH
    if p_value < 0.01:
        return FactorSignificance.MEDIUM
    if p_value < 0.05:
        return FactorSignificance.LOW
    return FactorSignificance.NONE


_SIGNIFICANCE_ORDER = {
    FactorSignificance.NONE: 0,
    FactorSignificance.LOW: 1,
    FactorSignificance.MEDIUM: 2,
    FactorSignificance.HIGH: 3,
}


class RootCauseAnalyzer:
    """Identify root causes of SLA violations in benchmark data."""

    def analyze(
        self,
        data: BenchmarkData,
        sla_metric: str = "ttft",
        sla_threshold_ms: float = 100.0,
    ) -> RootCauseReport:
        """Analyze root causes of SLA violations.

        Args:
            data: Benchmark data to analyze.
            sla_metric: Metric to check ("ttft", "tpot", "total_latency").
            sla_threshold_ms: SLA threshold in milliseconds.

        Returns:
            RootCauseReport with ranked contributing factors.

        Raises:
            ValueError: If sla_metric is invalid or data has < 2 requests.
        """
        metric_attr_map = {
            "ttft": "ttft_ms",
            "tpot": "tpot_ms",
            "total_latency": "total_latency_ms",
        }
        if sla_metric not in metric_attr_map:
            raise ValueError(
                f"Invalid sla_metric '{sla_metric}'. "
                f"Must be one of: {list(metric_attr_map.keys())}"
            )
        if len(data.requests) < 2:
            raise ValueError("Data must have at least 2 requests")

        attr = metric_attr_map[sla_metric]

        # Segment into pass/fail
        passing: list[BenchmarkRequest] = []
        failing: list[BenchmarkRequest] = []
        for r in data.requests:
            if getattr(r, attr) <= sla_threshold_ms:
                passing.append(r)
            else:
                failing.append(r)

        total = len(data.requests)
        n_pass = len(passing)
        n_fail = len(failing)
        failure_rate = n_fail / total

        factors: list[CauseFactor] = []

        if n_fail == 0 or n_pass == 0:
            recommendation = (
                "No SLA violations detected — nothing to analyze."
                if n_fail == 0
                else "All requests violate SLA — threshold may be too strict."
            )
            return RootCauseReport(
                total_requests=total,
                passing_requests=n_pass,
                failing_requests=n_fail,
                failure_rate=round(failure_rate, 4),
                sla_metric=sla_metric,
                sla_threshold_ms=sla_threshold_ms,
                factors=[],
                recommendation=recommendation,
            )

        # Analyze factors
        factor_defs = [
            ("prompt_tokens", "prompt_tokens", "Higher prompt token count"),
            ("output_tokens", "output_tokens", "Higher output token count"),
            ("timestamp", "timestamp", "Temporal position (later requests)"),
        ]

        # Also check other latency metrics as factors if not the target
        if sla_metric != "ttft":
            factor_defs.append(("ttft_ms", "ttft_ms", "Higher TTFT"))
        if sla_metric != "tpot":
            factor_defs.append(("tpot_ms", "tpot_ms", "Higher TPOT"))
        if sla_metric != "total_latency":
            factor_defs.append(
                (
                    "total_latency_ms",
                    "total_latency_ms",
                    "Higher total latency",
                )
            )

        for factor_name, attr_name, desc_prefix in factor_defs:
            pass_vals = [getattr(r, attr_name) for r in passing]
            fail_vals = [getattr(r, attr_name) for r in failing]

            _, p_value = _mann_whitney_u(pass_vals, fail_vals)
            effect = _cohens_d(pass_vals, fail_vals)
            significance = _classify_significance(p_value)

            pass_mean = sum(pass_vals) / len(pass_vals)
            fail_mean = sum(fail_vals) / len(fail_vals)

            direction = "higher" if fail_mean > pass_mean else "lower"
            description = (
                f"{desc_prefix}: failing requests have {direction} "
                f"{factor_name} (mean {fail_mean:.1f} vs {pass_mean:.1f})"
            )

            factors.append(
                CauseFactor(
                    factor=factor_name,
                    description=description,
                    significance=significance,
                    p_value=p_value,
                    effect_size=effect,
                    pass_mean=round(pass_mean, 3),
                    fail_mean=round(fail_mean, 3),
                )
            )

        # Sort by significance (descending), then absolute effect size
        factors.sort(
            key=lambda f: (
                _SIGNIFICANCE_ORDER[f.significance],
                abs(f.effect_size),
            ),
            reverse=True,
        )

        recommendation = self._build_recommendation(
            factors, n_fail, total, failure_rate, sla_metric, sla_threshold_ms
        )

        return RootCauseReport(
            total_requests=total,
            passing_requests=n_pass,
            failing_requests=n_fail,
            failure_rate=round(failure_rate, 4),
            sla_metric=sla_metric,
            sla_threshold_ms=sla_threshold_ms,
            factors=factors,
            recommendation=recommendation,
        )

    def _build_recommendation(
        self,
        factors: list[CauseFactor],
        n_fail: int,
        total: int,
        failure_rate: float,
        sla_metric: str,
        sla_threshold_ms: float,
    ) -> str:
        """Build human-readable recommendation."""
        significant = [
            f for f in factors if f.significance != FactorSignificance.NONE
        ]

        parts = [
            f"{n_fail}/{total} requests ({failure_rate:.1%}) violate "
            f"{sla_metric} SLA ({sla_threshold_ms}ms)."
        ]

        if not significant:
            parts.append(
                "No statistically significant contributing factors identified. "
                "Violations may be due to random variance."
            )
        else:
            parts.append("Contributing factors (by significance):")
            for f in significant:
                parts.append(
                    f"  - {f.factor} ({f.significance.value}, "
                    f"effect={f.effect_size:+.2f})"
                )

            top = significant[0]
            if top.significance == FactorSignificance.HIGH:
                parts.append(
                    f"Strong evidence that {top.factor} is a primary driver."
                )

        return " ".join(parts)


def analyze_root_cause(
    data: BenchmarkData,
    sla_metric: str = "ttft",
    sla_threshold_ms: float = 100.0,
) -> dict:
    """Programmatic API for root cause analysis.

    Args:
        data: Benchmark data.
        sla_metric: Metric to check ("ttft", "tpot", "total_latency").
        sla_threshold_ms: SLA threshold in milliseconds.

    Returns:
        Dict with root cause analysis results.
    """
    analyzer = RootCauseAnalyzer()
    report = analyzer.analyze(data, sla_metric, sla_threshold_ms)
    return report.model_dump()

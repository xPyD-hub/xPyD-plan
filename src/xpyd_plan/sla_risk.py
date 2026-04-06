"""SLA Risk Score — composite 0-100 risk assessment combining multiple signal dimensions.

Combines headroom tightness, tail heaviness, latency jitter, convergence adequacy,
and error budget burn rate into a single actionable risk score.
"""

from __future__ import annotations

import math
import statistics
from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class RiskLevel(str, Enum):
    """Risk level classification."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RiskFactor(BaseModel):
    """Individual risk dimension score."""

    name: str = Field(..., description="Risk factor name")
    score: float = Field(..., description="Factor score 0-100")
    weight: float = Field(..., description="Weight in composite score")
    weighted_score: float = Field(..., description="score * weight")
    detail: str = Field(..., description="Human-readable explanation")


class RiskScore(BaseModel):
    """Composite risk score."""

    total_score: float = Field(..., description="Composite risk score 0-100")
    risk_level: RiskLevel = Field(..., description="Overall risk classification")


class SLARiskReport(BaseModel):
    """Complete SLA risk assessment report."""

    risk_score: RiskScore = Field(..., description="Composite risk result")
    factors: list[RiskFactor] = Field(..., description="Per-factor breakdown")
    total_requests: int = Field(..., description="Requests analyzed")
    recommendation: str = Field(..., description="Actionable recommendation")


def _percentile_value(values: list[float], pct: float) -> float:
    """Compute percentile via linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    k = (pct / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _classify_risk(score: float) -> RiskLevel:
    if score >= 75.0:
        return RiskLevel.CRITICAL
    if score >= 50.0:
        return RiskLevel.HIGH
    if score >= 25.0:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def _headroom_score(
    values: list[float], threshold: float, percentile: float = 95.0,
) -> float:
    """Score 0-100 for headroom tightness. 100 = SLA violated, 0 = lots of margin."""
    if threshold <= 0:
        return 0.0
    actual = _percentile_value(values, percentile)
    headroom_pct = (threshold - actual) / threshold * 100.0
    if headroom_pct <= 0:
        return 100.0  # violated
    if headroom_pct >= 50:
        return 0.0
    # Linear: 0% headroom → 100, 50% headroom → 0
    return max(0.0, 100.0 - headroom_pct * 2.0)


def _tail_score(values: list[float]) -> float:
    """Score 0-100 for tail heaviness based on P99/P50 ratio."""
    if len(values) < 2:
        return 0.0
    p50 = _percentile_value(values, 50.0)
    p99 = _percentile_value(values, 99.0)
    if p50 <= 0:
        return 0.0
    ratio = p99 / p50
    # ratio < 2 → 0, ratio >= 10 → 100
    if ratio <= 2.0:
        return 0.0
    if ratio >= 10.0:
        return 100.0
    return (ratio - 2.0) / 8.0 * 100.0


def _jitter_score(values: list[float]) -> float:
    """Score 0-100 for latency jitter based on CV."""
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean <= 0:
        return 0.0
    std = statistics.stdev(values)
    cv = std / mean
    # CV < 0.3 → 0, CV >= 0.7 → 100
    if cv <= 0.3:
        return 0.0
    if cv >= 0.7:
        return 100.0
    return (cv - 0.3) / 0.4 * 100.0


def _convergence_score(values: list[float], steps: int = 10) -> float:
    """Score 0-100 for convergence adequacy. High = unstable percentiles."""
    n = len(values)
    if n < 10:
        return 100.0  # too few samples
    # Compute running P95 at progressive windows
    step_size = max(1, n // steps)
    p95_vals = []
    for i in range(1, steps + 1):
        end = min(i * step_size, n)
        window = values[:end]
        p95_vals.append(_percentile_value(window, 95.0))

    if len(p95_vals) < 2:
        return 0.0
    mean_p95 = statistics.mean(p95_vals)
    if mean_p95 <= 0:
        return 0.0
    cv = statistics.stdev(p95_vals) / mean_p95
    # CV < 0.05 → 0 (well converged), CV >= 0.25 → 100
    if cv <= 0.05:
        return 0.0
    if cv >= 0.25:
        return 100.0
    return (cv - 0.05) / 0.20 * 100.0


def _burn_rate_score(
    data: BenchmarkData,
    sla_ttft_ms: float | None,
    sla_tpot_ms: float | None,
    sla_total_ms: float | None,
) -> float:
    """Score 0-100 for error budget burn rate."""
    if not data.requests:
        return 0.0
    violations = 0
    for r in data.requests:
        if sla_ttft_ms is not None and r.ttft_ms > sla_ttft_ms:
            violations += 1
            continue
        if sla_tpot_ms is not None and r.tpot_ms > sla_tpot_ms:
            violations += 1
            continue
        if sla_total_ms is not None and r.total_latency_ms > sla_total_ms:
            violations += 1
            continue
    error_rate = violations / len(data.requests)
    # 0% → 0, >=10% → 100
    if error_rate <= 0:
        return 0.0
    if error_rate >= 0.10:
        return 100.0
    return error_rate / 0.10 * 100.0


class SLARiskScorer:
    """Composite SLA risk scorer combining multiple signal dimensions."""

    def assess(
        self,
        data: BenchmarkData,
        *,
        sla_ttft_ms: float | None = None,
        sla_tpot_ms: float | None = None,
        sla_total_ms: float | None = None,
        weight_headroom: float = 0.30,
        weight_tail: float = 0.20,
        weight_jitter: float = 0.20,
        weight_convergence: float = 0.15,
        weight_burn_rate: float = 0.15,
    ) -> SLARiskReport:
        """Assess SLA risk from benchmark data.

        Args:
            data: Benchmark data to analyze.
            sla_ttft_ms: TTFT SLA threshold in ms.
            sla_tpot_ms: TPOT SLA threshold in ms.
            sla_total_ms: Total latency SLA threshold in ms.
            weight_headroom: Weight for headroom factor (default 0.30).
            weight_tail: Weight for tail heaviness factor (default 0.20).
            weight_jitter: Weight for jitter factor (default 0.20).
            weight_convergence: Weight for convergence factor (default 0.15).
            weight_burn_rate: Weight for burn rate factor (default 0.15).

        Returns:
            SLARiskReport with composite score and per-factor breakdown.
        """
        ttft_vals = [r.ttft_ms for r in data.requests]
        tpot_vals = [r.tpot_ms for r in data.requests]
        total_vals = [r.total_latency_ms for r in data.requests]

        # 1. Headroom tightness — worst across configured metrics
        headroom_scores = []
        if sla_ttft_ms is not None:
            headroom_scores.append(_headroom_score(ttft_vals, sla_ttft_ms))
        if sla_tpot_ms is not None:
            headroom_scores.append(_headroom_score(tpot_vals, sla_tpot_ms))
        if sla_total_ms is not None:
            headroom_scores.append(_headroom_score(total_vals, sla_total_ms))
        h_score = max(headroom_scores) if headroom_scores else 0.0

        # 2. Tail heaviness — worst across all latency fields
        t_scores = [_tail_score(v) for v in [ttft_vals, tpot_vals, total_vals] if v]
        t_score = max(t_scores) if t_scores else 0.0

        # 3. Jitter — worst across all latency fields
        j_scores = [_jitter_score(v) for v in [ttft_vals, tpot_vals, total_vals] if v]
        j_score = max(j_scores) if j_scores else 0.0

        # 4. Convergence — worst across all fields
        c_scores = [
            _convergence_score(v)
            for v in [ttft_vals, tpot_vals, total_vals]
            if v
        ]
        c_score = max(c_scores) if c_scores else 0.0

        # 5. Burn rate
        b_score = _burn_rate_score(data, sla_ttft_ms, sla_tpot_ms, sla_total_ms)

        factors = [
            RiskFactor(
                name="headroom_tightness",
                score=round(h_score, 2),
                weight=weight_headroom,
                weighted_score=round(h_score * weight_headroom, 2),
                detail=self._headroom_detail(h_score),
            ),
            RiskFactor(
                name="tail_heaviness",
                score=round(t_score, 2),
                weight=weight_tail,
                weighted_score=round(t_score * weight_tail, 2),
                detail=self._tail_detail(t_score),
            ),
            RiskFactor(
                name="latency_jitter",
                score=round(j_score, 2),
                weight=weight_jitter,
                weighted_score=round(j_score * weight_jitter, 2),
                detail=self._jitter_detail(j_score),
            ),
            RiskFactor(
                name="convergence_adequacy",
                score=round(c_score, 2),
                weight=weight_convergence,
                weighted_score=round(c_score * weight_convergence, 2),
                detail=self._convergence_detail(c_score),
            ),
            RiskFactor(
                name="error_budget_burn_rate",
                score=round(b_score, 2),
                weight=weight_burn_rate,
                weighted_score=round(b_score * weight_burn_rate, 2),
                detail=self._burn_rate_detail(b_score),
            ),
        ]

        total = sum(f.weighted_score for f in factors)
        total = min(100.0, max(0.0, round(total, 2)))
        risk_level = _classify_risk(total)

        recommendation = self._generate_recommendation(factors, risk_level)

        return SLARiskReport(
            risk_score=RiskScore(total_score=total, risk_level=risk_level),
            factors=factors,
            total_requests=len(data.requests),
            recommendation=recommendation,
        )

    @staticmethod
    def _headroom_detail(score: float) -> str:
        if score >= 80:
            return "SLA threshold nearly or already breached"
        if score >= 40:
            return "Limited headroom — monitor closely"
        return "Comfortable SLA margin"

    @staticmethod
    def _tail_detail(score: float) -> str:
        if score >= 80:
            return "Extreme tail latency (P99/P50 ratio very high)"
        if score >= 40:
            return "Moderate tail heaviness"
        return "Light tail — latency distribution is tight"

    @staticmethod
    def _jitter_detail(score: float) -> str:
        if score >= 80:
            return "High latency variability (unstable)"
        if score >= 40:
            return "Moderate latency variability"
        return "Stable latency"

    @staticmethod
    def _convergence_detail(score: float) -> str:
        if score >= 80:
            return "Percentile estimates unreliable — insufficient data"
        if score >= 40:
            return "Marginal sample size — percentiles may shift"
        return "Sufficient data for reliable percentile estimates"

    @staticmethod
    def _burn_rate_detail(score: float) -> str:
        if score >= 80:
            return "High SLA violation rate — error budget depleting fast"
        if score >= 40:
            return "Noticeable SLA violations"
        return "Low or zero SLA violations"

    @staticmethod
    def _generate_recommendation(
        factors: list[RiskFactor], risk_level: RiskLevel,
    ) -> str:
        if risk_level == RiskLevel.LOW:
            return "Low risk. Current configuration is operating within safe margins."

        # Find dominant factor
        dominant = max(factors, key=lambda f: f.weighted_score)
        recs = {
            "headroom_tightness": (
                "Scale up instances or relax SLA thresholds to increase headroom."
            ),
            "tail_heaviness": (
                "Investigate tail latency causes (large requests, cold starts). "
                "Consider request size limits or prefill optimization."
            ),
            "latency_jitter": (
                "Investigate latency instability. Check for load spikes, "
                "resource contention, or thermal throttling."
            ),
            "convergence_adequacy": (
                "Collect more benchmark data. Current sample size is insufficient "
                "for reliable percentile estimates."
            ),
            "error_budget_burn_rate": (
                "SLA violations are consuming error budget. "
                "Scale up or rebalance P:D ratio to reduce violations."
            ),
        }

        prefix = {
            RiskLevel.MODERATE: "Moderate risk.",
            RiskLevel.HIGH: "High risk — action recommended.",
            RiskLevel.CRITICAL: "CRITICAL risk — immediate attention required.",
        }

        return (
            f"{prefix.get(risk_level, 'Risk detected.')} "
            f"Dominant factor: {dominant.name} (score {dominant.score:.0f}). "
            f"{recs.get(dominant.name, 'Review risk factors.')}"
        )


def assess_sla_risk(
    benchmark_path: str,
    *,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_ms: float | None = None,
    weight_headroom: float = 0.30,
    weight_tail: float = 0.20,
    weight_jitter: float = 0.20,
    weight_convergence: float = 0.15,
    weight_burn_rate: float = 0.15,
) -> dict:
    """Programmatic API for SLA risk assessment.

    Args:
        benchmark_path: Path to benchmark JSON file.
        sla_ttft_ms: TTFT SLA threshold in ms.
        sla_tpot_ms: TPOT SLA threshold in ms.
        sla_total_ms: Total latency SLA threshold in ms.
        weight_headroom: Weight for headroom factor.
        weight_tail: Weight for tail heaviness factor.
        weight_jitter: Weight for jitter factor.
        weight_convergence: Weight for convergence factor.
        weight_burn_rate: Weight for burn rate factor.

    Returns:
        Dict representation of SLARiskReport.
    """
    from .bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    scorer = SLARiskScorer()
    report = scorer.assess(
        data,
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_ms=sla_total_ms,
        weight_headroom=weight_headroom,
        weight_tail=weight_tail,
        weight_jitter=weight_jitter,
        weight_convergence=weight_convergence,
        weight_burn_rate=weight_burn_rate,
    )
    return report.model_dump()

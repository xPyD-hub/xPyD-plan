"""Deployment Readiness Report — unified go/no-go assessment for production deployment.

Combines quality gate, SLA risk score, SLA headroom, cost efficiency, and rate
limit headroom into a single deployment readiness verdict with actionable
recommendations.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData
from .quality_gate import GateVerdict, QualityGate
from .sla_headroom import SLAHeadroomCalculator
from .sla_risk import SLARiskScorer


class ReadinessVerdict(str, Enum):
    """Overall deployment readiness verdict."""

    READY = "ready"
    CAUTION = "caution"
    NOT_READY = "not_ready"


class CheckStatus(str, Enum):
    """Status of a single readiness check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class ReadinessCheck(BaseModel):
    """Result of a single readiness check."""

    name: str = Field(..., description="Check name")
    status: CheckStatus = Field(..., description="Check status")
    detail: str = Field(..., description="Human-readable explanation")
    value: str = Field(..., description="Measured value")
    threshold: str = Field(..., description="Threshold applied")


class ReadinessConfig(BaseModel):
    """Configuration for readiness assessment thresholds."""

    max_risk_score: float = Field(
        default=50.0, ge=0, le=100, description="Max acceptable SLA risk score"
    )
    risk_warn_score: float = Field(
        default=35.0, ge=0, le=100, description="Risk score warn threshold"
    )
    min_headroom_pct: float = Field(
        default=10.0, ge=0, description="Minimum SLA headroom percentage"
    )
    headroom_warn_pct: float = Field(
        default=20.0, ge=0, description="Headroom warn threshold percentage"
    )
    require_quality_gate: bool = Field(
        default=True, description="Require quality gate pass"
    )
    min_cost_efficiency: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum cost efficiency ratio"
    )
    cost_efficiency_warn: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Cost efficiency warn threshold"
    )
    min_rate_headroom_pct: float = Field(
        default=15.0, ge=0, description="Minimum rate limit headroom percentage"
    )
    rate_headroom_warn_pct: float = Field(
        default=25.0, ge=0, description="Rate headroom warn threshold percentage"
    )


def _derive_verdict(checks: list[ReadinessCheck]) -> ReadinessVerdict:
    """Derive overall verdict from individual check statuses."""
    statuses = {c.status for c in checks}
    if CheckStatus.FAIL in statuses:
        return ReadinessVerdict.NOT_READY
    if CheckStatus.WARN in statuses:
        return ReadinessVerdict.CAUTION
    return ReadinessVerdict.READY


class ReadinessReport(BaseModel):
    """Complete deployment readiness report."""

    verdict: ReadinessVerdict = Field(..., description="Overall deployment readiness")
    checks: list[ReadinessCheck] = Field(
        ..., description="Individual check results"
    )
    blockers: list[str] = Field(
        default_factory=list, description="Failing checks that block deployment"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Checks in warn status"
    )
    recommendation: str = Field(..., description="Actionable deployment recommendation")
    total_requests: int = Field(..., description="Requests in benchmark data")


class ReadinessAssessor:
    """Unified deployment readiness assessor."""

    def __init__(self, config: ReadinessConfig | None = None) -> None:
        self._config = config or ReadinessConfig()

    @property
    def config(self) -> ReadinessConfig:
        return self._config

    def assess(
        self,
        data: BenchmarkData,
        *,
        sla_ttft_ms: float | None = None,
        sla_tpot_ms: float | None = None,
        sla_total_ms: float | None = None,
        cost_per_request: float | None = None,
        optimal_cost_per_request: float | None = None,
        measured_qps: float | None = None,
        max_safe_qps: float | None = None,
    ) -> ReadinessReport:
        """Assess deployment readiness from benchmark data and optional context.

        Args:
            data: Benchmark data to analyze.
            sla_ttft_ms: TTFT SLA threshold in ms.
            sla_tpot_ms: TPOT SLA threshold in ms.
            sla_total_ms: Total latency SLA threshold in ms.
            cost_per_request: Actual cost per request (for cost efficiency check).
            optimal_cost_per_request: Optimal cost per request baseline.
            measured_qps: Current measured QPS.
            max_safe_qps: Maximum safe QPS from rate limit analysis.

        Returns:
            ReadinessReport with verdict and per-check details.
        """
        checks: list[ReadinessCheck] = []

        # 1. Quality gate check
        checks.append(self._check_quality_gate(data))

        # 2. SLA risk score check
        has_sla = any(x is not None for x in (sla_ttft_ms, sla_tpot_ms, sla_total_ms))
        if has_sla:
            checks.append(
                self._check_sla_risk(
                    data,
                    sla_ttft_ms=sla_ttft_ms,
                    sla_tpot_ms=sla_tpot_ms,
                    sla_total_ms=sla_total_ms,
                )
            )
            checks.append(
                self._check_sla_headroom(
                    data,
                    sla_ttft_ms=sla_ttft_ms,
                    sla_tpot_ms=sla_tpot_ms,
                    sla_total_ms=sla_total_ms,
                )
            )

        # 4. Cost efficiency check
        if cost_per_request is not None and optimal_cost_per_request is not None:
            checks.append(
                self._check_cost_efficiency(
                    cost_per_request, optimal_cost_per_request
                )
            )

        # 5. Rate limit headroom check
        if measured_qps is not None and max_safe_qps is not None:
            checks.append(
                self._check_rate_headroom(measured_qps, max_safe_qps)
            )

        verdict = _derive_verdict(checks)
        blockers = [c.name for c in checks if c.status == CheckStatus.FAIL]
        warnings = [c.name for c in checks if c.status == CheckStatus.WARN]
        recommendation = self._generate_recommendation(verdict, blockers, warnings)

        return ReadinessReport(
            verdict=verdict,
            checks=checks,
            blockers=blockers,
            warnings=warnings,
            recommendation=recommendation,
            total_requests=len(data.requests),
        )

    def _check_quality_gate(self, data: BenchmarkData) -> ReadinessCheck:
        """Run quality gate and return readiness check."""
        gate = QualityGate()
        result = gate.evaluate(data)
        verdict_val = result.verdict.value  # "pass", "warn", "fail"
        passed_count = sum(
            1 for c in result.checks
            if c.verdict == GateVerdict.PASS
        )
        total_count = len(result.checks)
        if verdict_val == GateVerdict.PASS.value:
            status = CheckStatus.PASS
        elif verdict_val == GateVerdict.WARN.value:
            status = CheckStatus.WARN
        else:
            status = CheckStatus.FAIL
        return ReadinessCheck(
            name="quality_gate",
            status=status,
            detail=(
                f"Quality gate: {result.verdict.value} "
                f"({passed_count}/{total_count} checks passed)"
            ),
            value=result.verdict.value,
            threshold="pass required" if self._config.require_quality_gate else "advisory",
        )

    def _check_sla_risk(
        self,
        data: BenchmarkData,
        *,
        sla_ttft_ms: float | None,
        sla_tpot_ms: float | None,
        sla_total_ms: float | None,
    ) -> ReadinessCheck:
        """Assess SLA risk score."""
        scorer = SLARiskScorer()
        report = scorer.assess(
            data,
            sla_ttft_ms=sla_ttft_ms,
            sla_tpot_ms=sla_tpot_ms,
            sla_total_ms=sla_total_ms,
        )
        score = report.risk_score.total_score
        cfg = self._config
        if score > cfg.max_risk_score:
            status = CheckStatus.FAIL
        elif score > cfg.risk_warn_score:
            status = CheckStatus.WARN
        else:
            status = CheckStatus.PASS
        return ReadinessCheck(
            name="sla_risk_score",
            status=status,
            detail=f"SLA risk score: {score:.1f}/100 ({report.risk_score.risk_level.value})",
            value=f"{score:.1f}",
            threshold=f"fail>{cfg.max_risk_score}, warn>{cfg.risk_warn_score}",
        )

    def _check_sla_headroom(
        self,
        data: BenchmarkData,
        *,
        sla_ttft_ms: float | None,
        sla_tpot_ms: float | None,
        sla_total_ms: float | None,
    ) -> ReadinessCheck:
        """Check SLA headroom safety level."""
        report = SLAHeadroomCalculator().calculate(
            data,
            sla_ttft_ms=sla_ttft_ms,
            sla_tpot_ms=sla_tpot_ms,
            sla_total_ms=sla_total_ms,
        )
        # Find tightest headroom
        if not report.metrics:
            return ReadinessCheck(
                name="sla_headroom",
                status=CheckStatus.PASS,
                detail="No SLA metrics configured for headroom analysis",
                value="N/A",
                threshold=f"min {self._config.min_headroom_pct}%",
            )

        tightest = min(report.metrics, key=lambda m: m.headroom_pct)
        pct = tightest.headroom_pct
        cfg = self._config

        if not tightest.passes_sla or pct < cfg.min_headroom_pct:
            status = CheckStatus.FAIL
        elif pct < cfg.headroom_warn_pct:
            status = CheckStatus.WARN
        else:
            status = CheckStatus.PASS

        detail = (
            f"Tightest headroom: {tightest.metric} at P{tightest.percentile:.0f} "
            f"— {pct:.1f}% ({tightest.safety_level.value})"
        )
        return ReadinessCheck(
            name="sla_headroom",
            status=status,
            detail=detail,
            value=f"{pct:.1f}%",
            threshold=f"fail<{cfg.min_headroom_pct}%, warn<{cfg.headroom_warn_pct}%",
        )

    def _check_cost_efficiency(
        self,
        cost_per_request: float,
        optimal_cost: float,
    ) -> ReadinessCheck:
        """Check cost efficiency ratio (optimal / actual)."""
        if cost_per_request <= 0:
            return ReadinessCheck(
                name="cost_efficiency",
                status=CheckStatus.PASS,
                detail="Cost per request is zero or negative — skipped",
                value="N/A",
                threshold="N/A",
            )
        ratio = optimal_cost / cost_per_request if cost_per_request > 0 else 1.0
        ratio = min(ratio, 1.0)  # cap at 1.0
        cfg = self._config
        if ratio < cfg.min_cost_efficiency:
            status = CheckStatus.FAIL
        elif ratio < cfg.cost_efficiency_warn:
            status = CheckStatus.WARN
        else:
            status = CheckStatus.PASS
        return ReadinessCheck(
            name="cost_efficiency",
            status=status,
            detail=f"Cost efficiency: {ratio:.2f} (optimal/actual cost ratio)",
            value=f"{ratio:.2f}",
            threshold=f"fail<{cfg.min_cost_efficiency}, warn<{cfg.cost_efficiency_warn}",
        )

    def _check_rate_headroom(
        self,
        measured_qps: float,
        max_safe_qps: float,
    ) -> ReadinessCheck:
        """Check rate limit headroom."""
        if max_safe_qps <= 0:
            return ReadinessCheck(
                name="rate_headroom",
                status=CheckStatus.FAIL,
                detail="Max safe QPS is zero — no headroom",
                value="0.0%",
                threshold=f"min {self._config.min_rate_headroom_pct}%",
            )
        headroom_pct = ((max_safe_qps - measured_qps) / max_safe_qps) * 100
        cfg = self._config
        if headroom_pct < cfg.min_rate_headroom_pct:
            status = CheckStatus.FAIL
        elif headroom_pct < cfg.rate_headroom_warn_pct:
            status = CheckStatus.WARN
        else:
            status = CheckStatus.PASS
        return ReadinessCheck(
            name="rate_headroom",
            status=status,
            detail=(
                f"Rate headroom: {headroom_pct:.1f}% "
                f"(measured {measured_qps:.1f} vs max safe {max_safe_qps:.1f} QPS)"
            ),
            value=f"{headroom_pct:.1f}%",
            threshold=f"fail<{cfg.min_rate_headroom_pct}%, warn<{cfg.rate_headroom_warn_pct}%",
        )

    @staticmethod
    def _generate_recommendation(
        verdict: ReadinessVerdict,
        blockers: list[str],
        warnings: list[str],
    ) -> str:
        """Generate actionable recommendation from verdict."""
        if verdict == ReadinessVerdict.READY:
            return "All checks passed. System is ready for production deployment."
        if verdict == ReadinessVerdict.NOT_READY:
            blocker_str = ", ".join(blockers)
            return (
                f"Deployment blocked by failing checks: {blocker_str}. "
                "Address these issues before deploying to production."
            )
        # CAUTION
        warn_str = ", ".join(warnings)
        return (
            f"Deployment possible with caution — warnings on: {warn_str}. "
            "Consider addressing these before high-traffic deployment."
        )


def assess_readiness(
    benchmark_path: str,
    *,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_ms: float | None = None,
    cost_per_request: float | None = None,
    optimal_cost_per_request: float | None = None,
    measured_qps: float | None = None,
    max_safe_qps: float | None = None,
    config: ReadinessConfig | None = None,
) -> ReadinessReport:
    """Convenience function: load benchmark and assess readiness.

    Args:
        benchmark_path: Path to benchmark JSON file.
        sla_ttft_ms: TTFT SLA threshold in ms.
        sla_tpot_ms: TPOT SLA threshold in ms.
        sla_total_ms: Total latency SLA threshold in ms.
        cost_per_request: Actual cost per request.
        optimal_cost_per_request: Optimal cost per request baseline.
        measured_qps: Current measured QPS.
        max_safe_qps: Maximum safe QPS from rate limit analysis.
        config: Optional readiness config overrides.

    Returns:
        ReadinessReport with verdict, checks, and recommendation.
    """
    from .bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    assessor = ReadinessAssessor(config=config)
    return assessor.assess(
        data,
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_ms=sla_total_ms,
        cost_per_request=cost_per_request,
        optimal_cost_per_request=optimal_cost_per_request,
        measured_qps=measured_qps,
        max_safe_qps=max_safe_qps,
    )

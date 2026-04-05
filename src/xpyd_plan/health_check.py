"""Benchmark health check — composite readiness assessment.

Runs data validation, percentile convergence, load profile classification,
and outlier impact analysis to produce a single PASS/WARN/FAIL verdict.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.convergence import ConvergenceAnalyzer, StabilityStatus
from xpyd_plan.load_profile import LoadProfileClassifier, ProfileType
from xpyd_plan.outlier_impact import ImpactRecommendation, OutlierImpactAnalyzer
from xpyd_plan.validator import DataValidator


class HealthStatus(str, Enum):
    """Overall health status."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class CheckResult(BaseModel):
    """Result of a single health check."""

    name: str = Field(description="Check name")
    status: HealthStatus = Field(description="Check status")
    detail: str = Field(description="Human-readable detail")


class HealthReport(BaseModel):
    """Complete health check report."""

    overall: HealthStatus = Field(description="Overall verdict")
    checks: list[CheckResult] = Field(description="Per-check results")
    request_count: int = Field(description="Total requests in benchmark")


class HealthChecker:
    """Run composite health checks on benchmark data.

    Checks:
    1. Data validation (completeness, consistency, outlier ratio)
    2. Percentile convergence (sample size adequacy)
    3. Load profile (reject unstable profiles)
    4. Outlier impact (flag if outliers distort P95+)
    """

    def __init__(
        self,
        *,
        convergence_steps: int = 10,
        convergence_threshold: float = 0.05,
        load_profile_window: float = 5.0,
        iqr_multiplier: float = 1.5,
    ) -> None:
        self._convergence_steps = convergence_steps
        self._convergence_threshold = convergence_threshold
        self._load_profile_window = load_profile_window
        self._iqr_multiplier = iqr_multiplier

    def check(self, data: BenchmarkData) -> HealthReport:
        """Run all health checks and return unified report."""
        checks: list[CheckResult] = []

        checks.append(self._check_validation(data))
        checks.append(self._check_convergence(data))
        checks.append(self._check_load_profile(data))
        checks.append(self._check_outlier_impact(data))

        # Determine overall: FAIL if any FAIL, WARN if any WARN, else PASS
        statuses = [c.status for c in checks]
        if HealthStatus.FAIL in statuses:
            overall = HealthStatus.FAIL
        elif HealthStatus.WARN in statuses:
            overall = HealthStatus.WARN
        else:
            overall = HealthStatus.PASS

        return HealthReport(
            overall=overall,
            checks=checks,
            request_count=len(data.requests),
        )

    def _check_validation(self, data: BenchmarkData) -> CheckResult:
        """Check data quality via DataValidator."""
        validator = DataValidator()
        result = validator.validate(data)

        quality = result.quality.overall
        if quality >= 0.8:
            status = HealthStatus.PASS
            detail = f"Data quality score: {quality:.2f} (good)"
        elif quality >= 0.5:
            status = HealthStatus.WARN
            detail = (
                f"Data quality score: {quality:.2f} — "
                f"completeness={result.quality.completeness:.2f}, "
                f"consistency={result.quality.consistency:.2f}, "
                f"outlier_ratio={result.quality.outlier_ratio:.2f}"
            )
        else:
            status = HealthStatus.FAIL
            detail = (
                f"Data quality score: {quality:.2f} (poor) — "
                f"completeness={result.quality.completeness:.2f}, "
                f"consistency={result.quality.consistency:.2f}, "
                f"outlier_ratio={result.quality.outlier_ratio:.2f}"
            )

        return CheckResult(name="validation", status=status, detail=detail)

    def _check_convergence(self, data: BenchmarkData) -> CheckResult:
        """Check percentile convergence (sample size adequacy)."""
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze(
            steps=self._convergence_steps,
            threshold=self._convergence_threshold,
        )

        if report.overall_status == StabilityStatus.STABLE:
            status = HealthStatus.PASS
            detail = "Percentile convergence: stable"
        elif report.overall_status == StabilityStatus.MARGINAL:
            status = HealthStatus.WARN
            unstable = [m.field for m in report.metrics if m.status != StabilityStatus.STABLE]
            detail = f"Percentile convergence: marginal — unstable metrics: {', '.join(unstable)}"
        else:
            status = HealthStatus.FAIL
            unstable = [m.field for m in report.metrics if m.status == StabilityStatus.UNSTABLE]
            detail = f"Percentile convergence: unstable — metrics: {', '.join(unstable)}"

        return CheckResult(name="convergence", status=status, detail=detail)

    def _check_load_profile(self, data: BenchmarkData) -> CheckResult:
        """Check load profile stability."""
        classifier = LoadProfileClassifier(data)
        report = classifier.classify(window_size=self._load_profile_window)
        profile_type = report.profile.profile_type

        # Steady state and ramp are acceptable; burst/cyclic/unknown are warnings or failures
        if profile_type == ProfileType.STEADY_STATE:
            status = HealthStatus.PASS
            detail = f"Load profile: {profile_type.value} (stable)"
        elif profile_type in (ProfileType.RAMP_UP, ProfileType.RAMP_DOWN):
            status = HealthStatus.WARN
            detail = f"Load profile: {profile_type.value} — non-steady traffic pattern"
        elif profile_type == ProfileType.UNKNOWN:
            status = HealthStatus.WARN
            detail = f"Load profile: {profile_type.value} — could not classify traffic pattern"
        else:
            # BURST or CYCLIC
            status = HealthStatus.WARN
            detail = f"Load profile: {profile_type.value} — irregular traffic pattern"

        return CheckResult(name="load_profile", status=status, detail=detail)

    def _check_outlier_impact(self, data: BenchmarkData) -> CheckResult:
        """Check whether outliers materially affect analysis."""
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data, iqr_multiplier=self._iqr_multiplier)

        pct = (
            report.outlier_count / report.total_requests * 100
            if report.total_requests > 0
            else 0.0
        )

        if report.recommendation == ImpactRecommendation.KEEP:
            status = HealthStatus.PASS
            detail = (
                f"Outlier impact: negligible "
                f"({report.outlier_count}/{report.total_requests}, {pct:.1f}%)"
            )
        else:
            # FILTER recommendation means outliers distort results
            if pct > 10:
                status = HealthStatus.FAIL
                detail = (
                    f"Outlier impact: severe "
                    f"({report.outlier_count}/{report.total_requests}, "
                    f"{pct:.1f}%) — recommend filtering"
                )
            else:
                status = HealthStatus.WARN
                detail = (
                    f"Outlier impact: moderate "
                    f"({report.outlier_count}/{report.total_requests}, "
                    f"{pct:.1f}%) — consider filtering"
                )

        return CheckResult(name="outlier_impact", status=status, detail=detail)


def check_health(
    benchmark_path: str,
    *,
    convergence_steps: int = 10,
    convergence_threshold: float = 0.05,
    load_profile_window: float = 5.0,
    iqr_multiplier: float = 1.5,
) -> dict:
    """Programmatic API for benchmark health check.

    Args:
        benchmark_path: Path to benchmark JSON file.
        convergence_steps: Number of convergence windows.
        convergence_threshold: CV threshold for convergence stability.
        load_profile_window: Window size in seconds for load profile.
        iqr_multiplier: IQR multiplier for outlier detection.

    Returns:
        Dict with health report fields.
    """
    data = load_benchmark_auto(benchmark_path)
    checker = HealthChecker(
        convergence_steps=convergence_steps,
        convergence_threshold=convergence_threshold,
        load_profile_window=load_profile_window,
        iqr_multiplier=iqr_multiplier,
    )
    report = checker.check(data)
    return report.model_dump()

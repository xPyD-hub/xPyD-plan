"""Benchmark Quality Gate — composite pass/fail gate for CI/CD pipelines.

Combines multiple quality checks (validation, convergence, load profile,
outlier ratio, minimum request count) into a single gate verdict with
YAML-configurable thresholds.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.convergence import ConvergenceAnalyzer, StabilityStatus
from xpyd_plan.load_profile import LoadProfileClassifier
from xpyd_plan.outlier_impact import OutlierImpactAnalyzer
from xpyd_plan.validator import DataValidator


class GateVerdict(str, Enum):
    """Overall gate verdict."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class GateCheck(BaseModel):
    """Result of a single gate check."""

    name: str = Field(description="Check name")
    verdict: GateVerdict = Field(description="Check verdict")
    detail: str = Field(description="Human-readable detail")
    threshold: Optional[str] = Field(
        default=None, description="Threshold that was evaluated"
    )


class GateConfig(BaseModel):
    """Configuration for quality gate thresholds."""

    min_requests: int = Field(
        default=100, ge=1, description="Minimum number of requests required"
    )
    min_quality_score: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum data quality score"
    )
    max_outlier_pct: float = Field(
        default=10.0, ge=0.0, le=100.0, description="Maximum outlier percentage"
    )
    require_stable_convergence: bool = Field(
        default=True, description="Require stable percentile convergence"
    )
    allowed_load_profiles: list[str] = Field(
        default_factory=lambda: ["steady_state", "ramp_up", "ramp_down"],
        description="Allowed load profile types",
    )
    convergence_steps: int = Field(default=10, ge=2, description="Convergence window steps")
    convergence_threshold: float = Field(
        default=0.05, ge=0.0, description="CV threshold for convergence"
    )
    iqr_multiplier: float = Field(
        default=1.5, ge=0.0, description="IQR multiplier for outlier detection"
    )
    load_profile_window: float = Field(
        default=5.0, ge=0.1, description="Window size for load profile (seconds)"
    )


class GateResult(BaseModel):
    """Complete quality gate result."""

    verdict: GateVerdict = Field(description="Overall gate verdict")
    checks: list[GateCheck] = Field(description="Per-check results")
    request_count: int = Field(description="Total requests in benchmark")
    config: GateConfig = Field(description="Config used for evaluation")
    passed: bool = Field(description="True if verdict is PASS")


def load_gate_config(path: str) -> GateConfig:
    """Load gate config from a YAML file.

    Args:
        path: Path to YAML config file.

    Returns:
        Parsed GateConfig.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return GateConfig()
    return GateConfig(**raw)


class QualityGate:
    """Evaluate benchmark data against configurable quality checks.

    Checks:
    1. Minimum request count
    2. Data validation quality score
    3. Outlier percentage
    4. Percentile convergence stability
    5. Load profile type
    """

    def __init__(self, config: GateConfig | None = None) -> None:
        self._config = config or GateConfig()

    @property
    def config(self) -> GateConfig:
        """Return current gate configuration."""
        return self._config

    def evaluate(self, data: BenchmarkData) -> GateResult:
        """Evaluate all gate checks and return unified result."""
        checks: list[GateCheck] = []

        checks.append(self._check_min_requests(data))
        checks.append(self._check_validation(data))
        checks.append(self._check_outlier_ratio(data))
        checks.append(self._check_convergence(data))
        checks.append(self._check_load_profile(data))

        verdicts = [c.verdict for c in checks]
        if GateVerdict.FAIL in verdicts:
            overall = GateVerdict.FAIL
        elif GateVerdict.WARN in verdicts:
            overall = GateVerdict.WARN
        else:
            overall = GateVerdict.PASS

        return GateResult(
            verdict=overall,
            checks=checks,
            request_count=len(data.requests),
            config=self._config,
            passed=overall == GateVerdict.PASS,
        )

    def _check_min_requests(self, data: BenchmarkData) -> GateCheck:
        """Check minimum request count."""
        count = len(data.requests)
        threshold = self._config.min_requests
        if count >= threshold:
            return GateCheck(
                name="min_requests",
                verdict=GateVerdict.PASS,
                detail=f"{count} requests (≥{threshold})",
                threshold=str(threshold),
            )
        return GateCheck(
            name="min_requests",
            verdict=GateVerdict.FAIL,
            detail=f"{count} requests (required ≥{threshold})",
            threshold=str(threshold),
        )

    def _check_validation(self, data: BenchmarkData) -> GateCheck:
        """Check data quality score."""
        validator = DataValidator()
        result = validator.validate(data)
        score = result.quality.overall
        threshold = self._config.min_quality_score

        if score >= threshold:
            return GateCheck(
                name="quality_score",
                verdict=GateVerdict.PASS,
                detail=f"Quality score {score:.2f} (≥{threshold:.2f})",
                threshold=str(threshold),
            )
        # Warn if close, fail if far
        if score >= threshold * 0.8:
            return GateCheck(
                name="quality_score",
                verdict=GateVerdict.WARN,
                detail=f"Quality score {score:.2f} (threshold {threshold:.2f})",
                threshold=str(threshold),
            )
        return GateCheck(
            name="quality_score",
            verdict=GateVerdict.FAIL,
            detail=f"Quality score {score:.2f} (required ≥{threshold:.2f})",
            threshold=str(threshold),
        )

    def _check_outlier_ratio(self, data: BenchmarkData) -> GateCheck:
        """Check outlier percentage."""
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data, iqr_multiplier=self._config.iqr_multiplier)

        pct = (
            report.outlier_count / report.total_requests * 100
            if report.total_requests > 0
            else 0.0
        )
        max_pct = self._config.max_outlier_pct

        if pct <= max_pct:
            return GateCheck(
                name="outlier_ratio",
                verdict=GateVerdict.PASS,
                detail=f"Outlier ratio {pct:.1f}% (≤{max_pct:.1f}%)",
                threshold=f"{max_pct}%",
            )
        if pct <= max_pct * 1.5:
            return GateCheck(
                name="outlier_ratio",
                verdict=GateVerdict.WARN,
                detail=f"Outlier ratio {pct:.1f}% (threshold {max_pct:.1f}%)",
                threshold=f"{max_pct}%",
            )
        return GateCheck(
            name="outlier_ratio",
            verdict=GateVerdict.FAIL,
            detail=f"Outlier ratio {pct:.1f}% (max {max_pct:.1f}%)",
            threshold=f"{max_pct}%",
        )

    def _check_convergence(self, data: BenchmarkData) -> GateCheck:
        """Check percentile convergence stability."""
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze(
            steps=self._config.convergence_steps,
            threshold=self._config.convergence_threshold,
        )

        if report.overall_status == StabilityStatus.STABLE:
            return GateCheck(
                name="convergence",
                verdict=GateVerdict.PASS,
                detail="Percentile convergence: stable",
            )
        if report.overall_status == StabilityStatus.MARGINAL:
            unstable = [
                m.field
                for m in report.metrics
                if m.status != StabilityStatus.STABLE
            ]
            verdict = (
                GateVerdict.WARN
                if not self._config.require_stable_convergence
                else GateVerdict.FAIL
            )
            return GateCheck(
                name="convergence",
                verdict=verdict,
                detail=f"Percentile convergence: marginal ({', '.join(unstable)})",
            )

        # UNSTABLE
        unstable = [
            m.field
            for m in report.metrics
            if m.status == StabilityStatus.UNSTABLE
        ]
        return GateCheck(
            name="convergence",
            verdict=GateVerdict.FAIL,
            detail=f"Percentile convergence: unstable ({', '.join(unstable)})",
        )

    def _check_load_profile(self, data: BenchmarkData) -> GateCheck:
        """Check load profile type against allowed list."""
        classifier = LoadProfileClassifier(data)
        report = classifier.classify(window_size=self._config.load_profile_window)
        profile_type = report.profile.profile_type
        allowed = self._config.allowed_load_profiles

        if profile_type.value in allowed:
            return GateCheck(
                name="load_profile",
                verdict=GateVerdict.PASS,
                detail=f"Load profile: {profile_type.value}",
                threshold=f"allowed: {', '.join(allowed)}",
            )
        return GateCheck(
            name="load_profile",
            verdict=GateVerdict.WARN,
            detail=f"Load profile: {profile_type.value} (not in allowed: {', '.join(allowed)})",
            threshold=f"allowed: {', '.join(allowed)}",
        )


def evaluate_quality_gate(
    benchmark_path: str,
    *,
    config_path: str | None = None,
    min_requests: int = 100,
    min_quality_score: float = 0.7,
    max_outlier_pct: float = 10.0,
    require_stable_convergence: bool = True,
) -> dict[str, Any]:
    """Programmatic API for benchmark quality gate.

    Args:
        benchmark_path: Path to benchmark JSON file.
        config_path: Optional YAML config file path. If provided, overrides
            other keyword arguments.
        min_requests: Minimum request count.
        min_quality_score: Minimum data quality score (0-1).
        max_outlier_pct: Maximum allowed outlier percentage.
        require_stable_convergence: Whether to require stable convergence.

    Returns:
        Dict with gate result fields.
    """
    if config_path:
        config = load_gate_config(config_path)
    else:
        config = GateConfig(
            min_requests=min_requests,
            min_quality_score=min_quality_score,
            max_outlier_pct=max_outlier_pct,
            require_stable_convergence=require_stable_convergence,
        )

    data = load_benchmark_auto(benchmark_path)
    gate = QualityGate(config)
    result = gate.evaluate(data)
    return result.model_dump()

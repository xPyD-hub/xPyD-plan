"""A/B Test Analysis for benchmark comparisons.

Compare two benchmark datasets (control vs treatment) with statistical
rigor: hypothesis testing, effect sizes, and power analysis.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class EffectMagnitude(str, Enum):
    """Cohen's d magnitude classification."""

    NEGLIGIBLE = "negligible"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class StatisticalTest(BaseModel):
    """Result of a single statistical test."""

    test_name: str = Field(..., description="Name of the test (welch_t, mann_whitney_u)")
    statistic: float = Field(..., description="Test statistic value")
    p_value: float = Field(..., description="Two-sided p-value")
    significant: bool = Field(..., description="True if p_value < alpha")


class EffectSize(BaseModel):
    """Effect size measurement."""

    cohens_d: float = Field(..., description="Cohen's d value")
    magnitude: EffectMagnitude = Field(..., description="Effect magnitude classification")


class ABConfidenceInterval(BaseModel):
    """Confidence interval for mean difference."""

    lower: float = Field(..., description="Lower bound")
    upper: float = Field(..., description="Upper bound")
    confidence_level: float = Field(0.95, description="Confidence level (e.g. 0.95)")


class PowerWarning(BaseModel):
    """Warning about statistical power."""

    adequate: bool = Field(..., description="True if power is adequate")
    reason: str = Field(..., description="Explanation")
    control_n: int = Field(..., description="Control sample size")
    treatment_n: int = Field(..., description="Treatment sample size")


class ABTestResult(BaseModel):
    """Result of an A/B test for a single metric."""

    metric: str = Field(..., description="Metric name (e.g. ttft_ms)")
    control_mean: float = Field(..., description="Control group mean")
    treatment_mean: float = Field(..., description="Treatment group mean")
    mean_difference: float = Field(..., description="treatment_mean - control_mean")
    relative_difference: float = Field(
        ..., description="Relative difference: diff / control_mean, or 0.0 if control_mean is 0"
    )
    welch_t: StatisticalTest = Field(..., description="Welch's t-test result")
    mann_whitney_u: StatisticalTest = Field(..., description="Mann-Whitney U test result")
    effect_size: EffectSize = Field(..., description="Cohen's d effect size")
    ci: ABConfidenceInterval = Field(..., description="CI for mean difference")
    power_warning: PowerWarning = Field(..., description="Statistical power warning")
    winner: Optional[str] = Field(
        None, description="'control' or 'treatment' if significant, else None"
    )


class ABTestConfig(BaseModel):
    """Configuration for A/B test analysis."""

    alpha: float = Field(0.05, ge=0.001, le=0.5, description="Significance level")
    metrics: list[str] = Field(
        default_factory=lambda: ["ttft_ms", "tpot_ms", "total_latency_ms"],
        description="Metrics to compare",
    )
    confidence_level: float = Field(0.95, ge=0.5, le=0.999, description="CI confidence level")
    min_sample_size: int = Field(30, ge=2, description="Minimum sample size warning threshold")


class ABTestReport(BaseModel):
    """Full A/B test report across all metrics."""

    control_file: str = Field(..., description="Control benchmark file path")
    treatment_file: str = Field(..., description="Treatment benchmark file path")
    alpha: float = Field(..., description="Significance level used")
    results: list[ABTestResult] = Field(..., description="Per-metric results")
    summary: str = Field(..., description="Human-readable summary")


VALID_METRICS = {"ttft_ms", "tpot_ms", "total_latency_ms"}


def _classify_cohens_d(d: float) -> EffectMagnitude:
    """Classify Cohen's d magnitude."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return EffectMagnitude.NEGLIGIBLE
    if abs_d < 0.5:
        return EffectMagnitude.SMALL
    if abs_d < 0.8:
        return EffectMagnitude.MEDIUM
    return EffectMagnitude.LARGE


def _welch_t_test(
    a: np.ndarray, b: np.ndarray, alpha: float
) -> StatisticalTest:
    """Welch's t-test (unequal variance)."""
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

    se = math.sqrt(var_a / n_a + var_b / n_b) if (var_a / n_a + var_b / n_b) > 0 else 1e-12
    t_stat = float((mean_b - mean_a) / se)

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom_a = (var_a / n_a) ** 2 / (n_a - 1) if n_a > 1 else 1e-12
    denom_b = (var_b / n_b) ** 2 / (n_b - 1) if n_b > 1 else 1e-12
    df = num / (denom_a + denom_b) if (denom_a + denom_b) > 0 else 1.0

    p_value = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))
    p_value = min(max(p_value, 0.0), 1.0)

    return StatisticalTest(
        test_name="welch_t",
        statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
    )


def _mann_whitney_u(
    a: np.ndarray, b: np.ndarray, alpha: float
) -> StatisticalTest:
    """Mann-Whitney U test (non-parametric)."""
    n_a, n_b = len(a), len(b)
    # Combine and rank
    combined = np.concatenate([a, b])
    ranks = np.empty_like(combined, dtype=float)
    order = np.argsort(combined)
    # Handle ties with average ranks
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and combined[order[j]] == combined[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-indexed average
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    r_a = np.sum(ranks[:n_a])
    u_a = r_a - n_a * (n_a + 1) / 2.0
    u_b = n_a * n_b - u_a
    u_stat = min(u_a, u_b)

    # Normal approximation for large samples
    mean_u = n_a * n_b / 2.0
    std_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12.0)
    if std_u == 0:
        z = 0.0
    else:
        z = (u_stat - mean_u) / std_u

    # Two-sided p-value from normal approximation
    p_value = 2.0 * _norm_cdf(-abs(z))
    p_value = min(max(p_value, 0.0), 1.0)

    return StatisticalTest(
        test_name="mann_whitney_u",
        statistic=float(u_stat),
        p_value=p_value,
        significant=p_value < alpha,
    )


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_cdf(x: float, df: float) -> float:
    """Approximate t-distribution CDF.

    Uses normal approximation for df >= 30, otherwise uses a more
    accurate approximation.
    """
    if df >= 30:
        return _norm_cdf(x)
    # Approximation using the formula from Abramowitz and Stegun
    # For moderate df, adjust x to normal equivalent
    g1 = 1.0 / (4.0 * df)
    z = x * (1.0 - g1) / math.sqrt(1.0 + x * x / (2.0 * df))
    return _norm_cdf(z)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d (pooled standard deviation)."""
    n_a, n_b = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled_std)


def _mean_diff_ci(
    a: np.ndarray, b: np.ndarray, confidence_level: float
) -> ABConfidenceInterval:
    """Confidence interval for difference of means."""
    n_a, n_b = len(a), len(b)
    mean_diff = float(np.mean(b) - np.mean(a))
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    se = math.sqrt(var_a / n_a + var_b / n_b) if (var_a / n_a + var_b / n_b) > 0 else 0.0

    # Use normal quantile for CI
    z = _norm_quantile((1.0 + confidence_level) / 2.0)
    margin = z * se

    return ABConfidenceInterval(
        lower=mean_diff - margin,
        upper=mean_diff + margin,
        confidence_level=confidence_level,
    )


def _norm_quantile(p: float) -> float:
    """Approximate inverse normal CDF (probit function).

    Uses rational approximation from Peter Acklam.
    """
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p == 0.5:
        return 0.0

    # Coefficients
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    return x


def _check_power(
    n_control: int, n_treatment: int, min_sample: int
) -> PowerWarning:
    """Check if sample sizes are adequate for statistical power."""
    adequate = n_control >= min_sample and n_treatment >= min_sample
    if adequate:
        reason = f"Sample sizes adequate (control={n_control}, treatment={n_treatment})"
    else:
        parts = []
        if n_control < min_sample:
            parts.append(f"control ({n_control} < {min_sample})")
        if n_treatment < min_sample:
            parts.append(f"treatment ({n_treatment} < {min_sample})")
        reason = (
            f"Insufficient sample size: {', '.join(parts)}. "
            "Results may lack statistical power."
        )

    return PowerWarning(
        adequate=adequate,
        reason=reason,
        control_n=n_control,
        treatment_n=n_treatment,
    )


class ABTestAnalyzer:
    """Analyze A/B test results between two benchmark datasets."""

    def __init__(self, config: ABTestConfig | None = None) -> None:
        self.config = config or ABTestConfig()

    def _extract_metric(self, data: BenchmarkData, metric: str) -> np.ndarray:
        """Extract metric values from benchmark data."""
        if metric not in VALID_METRICS:
            msg = f"Invalid metric '{metric}'. Valid: {sorted(VALID_METRICS)}"
            raise ValueError(msg)
        values = [getattr(r, metric) for r in data.requests]
        return np.array(values, dtype=float)

    def _analyze_metric(
        self, control: np.ndarray, treatment: np.ndarray, metric: str
    ) -> ABTestResult:
        """Run full analysis for a single metric."""
        alpha = self.config.alpha
        control_mean = float(np.mean(control))
        treatment_mean = float(np.mean(treatment))
        mean_diff = treatment_mean - control_mean
        rel_diff = mean_diff / control_mean if control_mean != 0 else 0.0

        welch = _welch_t_test(control, treatment, alpha)
        mwu = _mann_whitney_u(control, treatment, alpha)
        d = _cohens_d(control, treatment)
        effect = EffectSize(cohens_d=d, magnitude=_classify_cohens_d(d))
        ci = _mean_diff_ci(control, treatment, self.config.confidence_level)
        power = _check_power(len(control), len(treatment), self.config.min_sample_size)

        # Determine winner: lower latency is better
        winner = None
        if welch.significant:
            winner = "treatment" if mean_diff < 0 else "control"

        return ABTestResult(
            metric=metric,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            mean_difference=mean_diff,
            relative_difference=rel_diff,
            welch_t=welch,
            mann_whitney_u=mwu,
            effect_size=effect,
            ci=ci,
            power_warning=power,
            winner=winner,
        )

    def analyze(
        self, control_path: str, treatment_path: str
    ) -> ABTestReport:
        """Run A/B test analysis on two benchmark files.

        Args:
            control_path: Path to control (baseline) benchmark JSON.
            treatment_path: Path to treatment benchmark JSON.

        Returns:
            ABTestReport with per-metric results and summary.
        """
        control_data = BenchmarkData.model_validate(
            json.loads(Path(control_path).read_text())
        )
        treatment_data = BenchmarkData.model_validate(
            json.loads(Path(treatment_path).read_text())
        )

        results = []
        for metric in self.config.metrics:
            control_vals = self._extract_metric(control_data, metric)
            treatment_vals = self._extract_metric(treatment_data, metric)
            results.append(self._analyze_metric(control_vals, treatment_vals, metric))

        # Build summary
        sig_results = [r for r in results if r.welch_t.significant]
        if not sig_results:
            summary = (
                "No statistically significant differences found "
                "between control and treatment."
            )
        else:
            parts = []
            for r in sig_results:
                direction = "lower" if r.mean_difference < 0 else "higher"
                parts.append(
                    f"{r.metric}: treatment is "
                    f"{abs(r.relative_difference):.1%} {direction} "
                    f"(p={r.welch_t.p_value:.4f}, "
                    f"d={r.effect_size.cohens_d:.2f} "
                    f"[{r.effect_size.magnitude.value}])"
                )
            summary = "Significant differences: " + "; ".join(parts)

        return ABTestReport(
            control_file=control_path,
            treatment_file=treatment_path,
            alpha=self.config.alpha,
            results=results,
            summary=summary,
        )


def analyze_ab_test(
    control_path: str,
    treatment_path: str,
    alpha: float = 0.05,
    metrics: list[str] | None = None,
) -> ABTestReport:
    """Programmatic API: run A/B test analysis.

    Args:
        control_path: Path to control benchmark JSON.
        treatment_path: Path to treatment benchmark JSON.
        alpha: Significance level (default 0.05).
        metrics: Metrics to compare (default: ttft_ms, tpot_ms, total_latency_ms).

    Returns:
        ABTestReport with per-metric results and summary.
    """
    config = ABTestConfig(alpha=alpha)
    if metrics:
        config.metrics = metrics
    analyzer = ABTestAnalyzer(config=config)
    return analyzer.analyze(control_path, treatment_path)

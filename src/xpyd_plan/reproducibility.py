"""Benchmark reproducibility analysis — quantify result consistency across repeated runs."""

from __future__ import annotations

import enum
from itertools import combinations

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import ks_2samp

from xpyd_plan.benchmark_models import BenchmarkData


class ReproducibilityGrade(str, enum.Enum):
    """Reproducibility grade classification."""

    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"


class MetricReproducibility(BaseModel):
    """Reproducibility statistics for a single metric across runs."""

    metric_name: str = Field(..., description="Metric name (e.g., 'ttft_p95')")
    values: list[float] = Field(..., description="Per-run values")
    mean: float = Field(..., description="Mean across runs")
    std: float = Field(..., description="Standard deviation across runs")
    cv: float = Field(..., description="Coefficient of variation (std/mean)")
    reliable: bool = Field(
        ..., description="True if CV is below reliability threshold"
    )


class RunPairTest(BaseModel):
    """KS test result between two runs."""

    run_a: int = Field(..., description="Index of first run (0-based)")
    run_b: int = Field(..., description="Index of second run (0-based)")
    metric: str = Field(..., description="Latency metric tested")
    ks_statistic: float = Field(..., description="KS test statistic")
    p_value: float = Field(..., description="KS test p-value")
    distributions_consistent: bool = Field(
        ..., description="True if p-value > 0.05 (distributions not significantly different)"
    )


class ReproducibilityReport(BaseModel):
    """Complete reproducibility analysis report."""

    num_runs: int = Field(..., description="Number of benchmark runs analyzed")
    total_requests: int = Field(..., description="Total requests across all runs")
    metrics: list[MetricReproducibility] = Field(
        default_factory=list, description="Per-metric reproducibility stats"
    )
    pair_tests: list[RunPairTest] = Field(
        default_factory=list, description="KS test results between run pairs"
    )
    composite_score: float = Field(
        ..., ge=0, le=100, description="Composite reproducibility score (0-100)"
    )
    grade: ReproducibilityGrade = Field(..., description="Reproducibility grade")
    unreliable_metrics: list[str] = Field(
        default_factory=list, description="Metrics with CV above threshold"
    )
    recommended_min_runs: int = Field(
        ..., description="Recommended minimum runs for reliable results"
    )


def _grade_from_score(score: float) -> ReproducibilityGrade:
    """Classify score into grade."""
    if score >= 90:
        return ReproducibilityGrade.EXCELLENT
    if score >= 75:
        return ReproducibilityGrade.GOOD
    if score >= 60:
        return ReproducibilityGrade.FAIR
    return ReproducibilityGrade.POOR


class ReproducibilityAnalyzer:
    """Analyze reproducibility of benchmark results across multiple runs."""

    def __init__(self, cv_threshold: float = 0.10) -> None:
        """Initialize analyzer.

        Args:
            cv_threshold: CV threshold above which a metric is flagged as unreliable.
                Default 0.10 (10%).
        """
        if cv_threshold <= 0:
            raise ValueError("cv_threshold must be > 0")
        self.cv_threshold = cv_threshold

    def analyze(self, datasets: list[BenchmarkData]) -> ReproducibilityReport:
        """Analyze reproducibility across multiple benchmark runs.

        Args:
            datasets: List of BenchmarkData from repeated runs of the same config.

        Returns:
            ReproducibilityReport with scores and KS test results.

        Raises:
            ValueError: If fewer than 2 datasets provided.
        """
        if len(datasets) < 2:
            raise ValueError("At least 2 benchmark runs required for reproducibility analysis")

        total_requests = sum(len(d.requests) for d in datasets)

        # Extract per-run metrics
        metric_defs = [
            ("ttft_p50", lambda reqs: float(np.percentile([r.ttft_ms for r in reqs], 50))),
            ("ttft_p95", lambda reqs: float(np.percentile([r.ttft_ms for r in reqs], 95))),
            ("ttft_p99", lambda reqs: float(np.percentile([r.ttft_ms for r in reqs], 99))),
            ("tpot_p50", lambda reqs: float(np.percentile([r.tpot_ms for r in reqs], 50))),
            ("tpot_p95", lambda reqs: float(np.percentile([r.tpot_ms for r in reqs], 95))),
            ("tpot_p99", lambda reqs: float(np.percentile([r.tpot_ms for r in reqs], 99))),
            ("total_latency_p50", lambda reqs: float(
                np.percentile([r.total_latency_ms for r in reqs], 50))),
            ("total_latency_p95", lambda reqs: float(
                np.percentile([r.total_latency_ms for r in reqs], 95))),
            ("total_latency_p99", lambda reqs: float(
                np.percentile([r.total_latency_ms for r in reqs], 99))),
            ("qps", lambda reqs: None),  # handled separately
        ]

        metrics: list[MetricReproducibility] = []
        unreliable: list[str] = []

        for name, extractor in metric_defs:
            if name == "qps":
                values = [d.metadata.measured_qps for d in datasets]
            else:
                values = [extractor(d.requests) for d in datasets]

            arr = np.array(values, dtype=float)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            cv = std_val / mean_val if mean_val > 0 else 0.0
            reliable = cv <= self.cv_threshold

            if not reliable:
                unreliable.append(name)

            metrics.append(
                MetricReproducibility(
                    metric_name=name,
                    values=[round(v, 4) for v in values],
                    mean=round(mean_val, 4),
                    std=round(std_val, 4),
                    cv=round(cv, 6),
                    reliable=reliable,
                )
            )

        # KS tests between run pairs for raw latency distributions
        pair_tests: list[RunPairTest] = []
        ks_metrics = ["ttft_ms", "tpot_ms", "total_latency_ms"]
        for (i, da), (j, db) in combinations(enumerate(datasets), 2):
            for metric_field in ks_metrics:
                a_vals = np.array([getattr(r, metric_field) for r in da.requests])
                b_vals = np.array([getattr(r, metric_field) for r in db.requests])
                stat, pval = ks_2samp(a_vals, b_vals)
                pair_tests.append(
                    RunPairTest(
                        run_a=i,
                        run_b=j,
                        metric=metric_field,
                        ks_statistic=round(float(stat), 6),
                        p_value=round(float(pval), 6),
                        distributions_consistent=pval > 0.05,
                    )
                )

        # Composite score: 100 - mean(CV * 500), clamped to [0, 100]
        # CV of 0 → 100, CV of 0.20 → 0
        cvs = [m.cv for m in metrics]
        mean_cv = float(np.mean(cvs))
        composite = max(0.0, min(100.0, 100.0 - mean_cv * 500.0))
        composite = round(composite, 1)

        grade = _grade_from_score(composite)

        # Recommended min runs: based on observed variance
        # If CV is high, suggest more runs. Minimum 3, scale with mean CV.
        if mean_cv < 0.02:
            recommended = 3
        elif mean_cv < 0.05:
            recommended = 5
        elif mean_cv < 0.10:
            recommended = 7
        else:
            recommended = 10

        return ReproducibilityReport(
            num_runs=len(datasets),
            total_requests=total_requests,
            metrics=metrics,
            pair_tests=pair_tests,
            composite_score=composite,
            grade=grade,
            unreliable_metrics=unreliable,
            recommended_min_runs=recommended,
        )


def analyze_reproducibility(
    datasets: list[BenchmarkData],
    cv_threshold: float = 0.10,
) -> ReproducibilityReport:
    """Programmatic API: analyze benchmark reproducibility.

    Args:
        datasets: List of BenchmarkData from repeated runs.
        cv_threshold: CV threshold for reliability flagging (default 0.10).

    Returns:
        ReproducibilityReport with scores, KS tests, and recommendations.
    """
    analyzer = ReproducibilityAnalyzer(cv_threshold=cv_threshold)
    return analyzer.analyze(datasets)

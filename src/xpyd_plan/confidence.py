"""Bootstrap confidence interval analysis for latency percentiles."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class Adequacy(str, Enum):
    """Sample size adequacy classification."""

    SUFFICIENT = "sufficient"
    MARGINAL = "marginal"
    INSUFFICIENT = "insufficient"


class ConfidenceInterval(BaseModel):
    """A single confidence interval for a metric at a given percentile."""

    metric: str = Field(description="Metric name (ttft_ms, tpot_ms, total_latency_ms)")
    percentile: float = Field(description="Percentile evaluated (e.g. 95.0)")
    point_estimate: float = Field(description="Point estimate of the percentile value")
    ci_lower: float = Field(description="Lower bound of confidence interval")
    ci_upper: float = Field(description="Upper bound of confidence interval")
    ci_width: float = Field(description="CI width (upper - lower)")
    relative_ci_width: float = Field(
        description="CI width relative to point estimate (ci_width / point_estimate)"
    )
    confidence_level: float = Field(description="Confidence level (e.g. 0.95)")
    iterations: int = Field(description="Number of bootstrap iterations")
    sample_size: int = Field(description="Number of data points")


class MetricConfidence(BaseModel):
    """Confidence intervals for all percentiles of a single metric."""

    metric: str
    intervals: list[ConfidenceInterval]
    adequacy: Adequacy = Field(description="Sample size adequacy for this metric")


class ConfidenceReport(BaseModel):
    """Full confidence analysis report."""

    metrics: list[MetricConfidence]
    confidence_level: float
    percentile: float
    iterations: int
    sample_size: int
    adequate: bool = Field(description="True if all metrics have sufficient sample size")
    warnings: list[str] = Field(default_factory=list)


class ConfidenceAnalyzer:
    """Compute bootstrap confidence intervals for benchmark latency percentiles."""

    METRICS = ["ttft_ms", "tpot_ms", "total_latency_ms"]
    RELATIVE_WIDTH_MARGINAL = 0.10  # >10% relative CI width = marginal
    RELATIVE_WIDTH_INSUFFICIENT = 0.25  # >25% relative CI width = insufficient

    def __init__(
        self,
        confidence_level: float = 0.95,
        iterations: int = 1000,
        seed: Optional[int] = None,
    ):
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1 (exclusive)")
        if iterations < 10:
            raise ValueError("iterations must be at least 10")
        self.confidence_level = confidence_level
        self.iterations = iterations
        self.seed = seed

    def analyze(
        self,
        data: BenchmarkData,
        percentile: float = 95.0,
    ) -> ConfidenceReport:
        """Compute bootstrap CIs for all latency metrics at the given percentile."""
        if not 0 < percentile <= 100:
            raise ValueError("percentile must be between 0 (exclusive) and 100 (inclusive)")

        requests = data.requests
        sample_size = len(requests)
        warnings: list[str] = []

        if sample_size == 0:
            raise ValueError("No requests in benchmark data")

        if sample_size < 10:
            warnings.append(
                f"Very small sample size ({sample_size}). "
                "Confidence intervals may be unreliable."
            )

        rng = np.random.default_rng(self.seed)
        metric_results: list[MetricConfidence] = []

        for metric_name in self.METRICS:
            values = np.array([getattr(r, metric_name) for r in requests], dtype=np.float64)
            interval = self._bootstrap_ci(values, percentile, rng)
            adequacy = self._classify_adequacy(interval.relative_ci_width)

            ci = ConfidenceInterval(
                metric=metric_name,
                percentile=percentile,
                point_estimate=interval.point_estimate,
                ci_lower=interval.ci_lower,
                ci_upper=interval.ci_upper,
                ci_width=interval.ci_width,
                relative_ci_width=interval.relative_ci_width,
                confidence_level=self.confidence_level,
                iterations=self.iterations,
                sample_size=sample_size,
            )

            mc = MetricConfidence(
                metric=metric_name,
                intervals=[ci],
                adequacy=adequacy,
            )
            metric_results.append(mc)

            if adequacy == Adequacy.INSUFFICIENT:
                warnings.append(
                    f"{metric_name}: relative CI width is {interval.relative_ci_width:.1%} "
                    f"(>{self.RELATIVE_WIDTH_INSUFFICIENT:.0%}). "
                    "Consider collecting more benchmark data."
                )
            elif adequacy == Adequacy.MARGINAL:
                warnings.append(
                    f"{metric_name}: relative CI width is {interval.relative_ci_width:.1%} "
                    f"(>{self.RELATIVE_WIDTH_MARGINAL:.0%}). "
                    "Results may have limited precision."
                )

        all_adequate = all(m.adequacy == Adequacy.SUFFICIENT for m in metric_results)

        return ConfidenceReport(
            metrics=metric_results,
            confidence_level=self.confidence_level,
            percentile=percentile,
            iterations=self.iterations,
            sample_size=sample_size,
            adequate=all_adequate,
            warnings=warnings,
        )

    def _bootstrap_ci(
        self,
        values: np.ndarray,
        percentile: float,
        rng: np.random.Generator,
    ) -> ConfidenceInterval:
        """Compute bootstrap CI for a single array of values at a percentile."""
        n = len(values)
        point_estimate = float(np.percentile(values, percentile))

        if n == 1:
            # Single data point — CI collapses to the point
            return ConfidenceInterval(
                metric="",
                percentile=percentile,
                point_estimate=point_estimate,
                ci_lower=point_estimate,
                ci_upper=point_estimate,
                ci_width=0.0,
                relative_ci_width=0.0,
                confidence_level=self.confidence_level,
                iterations=self.iterations,
                sample_size=n,
            )

        # Bootstrap resampling
        bootstrap_percentiles = np.empty(self.iterations)
        for i in range(self.iterations):
            sample = rng.choice(values, size=n, replace=True)
            bootstrap_percentiles[i] = np.percentile(sample, percentile)

        alpha = 1 - self.confidence_level
        ci_lower = float(np.percentile(bootstrap_percentiles, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_percentiles, 100 * (1 - alpha / 2)))
        ci_width = ci_upper - ci_lower
        relative_ci_width = ci_width / point_estimate if point_estimate > 0 else 0.0

        return ConfidenceInterval(
            metric="",
            percentile=percentile,
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_width=ci_width,
            relative_ci_width=relative_ci_width,
            confidence_level=self.confidence_level,
            iterations=self.iterations,
            sample_size=n,
        )

    def _classify_adequacy(self, relative_ci_width: float) -> Adequacy:
        """Classify sample adequacy based on relative CI width."""
        if relative_ci_width > self.RELATIVE_WIDTH_INSUFFICIENT:
            return Adequacy.INSUFFICIENT
        elif relative_ci_width > self.RELATIVE_WIDTH_MARGINAL:
            return Adequacy.MARGINAL
        return Adequacy.SUFFICIENT


def analyze_confidence(
    data: BenchmarkData,
    percentile: float = 95.0,
    confidence_level: float = 0.95,
    iterations: int = 1000,
    seed: Optional[int] = None,
) -> ConfidenceReport:
    """Convenience function: compute bootstrap confidence intervals.

    Args:
        data: Benchmark data to analyze.
        percentile: Latency percentile to evaluate (default: 95.0).
        confidence_level: Confidence level for intervals (default: 0.95).
        iterations: Number of bootstrap iterations (default: 1000).
        seed: Random seed for reproducibility.

    Returns:
        ConfidenceReport with intervals for TTFT, TPOT, and total latency.
    """
    analyzer = ConfidenceAnalyzer(
        confidence_level=confidence_level,
        iterations=iterations,
        seed=seed,
    )
    return analyzer.analyze(data, percentile=percentile)

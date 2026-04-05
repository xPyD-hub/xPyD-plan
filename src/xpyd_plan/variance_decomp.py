"""Latency variance decomposition.

Attribute variance to prompt size, output size, temporal, and residual.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class ComponentSignificance(str, Enum):
    """Significance classification based on contribution percentage."""

    DOMINANT = "dominant"      # >= 40%
    SIGNIFICANT = "significant"  # 15% <= x < 40%
    MINOR = "minor"            # 5% <= x < 15%
    NEGLIGIBLE = "negligible"  # < 5%


class VarianceComponent(BaseModel):
    """A single variance component."""

    name: str = Field(..., description="Component name")
    sum_of_squares: float = Field(..., ge=0, description="Type III sum of squares")
    contribution_pct: float = Field(..., ge=0, le=100, description="Percentage of total variance")
    significance: ComponentSignificance = Field(..., description="Significance classification")


class MetricDecomposition(BaseModel):
    """Variance decomposition for a single latency metric."""

    metric: str = Field(..., description="Latency metric name")
    total_variance: float = Field(..., ge=0, description="Total variance (sum of squares)")
    components: list[VarianceComponent] = Field(..., description="Decomposed components")
    dominant_factor: str | None = Field(None, description="Name of the dominant component")
    r_squared: float = Field(..., ge=0, le=1, description="R² of the full model")


class VarianceReport(BaseModel):
    """Complete variance decomposition report."""

    metrics: list[MetricDecomposition] = Field(..., description="Per-metric decomposition")
    sample_size: int = Field(..., ge=1, description="Number of requests analyzed")
    recommendation: str = Field(..., description="Actionable recommendation")


def _classify_significance(pct: float) -> ComponentSignificance:
    """Classify component significance from contribution percentage."""
    if pct >= 40:
        return ComponentSignificance.DOMINANT
    if pct >= 15:
        return ComponentSignificance.SIGNIFICANT
    if pct >= 5:
        return ComponentSignificance.MINOR
    return ComponentSignificance.NEGLIGIBLE


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _ss_total(ys: list[float]) -> float:
    """Total sum of squares."""
    m = _mean(ys)
    return sum((y - m) ** 2 for y in ys)


def _sequential_ss(
    y: list[float],
    predictors: list[tuple[str, list[float]]],
    n_temporal_bins: int = 10,
) -> list[tuple[str, float]]:
    """Compute sequential (Type I) SS via manual OLS residuals.

    Each predictor is added sequentially; the SS attributed to it is the
    reduction in residual SS when it is added.

    For simplicity we use univariate OLS at each step on residuals.
    """
    n = len(y)
    residuals = list(y)  # start with raw values

    results: list[tuple[str, float]] = []
    for name, x in predictors:
        # Fit simple OLS: residuals ~ x
        mean_x = _mean(x)
        mean_r = _mean(residuals)

        cov = sum((xi - mean_x) * (ri - mean_r) for xi, ri in zip(x, residuals))
        var_x = sum((xi - mean_x) ** 2 for xi in x)

        if var_x == 0:
            results.append((name, 0.0))
            continue

        slope = cov / var_x
        intercept = mean_r - slope * mean_x

        # Compute SS explained = SS_before - SS_after
        ss_before = sum(r**2 for r in residuals) - n * mean_r**2
        if ss_before < 0:
            ss_before = sum((r - mean_r) ** 2 for r in residuals)

        new_residuals = [r - (slope * xi + intercept) for r, xi in zip(residuals, x)]
        mean_new = _mean(new_residuals)
        ss_after = sum((r - mean_new) ** 2 for r in new_residuals)

        ss_explained = max(0.0, ss_before - ss_after)
        results.append((name, ss_explained))

        residuals = new_residuals

    return results


def _build_temporal_feature(timestamps: list[float], n_bins: int = 10) -> list[float]:
    """Convert timestamps to temporal bin indices (as floats for regression)."""
    if not timestamps:
        return []
    t_min = min(timestamps)
    t_max = max(timestamps)
    t_range = t_max - t_min
    if t_range == 0:
        return [0.0] * len(timestamps)
    return [((t - t_min) / t_range) * n_bins for t in timestamps]


_LATENCY_METRICS = ["ttft_ms", "tpot_ms", "total_latency_ms"]


class VarianceDecomposer:
    """Decompose latency variance into prompt, output, temporal, and residual components."""

    def __init__(self, n_temporal_bins: int = 10):
        self._n_temporal_bins = n_temporal_bins

    def decompose(self, data: BenchmarkData) -> VarianceReport:
        """Decompose variance for all latency metrics.

        Args:
            data: Benchmark data with at least 3 requests.

        Returns:
            VarianceReport with per-metric decomposition.

        Raises:
            ValueError: If fewer than 3 requests.
        """
        if len(data.requests) < 3:
            raise ValueError("Need at least 3 requests for variance decomposition")

        n = len(data.requests)

        # Extract vectors
        prompt_tokens = [float(r.prompt_tokens) for r in data.requests]
        output_tokens = [float(r.output_tokens) for r in data.requests]
        timestamps = [r.timestamp for r in data.requests]
        temporal = _build_temporal_feature(timestamps, self._n_temporal_bins)

        latency_vectors = {
            "ttft_ms": [r.ttft_ms for r in data.requests],
            "tpot_ms": [r.tpot_ms for r in data.requests],
            "total_latency_ms": [r.total_latency_ms for r in data.requests],
        }

        predictors = [
            ("prompt_tokens", prompt_tokens),
            ("output_tokens", output_tokens),
            ("temporal", temporal),
        ]

        metrics: list[MetricDecomposition] = []
        dominant_factors: list[str | None] = []

        for metric_name in _LATENCY_METRICS:
            y = latency_vectors[metric_name]
            ss_tot = _ss_total(y)

            if ss_tot == 0:
                # Zero variance — all values identical
                components = [
                    VarianceComponent(
                        name=name,
                        sum_of_squares=0.0,
                        contribution_pct=0.0,
                        significance=ComponentSignificance.NEGLIGIBLE,
                    )
                    for name, _ in predictors
                ]
                components.append(
                    VarianceComponent(
                        name="residual",
                        sum_of_squares=0.0,
                        contribution_pct=0.0,
                        significance=ComponentSignificance.NEGLIGIBLE,
                    )
                )
                metrics.append(
                    MetricDecomposition(
                        metric=metric_name,
                        total_variance=0.0,
                        components=components,
                        dominant_factor=None,
                        r_squared=0.0,
                    )
                )
                dominant_factors.append(None)
                continue

            seq_ss = _sequential_ss(y, predictors, self._n_temporal_bins)

            explained_total = sum(ss for _, ss in seq_ss)
            residual_ss = max(0.0, ss_tot - explained_total)

            components: list[VarianceComponent] = []
            for name, ss in seq_ss:
                pct = (ss / ss_tot * 100) if ss_tot > 0 else 0.0
                components.append(
                    VarianceComponent(
                        name=name,
                        sum_of_squares=round(ss, 4),
                        contribution_pct=round(pct, 2),
                        significance=_classify_significance(pct),
                    )
                )

            residual_pct = (residual_ss / ss_tot * 100) if ss_tot > 0 else 0.0
            components.append(
                VarianceComponent(
                    name="residual",
                    sum_of_squares=round(residual_ss, 4),
                    contribution_pct=round(residual_pct, 2),
                    significance=_classify_significance(residual_pct),
                )
            )

            r_squared = explained_total / ss_tot if ss_tot > 0 else 0.0
            r_squared = min(1.0, max(0.0, r_squared))

            # Find dominant factor (excluding residual)
            non_residual = [c for c in components if c.name != "residual"]
            dominant = max(non_residual, key=lambda c: c.contribution_pct) if non_residual else None
            dominant_name = dominant.name if dominant and dominant.contribution_pct >= 15 else None

            metrics.append(
                MetricDecomposition(
                    metric=metric_name,
                    total_variance=round(ss_tot, 4),
                    components=components,
                    dominant_factor=dominant_name,
                    r_squared=round(r_squared, 4),
                )
            )
            dominant_factors.append(dominant_name)

        # Build recommendation
        recommendation = _build_recommendation(metrics)

        return VarianceReport(
            metrics=metrics,
            sample_size=n,
            recommendation=recommendation,
        )


def _build_recommendation(metrics: list[MetricDecomposition]) -> str:
    """Generate actionable recommendation from decomposition results."""
    # Collect dominant factors across metrics
    dominant_set: set[str] = set()
    for m in metrics:
        if m.dominant_factor:
            dominant_set.add(m.dominant_factor)

    if not dominant_set:
        return (
            "No single factor dominates latency variance. "
            "Variance is spread across multiple factors or driven by residual noise. "
            "Consider collecting more data or investigating system-level factors."
        )

    parts: list[str] = []
    if "prompt_tokens" in dominant_set:
        parts.append(
            "Prompt size is a dominant variance source — consider grouping "
            "similar-length prompts or optimizing prefill for long prompts."
        )
    if "output_tokens" in dominant_set:
        parts.append(
            "Output size is a dominant variance source — consider output length "
            "prediction or decode-instance scaling for long-generation workloads."
        )
    if "temporal" in dominant_set:
        parts.append(
            "Temporal patterns explain significant variance — investigate "
            "time-of-day load patterns, warmup effects, or resource contention."
        )

    return " ".join(parts)


def decompose_variance(data: BenchmarkData, *, n_temporal_bins: int = 10) -> dict:
    """Programmatic API for variance decomposition.

    Args:
        data: Benchmark data to analyze.
        n_temporal_bins: Number of bins for temporal feature.

    Returns:
        Dictionary representation of the VarianceReport.
    """
    decomposer = VarianceDecomposer(n_temporal_bins=n_temporal_bins)
    report = decomposer.decompose(data)
    return report.model_dump()

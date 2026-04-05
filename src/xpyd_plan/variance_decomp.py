"""Latency variance decomposition.

Attribute variance to prompt size, output size, temporal, and residual.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class Significance(str, Enum):
    """Statistical significance classification."""

    HIGH = "HIGH"        # contribution >= 30%
    MODERATE = "MODERATE"  # 10% <= contribution < 30%
    LOW = "LOW"          # 1% <= contribution < 10%
    NEGLIGIBLE = "NEGLIGIBLE"  # contribution < 1%


class VarianceComponent(BaseModel):
    """Single variance component contribution."""

    factor: str = Field(
        ...,
        description="Factor name (prompt_tokens, output_tokens, temporal, residual)",
    )
    sum_of_squares: float = Field(
        ..., ge=0, description="Sum of squares attributed to this factor",
    )
    contribution_pct: float = Field(
        ..., ge=0, le=100, description="Percentage of total variance explained",
    )
    significance: Significance = Field(
        ..., description="Significance classification",
    )


class MetricDecomposition(BaseModel):
    """Variance decomposition for a single latency metric."""

    metric: str = Field(
        ..., description="Latency metric name (ttft, tpot, total_latency)",
    )
    total_variance: float = Field(
        ..., ge=0, description="Total variance (sum of squares total)",
    )
    components: list[VarianceComponent] = Field(
        default_factory=list, description="Variance components",
    )
    dominant_factor: str = Field(
        ..., description="Factor with highest contribution (excl. residual)",
    )
    r_squared: float = Field(
        ..., ge=0, le=1,
        description="R-squared: fraction of variance explained by all factors",
    )


class VarianceReport(BaseModel):
    """Complete variance decomposition report."""

    total_requests: int = Field(
        ..., description="Number of requests analyzed",
    )
    decompositions: list[MetricDecomposition] = Field(
        default_factory=list, description="Per-metric decompositions",
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Actionable recommendations",
    )


def _classify_significance(pct: float) -> Significance:
    if pct >= 30:
        return Significance.HIGH
    if pct >= 10:
        return Significance.MODERATE
    if pct >= 1:
        return Significance.LOW
    return Significance.NEGLIGIBLE


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ss_total(values: list[float]) -> float:
    """Total sum of squares."""
    m = _mean(values)
    return sum((v - m) ** 2 for v in values)


def _ss_regression(x: list[float], y: list[float]) -> float:
    """Sum of squares explained by simple linear regression of y on x."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = _mean(x)
    my = _mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    if den == 0:
        return 0.0
    slope = num / den
    return slope * slope * den


def _residuals(x: list[float], y: list[float]) -> list[float]:
    """Return residuals from linear regression of y on x."""
    n = len(x)
    if n < 2:
        return list(y)
    mx = _mean(x)
    my = _mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    if den == 0:
        return [yi - my for yi in y]
    slope = num / den
    intercept = my - slope * mx
    return [yi - (slope * xi + intercept) for xi, yi in zip(x, y)]


def _bin_timestamps(timestamps: list[float], n_bins: int = 10) -> list[float]:
    """Convert timestamps to bin indices for temporal factor."""
    if not timestamps:
        return []
    mn = min(timestamps)
    mx = max(timestamps)
    rng = mx - mn
    if rng == 0:
        return [0.0] * len(timestamps)
    return [int((t - mn) / rng * (n_bins - 1)) for t in timestamps]


class VarianceDecomposer:
    """Decompose latency variance into components.

    Uses sequential (Type I) sum-of-squares: each factor is regressed
    in order, and its SS is the reduction in residual SS.
    """

    def __init__(self, temporal_bins: int = 10) -> None:
        self.temporal_bins = temporal_bins

    def analyze(self, data: BenchmarkData) -> VarianceReport:
        """Run variance decomposition on benchmark data."""
        if len(data.requests) < 3:
            msg = "At least 3 requests required for variance decomposition"
            raise ValueError(msg)

        prompt_tokens = [float(r.prompt_tokens) for r in data.requests]
        output_tokens = [float(r.output_tokens) for r in data.requests]
        timestamps = [r.timestamp for r in data.requests]
        temporal = [
            float(b) for b in _bin_timestamps(timestamps, self.temporal_bins)
        ]

        metrics = {
            "ttft": [r.ttft_ms for r in data.requests],
            "tpot": [r.tpot_ms for r in data.requests],
            "total_latency": [r.total_latency_ms for r in data.requests],
        }

        decompositions: list[MetricDecomposition] = []
        all_recommendations: list[str] = []

        for metric_name, values in metrics.items():
            decomp = self._decompose_metric(
                metric_name, values, prompt_tokens, output_tokens, temporal,
            )
            decompositions.append(decomp)
            all_recommendations.extend(
                self._generate_recommendations(decomp),
            )

        # Deduplicate recommendations
        seen: set[str] = set()
        unique_recs: list[str] = []
        for r in all_recommendations:
            if r not in seen:
                seen.add(r)
                unique_recs.append(r)

        return VarianceReport(
            total_requests=len(data.requests),
            decompositions=decompositions,
            recommendations=unique_recs,
        )

    def _decompose_metric(
        self,
        metric_name: str,
        values: list[float],
        prompt_tokens: list[float],
        output_tokens: list[float],
        temporal: list[float],
    ) -> MetricDecomposition:
        ss_tot = _ss_total(values)
        if ss_tot == 0:
            components = [
                VarianceComponent(
                    factor=f, sum_of_squares=0,
                    contribution_pct=0,
                    significance=Significance.NEGLIGIBLE,
                )
                for f in (
                    "prompt_tokens", "output_tokens",
                    "temporal", "residual",
                )
            ]
            return MetricDecomposition(
                metric=metric_name, total_variance=0,
                components=components,
                dominant_factor="residual", r_squared=0,
            )

        factors = [
            ("prompt_tokens", prompt_tokens),
            ("output_tokens", output_tokens),
            ("temporal", temporal),
        ]

        current_y = list(values)
        component_list: list[VarianceComponent] = []
        explained_ss = 0.0

        for factor_name, factor_x in factors:
            ss = _ss_regression(factor_x, current_y)
            pct = (ss / ss_tot) * 100 if ss_tot > 0 else 0.0
            component_list.append(VarianceComponent(
                factor=factor_name,
                sum_of_squares=round(ss, 4),
                contribution_pct=round(pct, 2),
                significance=_classify_significance(pct),
            ))
            explained_ss += ss
            current_y = _residuals(factor_x, current_y)

        residual_ss = max(0, ss_tot - explained_ss)
        residual_pct = (residual_ss / ss_tot) * 100
        component_list.append(VarianceComponent(
            factor="residual",
            sum_of_squares=round(residual_ss, 4),
            contribution_pct=round(residual_pct, 2),
            significance=_classify_significance(residual_pct),
        ))

        non_residual = [
            c for c in component_list if c.factor != "residual"
        ]
        if non_residual:
            dominant = max(
                non_residual, key=lambda c: c.contribution_pct,
            ).factor
        else:
            dominant = "residual"
        r_sq = explained_ss / ss_tot if ss_tot > 0 else 0.0

        return MetricDecomposition(
            metric=metric_name,
            total_variance=round(ss_tot, 4),
            components=component_list,
            dominant_factor=dominant,
            r_squared=round(min(r_sq, 1.0), 4),
        )

    def _generate_recommendations(
        self, decomp: MetricDecomposition,
    ) -> list[str]:
        recs: list[str] = []
        mn = decomp.metric
        dom = decomp.dominant_factor

        if dom == "prompt_tokens":
            recs.append(
                f"{mn}: variance dominated by prompt size"
                " — consider bucketing requests by prompt length",
            )
        elif dom == "output_tokens":
            recs.append(
                f"{mn}: variance dominated by output size"
                " — consider output length prediction or capping",
            )
        elif dom == "temporal":
            recs.append(
                f"{mn}: variance dominated by temporal effects"
                " — investigate load patterns or instance scaling",
            )

        residual = next(
            (c for c in decomp.components if c.factor == "residual"), None,
        )
        if residual and residual.contribution_pct > 50:
            recs.append(
                f"{mn}: >50% residual variance"
                " — latency largely driven by factors not captured here",
            )

        return recs


def decompose_variance(
    data: BenchmarkData, temporal_bins: int = 10,
) -> dict:
    """Programmatic API — returns dict of variance decomposition."""
    decomposer = VarianceDecomposer(temporal_bins=temporal_bins)
    report = decomposer.analyze(data)
    return report.model_dump()

"""Correlation analysis between request characteristics and latency metrics."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class CorrelationStrength(str, Enum):
    """Classification of correlation strength based on absolute value."""

    STRONG = "strong"       # |r| >= 0.7
    MODERATE = "moderate"   # 0.4 <= |r| < 0.7
    WEAK = "weak"           # 0.2 <= |r| < 0.4
    NEGLIGIBLE = "negligible"  # |r| < 0.2


class CorrelationPair(BaseModel):
    """Correlation result between two metrics."""

    x_metric: str = Field(..., description="Independent variable name")
    y_metric: str = Field(..., description="Dependent variable name")
    pearson_r: float = Field(..., ge=-1, le=1, description="Pearson correlation coefficient")
    strength: CorrelationStrength = Field(..., description="Classified strength")
    sample_size: int = Field(..., ge=2, description="Number of data points")


class CorrelationReport(BaseModel):
    """Complete correlation analysis report."""

    pairs: list[CorrelationPair] = Field(..., description="All analyzed pairs")
    strongest_pair: CorrelationPair | None = Field(
        None, description="Pair with highest |r|"
    )
    recommendation: str = Field(..., description="Human-readable summary")


def _classify_strength(r: float) -> CorrelationStrength:
    """Classify correlation strength from Pearson r."""
    abs_r = abs(r)
    if abs_r >= 0.7:
        return CorrelationStrength.STRONG
    if abs_r >= 0.4:
        return CorrelationStrength.MODERATE
    if abs_r >= 0.2:
        return CorrelationStrength.WEAK
    return CorrelationStrength.NEGLIGIBLE


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient.

    Returns 0.0 if either variable has zero variance.
    """
    n = len(xs)
    if n < 2:
        raise ValueError("Need at least 2 data points for correlation")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0

    return cov / denom


# Metrics that can be used as x (independent) or y (dependent)
_REQUEST_METRICS = ["prompt_tokens", "output_tokens"]
_LATENCY_METRICS = ["ttft_ms", "tpot_ms", "total_latency_ms"]


class CorrelationAnalyzer:
    """Analyze correlations between request characteristics and latency metrics."""

    def analyze(self, data: BenchmarkData) -> CorrelationReport:
        """Compute pairwise Pearson correlations between request and latency metrics.

        Args:
            data: Benchmark data with at least 2 requests.

        Returns:
            CorrelationReport with all pairwise results.

        Raises:
            ValueError: If fewer than 2 requests in the benchmark.
        """
        if len(data.requests) < 2:
            raise ValueError("Need at least 2 requests for correlation analysis")

        # Extract metric vectors
        vectors: dict[str, list[float]] = {
            "prompt_tokens": [float(r.prompt_tokens) for r in data.requests],
            "output_tokens": [float(r.output_tokens) for r in data.requests],
            "ttft_ms": [r.ttft_ms for r in data.requests],
            "tpot_ms": [r.tpot_ms for r in data.requests],
            "total_latency_ms": [r.total_latency_ms for r in data.requests],
        }

        pairs: list[CorrelationPair] = []
        n = len(data.requests)

        for x_name in _REQUEST_METRICS:
            for y_name in _LATENCY_METRICS:
                r = _pearson(vectors[x_name], vectors[y_name])
                pairs.append(
                    CorrelationPair(
                        x_metric=x_name,
                        y_metric=y_name,
                        pearson_r=round(r, 6),
                        strength=_classify_strength(r),
                        sample_size=n,
                    )
                )

        strongest = max(pairs, key=lambda p: abs(p.pearson_r)) if pairs else None

        # Build recommendation
        if strongest and strongest.strength == CorrelationStrength.STRONG:
            recommendation = (
                f"Strong correlation ({strongest.pearson_r:+.3f}) between "
                f"{strongest.x_metric} and {strongest.y_metric}. "
                f"This is likely a primary latency driver."
            )
        elif strongest and strongest.strength == CorrelationStrength.MODERATE:
            recommendation = (
                f"Moderate correlation ({strongest.pearson_r:+.3f}) between "
                f"{strongest.x_metric} and {strongest.y_metric}. "
                f"Consider investigating this relationship further."
            )
        else:
            recommendation = (
                "No strong correlations found between request characteristics "
                "and latency metrics. Latency may be driven by system-level factors."
            )

        return CorrelationReport(
            pairs=pairs,
            strongest_pair=strongest,
            recommendation=recommendation,
        )


def analyze_correlation(data: BenchmarkData) -> dict:
    """Programmatic API for correlation analysis.

    Args:
        data: Benchmark data to analyze.

    Returns:
        Dictionary representation of the CorrelationReport.
    """
    analyzer = CorrelationAnalyzer()
    report = analyzer.analyze(data)
    return report.model_dump()

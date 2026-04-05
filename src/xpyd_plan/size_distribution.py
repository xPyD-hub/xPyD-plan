"""Request size distribution analysis — token size histograms and latency correlation."""

from __future__ import annotations

import enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class DistributionShape(str, enum.Enum):
    """Shape classification for a token distribution."""

    UNIFORM = "uniform"
    RIGHT_SKEWED = "right_skewed"
    LEFT_SKEWED = "left_skewed"
    BIMODAL = "bimodal"
    NORMAL = "normal"


class SizeBin(BaseModel):
    """A single histogram bin with latency statistics."""

    bin_start: float = Field(..., description="Bin lower bound (inclusive)")
    bin_end: float = Field(..., description="Bin upper bound (exclusive)")
    count: int = Field(..., ge=0, description="Number of requests in this bin")
    fraction: float = Field(..., ge=0, le=1, description="Fraction of total requests")
    ttft_p50_ms: Optional[float] = Field(None, description="TTFT P50 (ms)")
    ttft_p95_ms: Optional[float] = Field(None, description="TTFT P95 (ms)")
    tpot_p50_ms: Optional[float] = Field(None, description="TPOT P50 (ms)")
    tpot_p95_ms: Optional[float] = Field(None, description="TPOT P95 (ms)")
    total_latency_p50_ms: Optional[float] = Field(None, description="Total latency P50 (ms)")
    total_latency_p95_ms: Optional[float] = Field(None, description="Total latency P95 (ms)")


class Histogram(BaseModel):
    """Token size histogram for a single dimension (prompt or output)."""

    dimension: str = Field(..., description="Token dimension ('prompt' or 'output')")
    bins: list[SizeBin] = Field(default_factory=list, description="Histogram bins")
    min_tokens: float = Field(..., description="Minimum token count")
    max_tokens: float = Field(..., description="Maximum token count")
    mean_tokens: float = Field(..., description="Mean token count")
    std_tokens: float = Field(..., description="Standard deviation of token counts")
    shape: DistributionShape = Field(..., description="Distribution shape classification")


class SizeLatencyCorrelation(BaseModel):
    """Correlation between token size and latency metrics."""

    dimension: str = Field(..., description="Token dimension ('prompt' or 'output')")
    ttft_pearson_r: float = Field(..., description="Pearson r: tokens vs TTFT")
    tpot_pearson_r: float = Field(..., description="Pearson r: tokens vs TPOT")
    total_latency_pearson_r: float = Field(..., description="Pearson r: tokens vs total latency")


class SizeDistributionReport(BaseModel):
    """Complete size distribution analysis report."""

    total_requests: int = Field(..., description="Total requests analyzed")
    prompt_histogram: Histogram = Field(..., description="Prompt token histogram")
    output_histogram: Histogram = Field(..., description="Output token histogram")
    correlations: list[SizeLatencyCorrelation] = Field(
        default_factory=list, description="Size-latency correlations"
    )


def _classify_shape(values: np.ndarray) -> DistributionShape:
    """Classify the shape of a distribution."""
    if len(values) < 4:
        return DistributionShape.NORMAL

    from scipy import stats as _stats

    skewness = float(_stats.skew(values))
    kurtosis = float(_stats.kurtosis(values))

    # Check bimodality via Hartigan's dip approximation:
    # Use kurtosis < -1 as a rough bimodal indicator
    if kurtosis < -1.0:
        return DistributionShape.BIMODAL

    # Skewness thresholds
    if abs(skewness) < 0.5:
        # Check uniformity: low kurtosis (platykurtic)
        if kurtosis < -1.0:
            return DistributionShape.UNIFORM
        return DistributionShape.NORMAL
    elif skewness >= 0.5:
        return DistributionShape.RIGHT_SKEWED
    else:
        return DistributionShape.LEFT_SKEWED


def _compute_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient. Return 0.0 if degenerate."""
    if len(x) < 2:
        return 0.0
    std_x = float(np.std(x))
    std_y = float(np.std(y))
    if std_x == 0 or std_y == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


class SizeDistributionAnalyzer:
    """Analyze request token size distributions and their latency impact."""

    def __init__(self, bins: int = 10) -> None:
        if bins < 2:
            raise ValueError("bins must be >= 2")
        self.bins = bins

    def analyze(self, data: BenchmarkData) -> SizeDistributionReport:
        """Run size distribution analysis on benchmark data."""
        requests = data.requests
        if not requests:
            raise ValueError("No requests in benchmark data")

        prompt_tokens = np.array([r.prompt_tokens for r in requests], dtype=float)
        output_tokens = np.array([r.output_tokens for r in requests], dtype=float)
        ttft = np.array([r.ttft_ms for r in requests], dtype=float)
        tpot = np.array([r.tpot_ms for r in requests], dtype=float)
        total_lat = np.array([r.total_latency_ms for r in requests], dtype=float)

        prompt_hist = self._build_histogram("prompt", prompt_tokens, ttft, tpot, total_lat)
        output_hist = self._build_histogram("output", output_tokens, ttft, tpot, total_lat)

        correlations = [
            SizeLatencyCorrelation(
                dimension="prompt",
                ttft_pearson_r=_compute_pearson(prompt_tokens, ttft),
                tpot_pearson_r=_compute_pearson(prompt_tokens, tpot),
                total_latency_pearson_r=_compute_pearson(prompt_tokens, total_lat),
            ),
            SizeLatencyCorrelation(
                dimension="output",
                ttft_pearson_r=_compute_pearson(output_tokens, ttft),
                tpot_pearson_r=_compute_pearson(output_tokens, tpot),
                total_latency_pearson_r=_compute_pearson(output_tokens, total_lat),
            ),
        ]

        return SizeDistributionReport(
            total_requests=len(requests),
            prompt_histogram=prompt_hist,
            output_histogram=output_hist,
            correlations=correlations,
        )

    def _build_histogram(
        self,
        dimension: str,
        token_values: np.ndarray,
        ttft: np.ndarray,
        tpot: np.ndarray,
        total_lat: np.ndarray,
    ) -> Histogram:
        """Build histogram with per-bin latency stats."""
        total = len(token_values)
        min_val = float(np.min(token_values))
        max_val = float(np.max(token_values))
        mean_val = float(np.mean(token_values))
        std_val = float(np.std(token_values))
        shape = _classify_shape(token_values)

        # Create bin edges
        edges = np.linspace(min_val, max_val + 1e-9, self.bins + 1)
        bin_indices = np.digitize(token_values, edges) - 1
        # Clamp to valid range
        bin_indices = np.clip(bin_indices, 0, self.bins - 1)

        size_bins: list[SizeBin] = []
        for i in range(self.bins):
            mask = bin_indices == i
            count = int(np.sum(mask))
            fraction = count / total if total > 0 else 0.0

            if count > 0:
                bin_ttft = ttft[mask]
                bin_tpot = tpot[mask]
                bin_total = total_lat[mask]
                size_bins.append(
                    SizeBin(
                        bin_start=float(edges[i]),
                        bin_end=float(edges[i + 1]),
                        count=count,
                        fraction=fraction,
                        ttft_p50_ms=float(np.percentile(bin_ttft, 50)),
                        ttft_p95_ms=float(np.percentile(bin_ttft, 95)),
                        tpot_p50_ms=float(np.percentile(bin_tpot, 50)),
                        tpot_p95_ms=float(np.percentile(bin_tpot, 95)),
                        total_latency_p50_ms=float(np.percentile(bin_total, 50)),
                        total_latency_p95_ms=float(np.percentile(bin_total, 95)),
                    )
                )
            else:
                size_bins.append(
                    SizeBin(
                        bin_start=float(edges[i]),
                        bin_end=float(edges[i + 1]),
                        count=0,
                        fraction=0.0,
                    )
                )

        return Histogram(
            dimension=dimension,
            bins=size_bins,
            min_tokens=min_val,
            max_tokens=max_val,
            mean_tokens=mean_val,
            std_tokens=std_val,
            shape=shape,
        )


def analyze_size_distribution(
    data: BenchmarkData,
    bins: int = 10,
) -> SizeDistributionReport:
    """Programmatic API: analyze request size distribution.

    Args:
        data: Benchmark data to analyze.
        bins: Number of histogram bins (default 10).

    Returns:
        SizeDistributionReport with histograms and correlations.
    """
    analyzer = SizeDistributionAnalyzer(bins=bins)
    return analyzer.analyze(data)

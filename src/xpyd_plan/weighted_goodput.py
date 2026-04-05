"""Weighted goodput analysis — continuous SLA compliance scoring.

Instead of binary pass/fail, each request receives a score in [0, 1] based
on how closely it meets SLA thresholds.  Requests within SLA score 1.0;
requests beyond SLA decay smoothly toward 0.0 within a configurable grace
window.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData, BenchmarkRequest


class ScoringConfig(BaseModel):
    """Configuration for weighted goodput scoring."""

    sla_ttft_ms: float | None = Field(None, description="TTFT SLA threshold (ms)")
    sla_tpot_ms: float | None = Field(None, description="TPOT SLA threshold (ms)")
    sla_total_latency_ms: float | None = Field(
        None, description="Total latency SLA threshold (ms)"
    )
    grace_factor: float = Field(
        0.5,
        ge=0,
        description=(
            "Grace window as fraction of threshold.  "
            "Score decays linearly from 1→0 over threshold*(1+grace_factor)."
        ),
    )
    aggregation: str = Field(
        "min",
        description="How to combine per-metric scores: 'min' or 'mean'",
    )


class RequestScore(BaseModel):
    """Weighted score for a single request."""

    request_id: str = Field(..., description="Request identifier")
    ttft_score: float | None = Field(None, ge=0, le=1)
    tpot_score: float | None = Field(None, ge=0, le=1)
    total_latency_score: float | None = Field(None, ge=0, le=1)
    overall_score: float = Field(..., ge=0, le=1, description="Aggregated score")


class ScoreBucket(BaseModel):
    """One bucket in the score distribution histogram."""

    lower: float = Field(..., ge=0, le=1)
    upper: float = Field(..., ge=0, le=1)
    count: int = Field(..., ge=0)
    fraction: float = Field(..., ge=0, le=1)


class ScoreDistribution(BaseModel):
    """Distribution of request scores."""

    buckets: list[ScoreBucket] = Field(
        default_factory=list, description="Histogram buckets"
    )
    mean: float = Field(..., ge=0, le=1)
    median: float = Field(..., ge=0, le=1)
    p5: float = Field(..., ge=0, le=1, description="5th percentile score")
    std: float = Field(..., ge=0, description="Standard deviation")


class WeightedGoodputReport(BaseModel):
    """Complete weighted goodput analysis report."""

    total_requests: int = Field(..., ge=0)
    weighted_goodput: float = Field(
        ..., ge=0, le=1, description="Mean weighted score across all requests"
    )
    binary_goodput_ratio: float = Field(
        ..., ge=0, le=1, description="Fraction of requests with score == 1.0"
    )
    raw_qps: float = Field(..., ge=0)
    weighted_goodput_qps: float = Field(
        ..., ge=0, description="raw_qps * weighted_goodput"
    )
    score_distribution: ScoreDistribution = Field(
        ..., description="Histogram of request scores"
    )
    near_miss_count: int = Field(
        ..., ge=0, description="Requests with 0 < score < 1"
    )
    near_miss_fraction: float = Field(
        ..., ge=0, le=1, description="Fraction of near-miss requests"
    )
    scoring_config: ScoringConfig = Field(..., description="Config used")
    recommendation: str = Field(..., description="Human-readable recommendation")


def _score_metric(measured: float, threshold: float, grace_factor: float) -> float:
    """Compute score for a single metric.

    Returns 1.0 if measured <= threshold, decays linearly to 0.0 at
    threshold * (1 + grace_factor).  If grace_factor == 0, score is
    binary (1.0 or 0.0).
    """
    if measured <= threshold:
        return 1.0
    grace = threshold * grace_factor
    if grace <= 0:
        return 0.0
    return max(0.0, 1.0 - (measured - threshold) / grace)


def _score_request(
    req: BenchmarkRequest,
    config: ScoringConfig,
) -> RequestScore:
    """Score a single request against configured SLA thresholds."""
    scores: list[float] = []
    ttft_s: float | None = None
    tpot_s: float | None = None
    total_s: float | None = None

    if config.sla_ttft_ms is not None:
        ttft_s = _score_metric(req.ttft_ms, config.sla_ttft_ms, config.grace_factor)
        scores.append(ttft_s)
    if config.sla_tpot_ms is not None:
        tpot_s = _score_metric(req.tpot_ms, config.sla_tpot_ms, config.grace_factor)
        scores.append(tpot_s)
    if config.sla_total_latency_ms is not None:
        total_s = _score_metric(
            req.total_latency_ms, config.sla_total_latency_ms, config.grace_factor
        )
        scores.append(total_s)

    if not scores:
        overall = 1.0
    elif config.aggregation == "mean":
        overall = float(np.mean(scores))
    else:  # min
        overall = min(scores)

    return RequestScore(
        request_id=req.request_id,
        ttft_score=ttft_s,
        tpot_score=tpot_s,
        total_latency_score=total_s,
        overall_score=overall,
    )


def _build_distribution(scores: list[float]) -> ScoreDistribution:
    """Build a 5-bucket histogram from scores."""
    arr = np.array(scores)
    boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    total = len(scores)
    buckets: list[ScoreBucket] = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i < len(boundaries) - 2:
            count = int(np.sum((arr >= lo) & (arr < hi)))
        else:
            count = int(np.sum((arr >= lo) & (arr <= hi)))
        buckets.append(
            ScoreBucket(
                lower=lo,
                upper=hi,
                count=count,
                fraction=count / total if total > 0 else 0.0,
            )
        )
    return ScoreDistribution(
        buckets=buckets,
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        p5=float(np.percentile(arr, 5)),
        std=float(np.std(arr)),
    )


class WeightedGoodputAnalyzer:
    """Analyze weighted goodput from benchmark data."""

    def __init__(self, config: ScoringConfig) -> None:
        if (
            config.sla_ttft_ms is None
            and config.sla_tpot_ms is None
            and config.sla_total_latency_ms is None
        ):
            raise ValueError(
                "At least one SLA threshold must be specified "
                "(sla_ttft_ms, sla_tpot_ms, or sla_total_latency_ms)"
            )
        if config.aggregation not in ("min", "mean"):
            raise ValueError("aggregation must be 'min' or 'mean'")
        self._config = config

    def analyze(self, data: BenchmarkData) -> WeightedGoodputReport:
        """Run weighted goodput analysis."""
        total = len(data.requests)
        if total == 0:
            return WeightedGoodputReport(
                total_requests=0,
                weighted_goodput=1.0,
                binary_goodput_ratio=1.0,
                raw_qps=0.0,
                weighted_goodput_qps=0.0,
                score_distribution=ScoreDistribution(
                    buckets=[], mean=1.0, median=1.0, p5=1.0, std=0.0
                ),
                near_miss_count=0,
                near_miss_fraction=0.0,
                scoring_config=self._config,
                recommendation="No requests to analyze.",
            )

        request_scores = [_score_request(r, self._config) for r in data.requests]
        overall_scores = [rs.overall_score for rs in request_scores]

        perfect = sum(1 for s in overall_scores if s >= 1.0)
        binary_ratio = perfect / total
        near_miss = sum(1 for s in overall_scores if 0 < s < 1.0)
        weighted = float(np.mean(overall_scores))
        raw_qps = data.metadata.measured_qps

        distribution = _build_distribution(overall_scores)

        # Recommendation
        if weighted >= 0.99:
            rec = "Excellent weighted goodput. SLA margins are comfortable."
        elif weighted >= 0.95:
            rec = (
                "Good weighted goodput. Some requests are near SLA boundaries. "
                "Consider minor capacity adjustments."
            )
        elif weighted >= 0.80:
            rec = (
                f"Fair weighted goodput ({weighted:.1%}). "
                f"{near_miss} near-miss requests suggest SLA thresholds "
                "are tight for current capacity."
            )
        else:
            rec = (
                f"Poor weighted goodput ({weighted:.1%}). "
                "Significant SLA violations detected. "
                "Capacity increase or P:D ratio adjustment recommended."
            )

        return WeightedGoodputReport(
            total_requests=total,
            weighted_goodput=weighted,
            binary_goodput_ratio=binary_ratio,
            raw_qps=raw_qps,
            weighted_goodput_qps=raw_qps * weighted,
            score_distribution=distribution,
            near_miss_count=near_miss,
            near_miss_fraction=near_miss / total,
            scoring_config=self._config,
            recommendation=rec,
        )


def analyze_weighted_goodput(
    data: BenchmarkData,
    *,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_latency_ms: float | None = None,
    grace_factor: float = 0.5,
    aggregation: str = "min",
) -> dict:
    """Convenience function for programmatic API.

    Returns the report as a dictionary.
    """
    config = ScoringConfig(
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_latency_ms=sla_total_latency_ms,
        grace_factor=grace_factor,
        aggregation=aggregation,
    )
    analyzer = WeightedGoodputAnalyzer(config)
    report = analyzer.analyze(data)
    return report.model_dump()

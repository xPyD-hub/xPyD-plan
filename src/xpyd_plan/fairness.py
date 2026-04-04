"""Request fairness analysis across token-count buckets."""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class FairnessClassification(str, Enum):
    """Classification of fairness based on Jain's index."""

    FAIR = "fair"          # J >= 0.9
    MODERATE = "moderate"  # 0.7 <= J < 0.9
    UNFAIR = "unfair"      # J < 0.7


class BucketStats(BaseModel):
    """Latency statistics for a single token-count bucket."""

    bucket_index: int = Field(..., ge=0, description="Bucket index (0-based)")
    min_tokens: int = Field(..., ge=0, description="Lower bound of token range (inclusive)")
    max_tokens: int = Field(..., description="Upper bound of token range (inclusive)")
    request_count: int = Field(..., ge=0, description="Number of requests in this bucket")
    ttft_p50_ms: float = Field(..., ge=0)
    ttft_p95_ms: float = Field(..., ge=0)
    tpot_p50_ms: float = Field(..., ge=0)
    tpot_p95_ms: float = Field(..., ge=0)
    total_latency_p50_ms: float = Field(..., ge=0)
    total_latency_p95_ms: float = Field(..., ge=0)


class FairnessIndex(BaseModel):
    """Jain's fairness index for a specific latency metric."""

    metric: str = Field(..., description="Latency metric name")
    jain_index: float = Field(..., ge=0, le=1, description="Jain's fairness index")
    classification: FairnessClassification = Field(
        ..., description="Fairness classification"
    )


class FairnessReport(BaseModel):
    """Complete fairness analysis report."""

    buckets: list[BucketStats] = Field(..., description="Per-bucket latency stats")
    num_buckets: int = Field(..., ge=1, description="Number of buckets used")
    fairness_indices: list[FairnessIndex] = Field(
        ..., description="Jain's index per latency metric"
    )
    overall_classification: FairnessClassification = Field(
        ..., description="Worst-case classification across all metrics"
    )
    recommendation: str = Field(..., description="Human-readable summary")


def _classify_fairness(j: float) -> FairnessClassification:
    """Classify fairness from Jain's index value."""
    if j >= 0.9:
        return FairnessClassification.FAIR
    if j >= 0.7:
        return FairnessClassification.MODERATE
    return FairnessClassification.UNFAIR


def _jain_index(values: list[float]) -> float:
    """Compute Jain's fairness index: J = (sum(x))^2 / (n * sum(x^2)).

    Returns 1.0 for empty or single-element lists.
    Returns 1.0 if all values are zero.
    """
    n = len(values)
    if n <= 1:
        return 1.0
    s = sum(values)
    s2 = sum(x * x for x in values)
    if s2 == 0:
        return 1.0
    return (s * s) / (n * s2)


class FairnessAnalyzer:
    """Analyze latency fairness across request token-count buckets."""

    def analyze(
        self,
        data: BenchmarkData,
        num_buckets: int = 4,
    ) -> FairnessReport:
        """Bucket requests by prompt_tokens and compute fairness indices.

        Args:
            data: Benchmark data to analyze.
            num_buckets: Number of quantile-based buckets (default 4).

        Returns:
            FairnessReport with per-bucket stats and Jain's indices.

        Raises:
            ValueError: If fewer than num_buckets requests or num_buckets < 2.
        """
        if num_buckets < 2:
            raise ValueError("num_buckets must be at least 2")
        if len(data.requests) < num_buckets:
            raise ValueError(
                f"Need at least {num_buckets} requests, got {len(data.requests)}"
            )

        # Compute quantile boundaries for prompt_tokens
        prompt_tokens = np.array([r.prompt_tokens for r in data.requests])
        quantiles = np.quantile(
            prompt_tokens,
            np.linspace(0, 1, num_buckets + 1),
        )
        # Ensure unique boundaries
        boundaries = sorted(set(quantiles))
        if len(boundaries) < 2:
            # All same token count — one bucket
            boundaries = [float(prompt_tokens.min()), float(prompt_tokens.max())]

        # Assign requests to buckets
        actual_num_buckets = len(boundaries) - 1
        bucket_requests: list[list[int]] = [[] for _ in range(actual_num_buckets)]
        for i, r in enumerate(data.requests):
            for b in range(actual_num_buckets):
                lo = boundaries[b]
                hi = boundaries[b + 1]
                if b == actual_num_buckets - 1:
                    # Last bucket: inclusive on both ends
                    if lo <= r.prompt_tokens <= hi:
                        bucket_requests[b].append(i)
                        break
                else:
                    if lo <= r.prompt_tokens < hi:
                        bucket_requests[b].append(i)
                        break

        # Compute per-bucket stats
        buckets: list[BucketStats] = []
        for b in range(actual_num_buckets):
            indices = bucket_requests[b]
            if not indices:
                buckets.append(
                    BucketStats(
                        bucket_index=b,
                        min_tokens=int(boundaries[b]),
                        max_tokens=int(boundaries[b + 1]),
                        request_count=0,
                        ttft_p50_ms=0.0,
                        ttft_p95_ms=0.0,
                        tpot_p50_ms=0.0,
                        tpot_p95_ms=0.0,
                        total_latency_p50_ms=0.0,
                        total_latency_p95_ms=0.0,
                    )
                )
                continue

            reqs = [data.requests[i] for i in indices]
            ttft = np.array([r.ttft_ms for r in reqs])
            tpot = np.array([r.tpot_ms for r in reqs])
            total = np.array([r.total_latency_ms for r in reqs])

            buckets.append(
                BucketStats(
                    bucket_index=b,
                    min_tokens=int(boundaries[b]),
                    max_tokens=int(boundaries[b + 1]),
                    request_count=len(indices),
                    ttft_p50_ms=round(float(np.percentile(ttft, 50)), 3),
                    ttft_p95_ms=round(float(np.percentile(ttft, 95)), 3),
                    tpot_p50_ms=round(float(np.percentile(tpot, 50)), 3),
                    tpot_p95_ms=round(float(np.percentile(tpot, 95)), 3),
                    total_latency_p50_ms=round(float(np.percentile(total, 50)), 3),
                    total_latency_p95_ms=round(float(np.percentile(total, 95)), 3),
                )
            )

        # Compute Jain's fairness index on P95 latencies per bucket (non-empty only)
        non_empty = [b for b in buckets if b.request_count > 0]

        metrics = [
            ("ttft_p95", [b.ttft_p95_ms for b in non_empty]),
            ("tpot_p95", [b.tpot_p95_ms for b in non_empty]),
            ("total_latency_p95", [b.total_latency_p95_ms for b in non_empty]),
        ]

        fairness_indices: list[FairnessIndex] = []
        for name, values in metrics:
            j = round(_jain_index(values), 6)
            fairness_indices.append(
                FairnessIndex(
                    metric=name,
                    jain_index=j,
                    classification=_classify_fairness(j),
                )
            )

        # Overall = worst case
        worst_j = min(fi.jain_index for fi in fairness_indices)
        overall = _classify_fairness(worst_j)

        # Recommendation
        if overall == FairnessClassification.FAIR:
            recommendation = (
                "Latency is fairly distributed across request token-count buckets. "
                "No action needed."
            )
        elif overall == FairnessClassification.MODERATE:
            unfair_metrics = [
                fi.metric for fi in fairness_indices
                if fi.classification != FairnessClassification.FAIR
            ]
            recommendation = (
                f"Moderate fairness gap detected in: {', '.join(unfair_metrics)}. "
                f"Some request sizes may experience disproportionately higher latency."
            )
        else:
            unfair_metrics = [
                fi.metric for fi in fairness_indices
                if fi.classification == FairnessClassification.UNFAIR
            ]
            recommendation = (
                f"Significant unfairness in: {', '.join(unfair_metrics)}. "
                f"Consider rebalancing P:D ratio or investigating token-size-dependent "
                f"performance degradation."
            )

        return FairnessReport(
            buckets=buckets,
            num_buckets=actual_num_buckets,
            fairness_indices=fairness_indices,
            overall_classification=overall,
            recommendation=recommendation,
        )


def analyze_fairness(data: BenchmarkData, num_buckets: int = 4) -> dict:
    """Programmatic API for fairness analysis.

    Args:
        data: Benchmark data to analyze.
        num_buckets: Number of quantile-based buckets.

    Returns:
        Dictionary representation of the FairnessReport.
    """
    analyzer = FairnessAnalyzer()
    report = analyzer.analyze(data, num_buckets=num_buckets)
    return report.model_dump()

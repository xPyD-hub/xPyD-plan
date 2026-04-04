"""Batch size impact analysis — group requests by temporal proximity and analyze impact."""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class BatchEfficiency(str, Enum):
    """Efficiency classification for a batch size."""

    OPTIMAL = "OPTIMAL"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"


class BatchBucket(BaseModel):
    """Statistics for a group of batches with the same inferred batch size."""

    batch_size: int = Field(..., ge=1, description="Number of requests in batch")
    count: int = Field(..., ge=1, description="Number of batches with this size")
    mean_ttft_ms: float = Field(..., description="Mean TTFT across batches")
    p50_ttft_ms: float = Field(..., description="P50 TTFT across batches")
    p95_ttft_ms: float = Field(..., description="P95 TTFT across batches")
    mean_tpot_ms: float = Field(..., description="Mean TPOT across batches")
    p50_tpot_ms: float = Field(..., description="P50 TPOT across batches")
    p95_tpot_ms: float = Field(..., description="P95 TPOT across batches")
    mean_total_latency_ms: float = Field(..., description="Mean total latency across batches")
    p50_total_latency_ms: float = Field(..., description="P50 total latency across batches")
    p95_total_latency_ms: float = Field(..., description="P95 total latency across batches")
    throughput_rps: float = Field(
        ..., ge=0, description="Effective requests/sec for this batch size"
    )
    efficiency: BatchEfficiency = Field(..., description="Efficiency classification")


class BatchReport(BaseModel):
    """Complete batch size impact analysis report."""

    total_requests: int = Field(..., ge=0, description="Total requests analyzed")
    total_batches: int = Field(..., ge=0, description="Total inferred batches")
    window_ms: float = Field(..., gt=0, description="Temporal window used for grouping (ms)")
    min_batch_size: int = Field(..., ge=1, description="Smallest inferred batch size")
    max_batch_size: int = Field(..., ge=1, description="Largest inferred batch size")
    mean_batch_size: float = Field(..., ge=1, description="Mean inferred batch size")
    buckets: list[BatchBucket] = Field(
        default_factory=list, description="Per-batch-size statistics"
    )
    optimal_batch_size: int = Field(..., ge=1, description="Recommended batch size")
    recommendation: str = Field(..., description="Human-readable recommendation")


class BatchAnalyzer:
    """Analyze batch size impact on latency and throughput.

    Groups temporally close requests into inferred batches and computes
    per-batch-size latency and throughput statistics.
    """

    def __init__(self, window_ms: float = 100.0) -> None:
        """Initialize with temporal grouping window.

        Args:
            window_ms: Maximum time gap (ms) between requests in the same batch.
        """
        if window_ms <= 0:
            msg = "window_ms must be positive"
            raise ValueError(msg)
        self.window_ms = window_ms

    def analyze(self, data: BenchmarkData) -> BatchReport:
        """Analyze batch size impact from benchmark data.

        Args:
            data: Loaded benchmark data.

        Returns:
            BatchReport with per-batch-size statistics and recommendation.
        """
        requests = data.requests
        if not requests:
            return BatchReport(
                total_requests=0,
                total_batches=0,
                window_ms=self.window_ms,
                min_batch_size=1,
                max_batch_size=1,
                mean_batch_size=1.0,
                buckets=[],
                optimal_batch_size=1,
                recommendation="No requests to analyze.",
            )

        # Sort requests by timestamp
        sorted_reqs = sorted(requests, key=lambda r: r.timestamp)

        # Group into batches by temporal proximity
        batches: list[list] = []
        current_batch = [sorted_reqs[0]]

        for req in sorted_reqs[1:]:
            gap_ms = (req.timestamp - current_batch[-1].timestamp) * 1000.0
            if gap_ms <= self.window_ms:
                current_batch.append(req)
            else:
                batches.append(current_batch)
                current_batch = [req]
        batches.append(current_batch)

        # Group batches by size
        size_to_batches: dict[int, list[list]] = {}
        for batch in batches:
            size = len(batch)
            size_to_batches.setdefault(size, []).append(batch)

        # Compute per-bucket stats
        buckets: list[BatchBucket] = []
        best_score = -1.0
        optimal_size = 1

        for size in sorted(size_to_batches.keys()):
            batch_list = size_to_batches[size]

            # Collect per-batch mean latencies
            batch_ttfts = []
            batch_tpots = []
            batch_totals = []
            batch_throughputs = []

            for batch in batch_list:
                ttfts = [r.ttft_ms for r in batch]
                tpots = [r.tpot_ms for r in batch]
                totals = [r.total_latency_ms for r in batch]

                batch_ttfts.append(np.mean(ttfts))
                batch_tpots.append(np.mean(tpots))
                batch_totals.append(np.mean(totals))

                # Throughput: batch_size / max_total_latency_seconds
                max_lat_s = max(totals) / 1000.0
                if max_lat_s > 0:
                    batch_throughputs.append(len(batch) / max_lat_s)
                else:
                    batch_throughputs.append(0.0)

            ttft_arr = np.array(batch_ttfts)
            tpot_arr = np.array(batch_tpots)
            total_arr = np.array(batch_totals)
            tp_arr = np.array(batch_throughputs)

            mean_tp = float(np.mean(tp_arr)) if len(tp_arr) > 0 else 0.0
            p95_total = float(np.percentile(total_arr, 95)) if len(total_arr) > 0 else 0.0

            # Score: throughput / p95_latency (higher is better)
            score = mean_tp / p95_total if p95_total > 0 else 0.0

            # Classify efficiency
            if score > best_score:
                best_score = score
                optimal_size = size

            efficiency = self._classify_efficiency(score, best_score if best_score > 0 else score)

            buckets.append(
                BatchBucket(
                    batch_size=size,
                    count=len(batch_list),
                    mean_ttft_ms=float(np.mean(ttft_arr)),
                    p50_ttft_ms=float(np.percentile(ttft_arr, 50)),
                    p95_ttft_ms=float(np.percentile(ttft_arr, 95)),
                    mean_tpot_ms=float(np.mean(tpot_arr)),
                    p50_tpot_ms=float(np.percentile(tpot_arr, 50)),
                    p95_tpot_ms=float(np.percentile(tpot_arr, 95)),
                    mean_total_latency_ms=float(np.mean(total_arr)),
                    p50_total_latency_ms=float(np.percentile(total_arr, 50)),
                    p95_total_latency_ms=float(np.percentile(total_arr, 95)),
                    throughput_rps=mean_tp,
                    efficiency=efficiency,
                )
            )

        # Re-classify efficiency now that we know the best score
        for bucket in buckets:
            tp = bucket.throughput_rps
            p95 = bucket.p95_total_latency_ms
            score = tp / p95 if p95 > 0 else 0.0
            bucket.efficiency = self._classify_efficiency(score, best_score)

        # Find optimal (highest score)
        if buckets:
            scores = []
            for b in buckets:
                p95 = b.p95_total_latency_ms
                s = b.throughput_rps / p95 if p95 > 0 else 0.0
                scores.append((s, b.batch_size))
            scores.sort(reverse=True)
            optimal_size = scores[0][1]

        batch_sizes = [len(b) for b in batches]
        bs_arr = np.array(batch_sizes)

        recommendation = (
            f"Optimal inferred batch size is {optimal_size}. "
            f"Observed {len(batches)} batches with sizes ranging from "
            f"{int(bs_arr.min())} to {int(bs_arr.max())} "
            f"(mean {float(bs_arr.mean()):.1f})."
        )

        return BatchReport(
            total_requests=len(requests),
            total_batches=len(batches),
            window_ms=self.window_ms,
            min_batch_size=int(bs_arr.min()),
            max_batch_size=int(bs_arr.max()),
            mean_batch_size=float(bs_arr.mean()),
            buckets=buckets,
            optimal_batch_size=optimal_size,
            recommendation=recommendation,
        )

    @staticmethod
    def _classify_efficiency(score: float, best_score: float) -> BatchEfficiency:
        """Classify efficiency relative to the best observed score."""
        if best_score <= 0:
            return BatchEfficiency.ACCEPTABLE
        ratio = score / best_score
        if ratio >= 0.9:
            return BatchEfficiency.OPTIMAL
        if ratio >= 0.7:
            return BatchEfficiency.GOOD
        if ratio >= 0.4:
            return BatchEfficiency.ACCEPTABLE
        return BatchEfficiency.POOR


def analyze_batch_impact(
    benchmark_path: str,
    window_ms: float = 100.0,
) -> dict:
    """Programmatic API for batch size impact analysis.

    Args:
        benchmark_path: Path to benchmark JSON file.
        window_ms: Temporal grouping window in milliseconds.

    Returns:
        Dictionary with batch analysis results.
    """
    from .bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    analyzer = BatchAnalyzer(window_ms=window_ms)
    report = analyzer.analyze(data)
    return report.model_dump()

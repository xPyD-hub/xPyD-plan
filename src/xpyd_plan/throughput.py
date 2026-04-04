"""Throughput percentile analysis — per-second request completion distribution."""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class ThroughputStability(str, Enum):
    """Throughput stability classification based on coefficient of variation."""

    STABLE = "STABLE"
    VARIABLE = "VARIABLE"
    UNSTABLE = "UNSTABLE"


class ThroughputBucket(BaseModel):
    """A single time bucket with its throughput count."""

    bucket_start: float = Field(..., description="Bucket start time (epoch seconds)")
    bucket_end: float = Field(..., description="Bucket end time (epoch seconds)")
    request_count: int = Field(..., ge=0, description="Completed requests in this bucket")


class ThroughputStats(BaseModel):
    """Throughput distribution statistics."""

    min_rps: float = Field(..., ge=0, description="Minimum requests per second")
    mean_rps: float = Field(..., ge=0, description="Mean requests per second")
    p5_rps: float = Field(..., ge=0, description="P5 requests per second (sustainable)")
    p50_rps: float = Field(..., ge=0, description="Median requests per second")
    p95_rps: float = Field(..., ge=0, description="P95 requests per second")
    p99_rps: float = Field(..., ge=0, description="P99 requests per second")
    max_rps: float = Field(..., ge=0, description="Maximum requests per second")
    cv: float = Field(..., ge=0, description="Coefficient of variation")
    stability: ThroughputStability = Field(
        ..., description="Stability classification"
    )


class ThroughputReport(BaseModel):
    """Complete throughput percentile analysis report."""

    bucket_size: float = Field(..., gt=0, description="Bucket size in seconds")
    total_buckets: int = Field(..., ge=0, description="Total number of time buckets")
    total_requests: int = Field(..., ge=0, description="Total requests analyzed")
    duration_seconds: float = Field(..., ge=0, description="Benchmark duration")
    stats: ThroughputStats = Field(..., description="Throughput distribution stats")
    bottleneck_buckets: list[ThroughputBucket] = Field(
        default_factory=list,
        description="Buckets where throughput fell below sustainable threshold",
    )
    sustainable_rps: float = Field(
        ..., ge=0, description="Sustainable throughput (P5)"
    )
    recommendation: str = Field(..., description="Human-readable recommendation")


class ThroughputAnalyzer:
    """Analyze per-second throughput distribution from benchmark data."""

    def __init__(self, bucket_size: float = 1.0) -> None:
        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        self._bucket_size = bucket_size

    def analyze(self, data: BenchmarkData) -> ThroughputReport:
        """Analyze throughput distribution from benchmark data."""
        if not data.requests:
            return ThroughputReport(
                bucket_size=self._bucket_size,
                total_buckets=0,
                total_requests=0,
                duration_seconds=0.0,
                stats=ThroughputStats(
                    min_rps=0.0,
                    mean_rps=0.0,
                    p5_rps=0.0,
                    p50_rps=0.0,
                    p95_rps=0.0,
                    p99_rps=0.0,
                    max_rps=0.0,
                    cv=0.0,
                    stability=ThroughputStability.STABLE,
                ),
                bottleneck_buckets=[],
                sustainable_rps=0.0,
                recommendation="No requests to analyze.",
            )

        # Compute completion timestamps (start + total_latency)
        completion_times = []
        for req in data.requests:
            ts = req.timestamp
            # completion = timestamp + total_latency (converted to seconds)
            completion = ts + req.total_latency_ms / 1000.0
            completion_times.append(completion)

        completion_times.sort()

        t_min = completion_times[0]
        t_max = completion_times[-1]
        duration = t_max - t_min

        if duration <= 0:
            # All completions at same instant
            return ThroughputReport(
                bucket_size=self._bucket_size,
                total_buckets=1,
                total_requests=len(data.requests),
                duration_seconds=0.0,
                stats=ThroughputStats(
                    min_rps=float(len(data.requests)),
                    mean_rps=float(len(data.requests)),
                    p5_rps=float(len(data.requests)),
                    p50_rps=float(len(data.requests)),
                    p95_rps=float(len(data.requests)),
                    p99_rps=float(len(data.requests)),
                    max_rps=float(len(data.requests)),
                    cv=0.0,
                    stability=ThroughputStability.STABLE,
                ),
                bottleneck_buckets=[],
                sustainable_rps=float(len(data.requests)),
                recommendation="All requests completed at the same time.",
            )

        # Create buckets
        num_buckets = max(1, math.ceil(duration / self._bucket_size))
        buckets: list[ThroughputBucket] = []
        counts: list[int] = [0] * num_buckets

        for ct in completion_times:
            idx = min(int((ct - t_min) / self._bucket_size), num_buckets - 1)
            counts[idx] += 1

        for i in range(num_buckets):
            bucket_start = t_min + i * self._bucket_size
            bucket_end = bucket_start + self._bucket_size
            buckets.append(
                ThroughputBucket(
                    bucket_start=bucket_start,
                    bucket_end=bucket_end,
                    request_count=counts[i],
                )
            )

        # Normalize to requests per second
        rps_values = np.array([c / self._bucket_size for c in counts], dtype=float)

        min_rps = float(np.min(rps_values))
        mean_rps = float(np.mean(rps_values))
        p5_rps = float(np.percentile(rps_values, 5))
        p50_rps = float(np.percentile(rps_values, 50))
        p95_rps = float(np.percentile(rps_values, 95))
        p99_rps = float(np.percentile(rps_values, 99))
        max_rps = float(np.max(rps_values))

        std_rps = float(np.std(rps_values))
        cv = std_rps / mean_rps if mean_rps > 0 else 0.0

        if cv <= 0.15:
            stability = ThroughputStability.STABLE
        elif cv <= 0.40:
            stability = ThroughputStability.VARIABLE
        else:
            stability = ThroughputStability.UNSTABLE

        stats = ThroughputStats(
            min_rps=min_rps,
            mean_rps=mean_rps,
            p5_rps=p5_rps,
            p50_rps=p50_rps,
            p95_rps=p95_rps,
            p99_rps=p99_rps,
            max_rps=max_rps,
            cv=cv,
            stability=stability,
        )

        # Bottleneck detection: buckets below P5 (sustainable level)
        bottleneck_buckets = [b for b in buckets if b.request_count / self._bucket_size < p5_rps]

        # Recommendation
        if stability == ThroughputStability.STABLE:
            rec = (
                f"Throughput is stable (CV={cv:.2f}). "
                f"Sustainable throughput: {p5_rps:.1f} req/s. "
                f"Mean throughput: {mean_rps:.1f} req/s."
            )
        elif stability == ThroughputStability.VARIABLE:
            rec = (
                f"Throughput is variable (CV={cv:.2f}). "
                f"Consider investigating {len(bottleneck_buckets)} low-throughput periods. "
                f"Sustainable throughput: {p5_rps:.1f} req/s, but mean is {mean_rps:.1f} req/s."
            )
        else:
            rec = (
                f"Throughput is unstable (CV={cv:.2f}). "
                f"Significant variability detected with "
                f"{len(bottleneck_buckets)} bottleneck periods. "
                f"Sustainable throughput ({p5_rps:.1f} req/s) is much lower "
                f"than mean ({mean_rps:.1f} req/s). "
                f"Investigate load balancing or resource contention."
            )

        return ThroughputReport(
            bucket_size=self._bucket_size,
            total_buckets=num_buckets,
            total_requests=len(data.requests),
            duration_seconds=duration,
            stats=stats,
            bottleneck_buckets=bottleneck_buckets,
            sustainable_rps=p5_rps,
            recommendation=rec,
        )


def analyze_throughput(
    benchmark_path: str,
    bucket_size: float = 1.0,
) -> dict:
    """Programmatic API for throughput analysis.

    Args:
        benchmark_path: Path to benchmark JSON file.
        bucket_size: Time bucket size in seconds (default: 1.0).

    Returns:
        dict: ThroughputReport as a dictionary.
    """
    from .bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    analyzer = ThroughputAnalyzer(bucket_size=bucket_size)
    report = analyzer.analyze(data)
    return report.model_dump()

"""Benchmark sampling and downsampling with statistical validation."""

from __future__ import annotations

import random
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData, BenchmarkRequest


class SamplingMethod(str, Enum):
    """Available sampling methods."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    RESERVOIR = "reservoir"


class MetricDeviation(BaseModel):
    """Deviation between original and sampled data for a single metric."""

    metric: str = Field(..., description="Metric name (e.g. ttft_ms)")
    original_p50: float = Field(..., description="Original P50 value")
    original_p95: float = Field(..., description="Original P95 value")
    original_p99: float = Field(..., description="Original P99 value")
    sampled_p50: float = Field(..., description="Sampled P50 value")
    sampled_p95: float = Field(..., description="Sampled P95 value")
    sampled_p99: float = Field(..., description="Sampled P99 value")
    p50_deviation_pct: float = Field(..., description="P50 relative deviation (%)")
    p95_deviation_pct: float = Field(..., description="P95 relative deviation (%)")
    p99_deviation_pct: float = Field(..., description="P99 relative deviation (%)")


class SampleValidation(BaseModel):
    """Statistical validation of sample quality."""

    deviations: list[MetricDeviation] = Field(
        default_factory=list, description="Per-metric deviations"
    )
    max_deviation_pct: float = Field(..., description="Worst-case deviation across all metrics (%)")
    is_representative: bool = Field(
        ..., description="True if max deviation is within tolerance"
    )
    tolerance_pct: float = Field(..., description="Tolerance threshold used (%)")


class SampleConfig(BaseModel):
    """Configuration for sampling."""

    method: SamplingMethod = Field(SamplingMethod.RANDOM, description="Sampling method")
    size: int = Field(..., ge=1, description="Target sample size")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    bins: int = Field(4, ge=2, description="Number of bins for stratified sampling")
    tolerance_pct: float = Field(
        5.0, ge=0, description="Max acceptable deviation percentage"
    )


class SampleResult(BaseModel):
    """Result of a sampling operation."""

    method: SamplingMethod = Field(..., description="Sampling method used")
    original_count: int = Field(..., description="Original request count")
    sample_count: int = Field(..., description="Sampled request count")
    sample_fraction: float = Field(..., description="Fraction of original retained")
    data: BenchmarkData = Field(..., description="Sampled benchmark data")
    validation: SampleValidation = Field(..., description="Statistical validation")


def _pct_dev(original: float, sampled: float) -> float:
    """Compute relative deviation percentage."""
    if original == 0:
        return 0.0 if sampled == 0 else 100.0
    return abs(sampled - original) / original * 100.0


def _compute_validation(
    original_requests: list[BenchmarkRequest],
    sampled_requests: list[BenchmarkRequest],
    tolerance_pct: float,
) -> SampleValidation:
    """Compare percentile distributions between original and sample."""
    metrics = ["ttft_ms", "tpot_ms", "total_latency_ms"]
    deviations: list[MetricDeviation] = []
    worst = 0.0

    for metric in metrics:
        orig_vals = np.array([getattr(r, metric) for r in original_requests])
        samp_vals = np.array([getattr(r, metric) for r in sampled_requests])

        op50 = float(np.percentile(orig_vals, 50))
        op95 = float(np.percentile(orig_vals, 95))
        op99 = float(np.percentile(orig_vals, 99))
        sp50 = float(np.percentile(samp_vals, 50))
        sp95 = float(np.percentile(samp_vals, 95))
        sp99 = float(np.percentile(samp_vals, 99))

        d50 = _pct_dev(op50, sp50)
        d95 = _pct_dev(op95, sp95)
        d99 = _pct_dev(op99, sp99)

        worst = max(worst, d50, d95, d99)

        deviations.append(MetricDeviation(
            metric=metric,
            original_p50=op50, original_p95=op95, original_p99=op99,
            sampled_p50=sp50, sampled_p95=sp95, sampled_p99=sp99,
            p50_deviation_pct=round(d50, 2),
            p95_deviation_pct=round(d95, 2),
            p99_deviation_pct=round(d99, 2),
        ))

    return SampleValidation(
        deviations=deviations,
        max_deviation_pct=round(worst, 2),
        is_representative=worst <= tolerance_pct,
        tolerance_pct=tolerance_pct,
    )


class BenchmarkSampler:
    """Downsample benchmark data while preserving statistical properties."""

    def __init__(self, seed: int | None = None, tolerance_pct: float = 5.0) -> None:
        self._seed = seed
        self._tolerance_pct = tolerance_pct

    def random_sample(self, data: BenchmarkData, size: int) -> SampleResult:
        """Simple random sampling without replacement."""
        requests = list(data.requests)
        if size >= len(requests):
            return self._build_result(data, requests, SamplingMethod.RANDOM)

        rng = random.Random(self._seed)
        sampled = rng.sample(requests, size)
        return self._build_result(data, sampled, SamplingMethod.RANDOM)

    def stratified_sample(self, data: BenchmarkData, size: int, bins: int = 4) -> SampleResult:
        """Stratified sampling by prompt_tokens bins to preserve distribution."""
        requests = list(data.requests)
        if size >= len(requests):
            return self._build_result(data, requests, SamplingMethod.STRATIFIED)

        # Bin by prompt_tokens
        token_vals = [r.prompt_tokens for r in requests]
        edges = np.percentile(token_vals, np.linspace(0, 100, bins + 1))
        edges[0] -= 1  # include minimum

        buckets: list[list[BenchmarkRequest]] = [[] for _ in range(bins)]
        for r in requests:
            for i in range(bins):
                if r.prompt_tokens > edges[i] and r.prompt_tokens <= edges[i + 1]:
                    buckets[i].append(r)
                    break
            else:
                buckets[-1].append(r)

        rng = random.Random(self._seed)
        sampled: list[BenchmarkRequest] = []
        for bucket in buckets:
            if not bucket:
                continue
            n = max(1, round(size * len(bucket) / len(requests)))
            n = min(n, len(bucket))
            sampled.extend(rng.sample(bucket, n))

        # Trim or pad to exact size
        if len(sampled) > size:
            sampled = rng.sample(sampled, size)

        return self._build_result(data, sampled, SamplingMethod.STRATIFIED)

    def reservoir_sample(self, data: BenchmarkData, size: int) -> SampleResult:
        """Reservoir sampling (Algorithm R) — works on streaming/unknown-size inputs."""
        requests = list(data.requests)
        if size >= len(requests):
            return self._build_result(data, requests, SamplingMethod.RESERVOIR)

        rng = random.Random(self._seed)
        reservoir = list(requests[:size])
        for i in range(size, len(requests)):
            j = rng.randint(0, i)
            if j < size:
                reservoir[j] = requests[i]

        return self._build_result(data, reservoir, SamplingMethod.RESERVOIR)

    def _build_result(
        self,
        original: BenchmarkData,
        sampled_requests: list[BenchmarkRequest],
        method: SamplingMethod,
    ) -> SampleResult:
        original_requests = list(original.requests)
        fraction = len(sampled_requests) / len(original_requests) if original_requests else 1.0

        sampled_data = BenchmarkData(
            metadata=original.metadata.model_copy(
                update={"measured_qps": original.metadata.measured_qps * fraction}
            ),
            requests=sampled_requests,
        )

        validation = _compute_validation(
            original_requests, sampled_requests, self._tolerance_pct
        )

        return SampleResult(
            method=method,
            original_count=len(original_requests),
            sample_count=len(sampled_requests),
            sample_fraction=round(fraction, 4),
            data=sampled_data,
            validation=validation,
        )


def sample_benchmark(
    data: BenchmarkData,
    method: SamplingMethod = SamplingMethod.RANDOM,
    size: int = 1000,
    seed: int | None = None,
    bins: int = 4,
    tolerance_pct: float = 5.0,
) -> SampleResult:
    """Programmatic API for benchmark sampling."""
    sampler = BenchmarkSampler(seed=seed, tolerance_pct=tolerance_pct)
    if method == SamplingMethod.RANDOM:
        return sampler.random_sample(data, size)
    elif method == SamplingMethod.STRATIFIED:
        return sampler.stratified_sample(data, size, bins=bins)
    elif method == SamplingMethod.RESERVOIR:
        return sampler.reservoir_sample(data, size)
    else:
        msg = f"Unknown sampling method: {method}"
        raise ValueError(msg)

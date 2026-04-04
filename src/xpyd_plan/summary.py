"""Benchmark quick summary statistics."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class TokenStats(BaseModel):
    """Token distribution statistics."""

    min: int = Field(..., description="Minimum token count")
    max: int = Field(..., description="Maximum token count")
    mean: float = Field(..., description="Mean token count")
    p50: float = Field(..., description="Median token count")
    p95: float = Field(..., description="P95 token count")


class LatencyOverview(BaseModel):
    """Latency distribution overview for a single metric."""

    min_ms: float = Field(..., description="Minimum latency (ms)")
    max_ms: float = Field(..., description="Maximum latency (ms)")
    mean_ms: float = Field(..., description="Mean latency (ms)")
    p50_ms: float = Field(..., description="P50 latency (ms)")
    p95_ms: float = Field(..., description="P95 latency (ms)")
    p99_ms: float = Field(..., description="P99 latency (ms)")


class SummaryReport(BaseModel):
    """Compact benchmark summary report."""

    request_count: int = Field(..., description="Total number of requests")
    duration_seconds: float = Field(..., description="Benchmark duration in seconds")
    measured_qps: float = Field(..., description="Measured queries per second")
    num_prefill_instances: int = Field(..., description="Prefill instance count")
    num_decode_instances: int = Field(..., description="Decode instance count")
    pd_ratio: str = Field(..., description="P:D ratio string (e.g. '2:6')")
    prompt_tokens: TokenStats = Field(..., description="Prompt token distribution")
    output_tokens: TokenStats = Field(..., description="Output token distribution")
    ttft: LatencyOverview = Field(..., description="TTFT latency overview")
    tpot: LatencyOverview = Field(..., description="TPOT latency overview")
    total_latency: LatencyOverview = Field(..., description="Total latency overview")


class SummaryGenerator:
    """Generate quick summary statistics from benchmark data."""

    def __init__(self, data: BenchmarkData) -> None:
        self._data = data

    def generate(self) -> SummaryReport:
        """Produce a compact summary of the benchmark dataset."""
        requests = self._data.requests
        meta = self._data.metadata

        prompt_arr = np.array([r.prompt_tokens for r in requests])
        output_arr = np.array([r.output_tokens for r in requests])
        ttft_arr = np.array([r.ttft_ms for r in requests])
        tpot_arr = np.array([r.tpot_ms for r in requests])
        total_arr = np.array([r.total_latency_ms for r in requests])
        timestamps = np.array([r.timestamp for r in requests])

        duration = float(timestamps.max() - timestamps.min()) if len(timestamps) > 1 else 0.0

        return SummaryReport(
            request_count=len(requests),
            duration_seconds=round(duration, 2),
            measured_qps=meta.measured_qps,
            num_prefill_instances=meta.num_prefill_instances,
            num_decode_instances=meta.num_decode_instances,
            pd_ratio=f"{meta.num_prefill_instances}:{meta.num_decode_instances}",
            prompt_tokens=self._token_stats(prompt_arr),
            output_tokens=self._token_stats(output_arr),
            ttft=self._latency_overview(ttft_arr),
            tpot=self._latency_overview(tpot_arr),
            total_latency=self._latency_overview(total_arr),
        )

    @staticmethod
    def _token_stats(arr: np.ndarray) -> TokenStats:
        return TokenStats(
            min=int(arr.min()),
            max=int(arr.max()),
            mean=round(float(arr.mean()), 1),
            p50=round(float(np.percentile(arr, 50)), 1),
            p95=round(float(np.percentile(arr, 95)), 1),
        )

    @staticmethod
    def _latency_overview(arr: np.ndarray) -> LatencyOverview:
        return LatencyOverview(
            min_ms=round(float(arr.min()), 2),
            max_ms=round(float(arr.max()), 2),
            mean_ms=round(float(arr.mean()), 2),
            p50_ms=round(float(np.percentile(arr, 50)), 2),
            p95_ms=round(float(np.percentile(arr, 95)), 2),
            p99_ms=round(float(np.percentile(arr, 99)), 2),
        )


def summarize_benchmark(data: BenchmarkData) -> SummaryReport:
    """Programmatic API: generate summary report from benchmark data."""
    return SummaryGenerator(data).generate()

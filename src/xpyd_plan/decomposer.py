"""Latency decomposition analysis for benchmark data."""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class BottleneckType(str, Enum):
    """Classification of where latency is predominantly spent."""

    PREFILL_BOUND = "prefill_bound"
    DECODE_BOUND = "decode_bound"
    OVERHEAD_BOUND = "overhead_bound"
    BALANCED = "balanced"


class DecomposedRequest(BaseModel):
    """Latency decomposition for a single request."""

    request_id: str = Field(description="Request identifier")
    total_latency_ms: float = Field(description="Total end-to-end latency (ms)")
    prefill_ms: float = Field(description="Prefill phase duration (TTFT) (ms)")
    decode_ms: float = Field(description="Decode phase duration (output_tokens × tpot_ms) (ms)")
    overhead_ms: float = Field(description="Overhead (total - prefill - decode), clamped to 0 (ms)")
    prefill_fraction: float = Field(description="Fraction of total latency in prefill phase")
    decode_fraction: float = Field(description="Fraction of total latency in decode phase")
    overhead_fraction: float = Field(description="Fraction of total latency in overhead")


class PhaseStats(BaseModel):
    """Aggregate statistics for a latency phase."""

    phase: str = Field(description="Phase name (prefill, decode, overhead)")
    mean_fraction: float = Field(description="Mean fraction across all requests")
    p50_fraction: float = Field(description="P50 fraction")
    p95_fraction: float = Field(description="P95 fraction")
    mean_ms: float = Field(description="Mean absolute duration (ms)")
    p50_ms: float = Field(description="P50 absolute duration (ms)")
    p95_ms: float = Field(description="P95 absolute duration (ms)")


class DecompositionReport(BaseModel):
    """Complete latency decomposition report."""

    total_requests: int = Field(description="Total requests analyzed")
    phase_stats: list[PhaseStats] = Field(description="Per-phase aggregate statistics")
    bottleneck: BottleneckType = Field(description="Overall bottleneck classification")
    bottleneck_threshold: float = Field(
        description="Fraction threshold used for bottleneck classification"
    )
    recommendation: str = Field(description="Actionable recommendation based on bottleneck")
    decomposed_requests: list[DecomposedRequest] = Field(
        default_factory=list, description="Per-request decomposition (optional detail)"
    )


def _decompose_request(
    request_id: str,
    ttft_ms: float,
    tpot_ms: float,
    output_tokens: int,
    total_latency_ms: float,
) -> DecomposedRequest:
    """Decompose a single request's latency into phases."""
    prefill_ms = ttft_ms
    decode_ms = output_tokens * tpot_ms
    overhead_ms = max(0.0, total_latency_ms - prefill_ms - decode_ms)

    if total_latency_ms > 0:
        # Recalculate fractions after clamping overhead
        actual_total = prefill_ms + decode_ms + overhead_ms
        prefill_frac = prefill_ms / actual_total
        decode_frac = decode_ms / actual_total
        overhead_frac = overhead_ms / actual_total
    else:
        prefill_frac = 0.0
        decode_frac = 0.0
        overhead_frac = 0.0

    return DecomposedRequest(
        request_id=request_id,
        total_latency_ms=total_latency_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        overhead_ms=overhead_ms,
        prefill_fraction=prefill_frac,
        decode_fraction=decode_frac,
        overhead_fraction=overhead_frac,
    )


def _classify_bottleneck(
    mean_prefill_frac: float,
    mean_decode_frac: float,
    mean_overhead_frac: float,
    threshold: float,
) -> BottleneckType:
    """Classify the bottleneck based on mean phase fractions."""
    if mean_prefill_frac >= threshold:
        return BottleneckType.PREFILL_BOUND
    if mean_decode_frac >= threshold:
        return BottleneckType.DECODE_BOUND
    if mean_overhead_frac >= threshold:
        return BottleneckType.OVERHEAD_BOUND
    return BottleneckType.BALANCED


def _recommendation(bottleneck: BottleneckType) -> str:
    """Return actionable recommendation based on bottleneck type."""
    recommendations = {
        BottleneckType.PREFILL_BOUND: (
            "Prefill phase dominates latency. Consider adding more prefill instances "
            "or optimizing prompt processing (shorter prompts, prefix caching)."
        ),
        BottleneckType.DECODE_BOUND: (
            "Decode phase dominates latency. Consider adding more decode instances "
            "or optimizing generation (speculative decoding, reduced output length)."
        ),
        BottleneckType.OVERHEAD_BOUND: (
            "Overhead (scheduling, network, queuing) dominates latency. "
            "Investigate infrastructure bottlenecks, load balancer configuration, "
            "and request queuing behavior."
        ),
        BottleneckType.BALANCED: (
            "Latency is balanced across phases. No single bottleneck dominates. "
            "Consider overall scaling or SLA relaxation if needed."
        ),
    }
    return recommendations[bottleneck]


class LatencyDecomposer:
    """Decompose benchmark latency into prefill, decode, and overhead phases."""

    def __init__(self, data: BenchmarkData, *, bottleneck_threshold: float = 0.5) -> None:
        self._data = data
        self._threshold = bottleneck_threshold

    def analyze(self, *, include_requests: bool = False) -> DecompositionReport:
        """Run latency decomposition analysis."""
        decomposed = [
            _decompose_request(
                request_id=r.request_id,
                ttft_ms=r.ttft_ms,
                tpot_ms=r.tpot_ms,
                output_tokens=r.output_tokens,
                total_latency_ms=r.total_latency_ms,
            )
            for r in self._data.requests
        ]

        prefill_fracs = np.array([d.prefill_fraction for d in decomposed])
        decode_fracs = np.array([d.decode_fraction for d in decomposed])
        overhead_fracs = np.array([d.overhead_fraction for d in decomposed])

        prefill_ms = np.array([d.prefill_ms for d in decomposed])
        decode_ms_arr = np.array([d.decode_ms for d in decomposed])
        overhead_ms_arr = np.array([d.overhead_ms for d in decomposed])

        def _phase_stats(name: str, fracs: np.ndarray, ms_arr: np.ndarray) -> PhaseStats:
            return PhaseStats(
                phase=name,
                mean_fraction=float(np.mean(fracs)),
                p50_fraction=float(np.percentile(fracs, 50)),
                p95_fraction=float(np.percentile(fracs, 95)),
                mean_ms=float(np.mean(ms_arr)),
                p50_ms=float(np.percentile(ms_arr, 50)),
                p95_ms=float(np.percentile(ms_arr, 95)),
            )

        phase_stats = [
            _phase_stats("prefill", prefill_fracs, prefill_ms),
            _phase_stats("decode", decode_fracs, decode_ms_arr),
            _phase_stats("overhead", overhead_fracs, overhead_ms_arr),
        ]

        mean_pf = float(np.mean(prefill_fracs))
        mean_df = float(np.mean(decode_fracs))
        mean_of = float(np.mean(overhead_fracs))

        bottleneck = _classify_bottleneck(mean_pf, mean_df, mean_of, self._threshold)

        return DecompositionReport(
            total_requests=len(decomposed),
            phase_stats=phase_stats,
            bottleneck=bottleneck,
            bottleneck_threshold=self._threshold,
            recommendation=_recommendation(bottleneck),
            decomposed_requests=decomposed if include_requests else [],
        )


def decompose_latency(
    data: BenchmarkData,
    *,
    bottleneck_threshold: float = 0.5,
    include_requests: bool = False,
) -> dict:
    """Programmatic API: decompose benchmark latency into phases.

    Returns a dict representation of DecompositionReport.
    """
    decomposer = LatencyDecomposer(data, bottleneck_threshold=bottleneck_threshold)
    report = decomposer.analyze(include_requests=include_requests)
    return report.model_dump()

"""Performance estimator — predict TTFT/TPOT for a given PD configuration."""

from __future__ import annotations

from xpyd_plan.models import (
    DatasetStats,
    GPUProfile,
    PDConfig,
    PerformanceEstimate,
)


def estimate_performance(
    dataset: DatasetStats,
    gpu: GPUProfile,
    config: PDConfig,
) -> PerformanceEstimate:
    """Estimate performance metrics for a PD configuration.

    Uses a simplified linear model (v1):
    - TTFT = prompt_tokens_p95 / (prefill_throughput_per_gpu * num_prefill) * 1000
    - TPOT = (num_concurrent_decode_requests / (decode_throughput_per_gpu * num_decode)) * 1000
      where concurrent decode requests ≈ num_requests spread across decode GPUs
    - throughput = decode_throughput_per_gpu * num_decode / output_len_mean
    - cost = total_gpus * cost_per_hour
    """
    # TTFT: time to process the P95 prompt on the prefill fleet
    total_prefill_tps = gpu.prefill_tokens_per_sec * config.num_prefill
    ttft_ms = (dataset.prompt_len_p95 / total_prefill_tps) * 1000.0

    # TPOT: per-token decode latency under load
    # Model: each decode GPU handles (num_requests / num_decode) concurrent sequences
    # Each sequence needs 1 token/step, so effective TPOT = concurrent_seqs / decode_tps * 1000
    concurrent_per_gpu = dataset.num_requests / config.num_decode
    tpot_ms = (concurrent_per_gpu / gpu.decode_tokens_per_sec) * 1000.0

    # Throughput: total decode tokens/s / avg output length = requests/s
    total_decode_tps = gpu.decode_tokens_per_sec * config.num_decode
    throughput_rps = total_decode_tps / dataset.output_len_mean

    # Cost
    total_cost = config.total * gpu.cost_per_hour

    return PerformanceEstimate(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        throughput_rps=throughput_rps,
        total_cost_per_hour=total_cost,
    )

"""Performance estimator — predict TTFT/TPOT for a given PD configuration.

.. deprecated::
    This module uses mathematical models to simulate performance.
    Use :mod:`xpyd_plan.analyzer` (BenchmarkAnalyzer) for measured-data analysis instead.

V2: Queuing-theory based estimator (M/M/c) with batching effects.

Model:
  In PD disaggregation with continuous batching, each GPU processes multiple
  sequences concurrently. The per-request throughput degrades with batch size
  but sub-linearly (continuous batching is efficient).

  - batch_degradation(bs) ∈ (0, 1]: fraction of single-request throughput
    retained at batch size bs. Models memory-bandwidth sharing.
  - per_request_tps = base_tps * batch_degradation(concurrent_per_gpu)
  - TTFT = prompt_len / per_request_prefill_tps + queue_wait
  - TPOT = 1 / per_request_decode_tps + queue_wait
"""

from __future__ import annotations

import math

from xpyd_plan.models import (
    DatasetStats,
    GPUProfile,
    PDConfig,
    PerformanceEstimate,
)


def _erlang_c(c: int, rho_total: float) -> float:
    """Compute Erlang-C probability P(queuing) for an M/M/c system.

    Args:
        c: Number of servers (GPUs).
        rho_total: Total offered load (lambda / mu). Must be < c for stability.

    Returns:
        Probability that an arriving request has to wait.
    """
    if c <= 0:
        return 1.0
    if rho_total <= 0.0:
        return 0.0
    if rho_total >= c:
        return 1.0

    log_rho_c = c * math.log(rho_total) - math.lgamma(c + 1)
    factor = c / (c - rho_total)

    terms = []
    for k in range(c):
        if k == 0:
            terms.append(1.0)
        else:
            terms.append(math.exp(k * math.log(rho_total) - math.lgamma(k + 1)))

    erlang_c_num = math.exp(log_rho_c) * factor
    return erlang_c_num / (sum(terms) + erlang_c_num)


def _batch_degradation(batch_size: float) -> float:
    """Per-request throughput degradation factor due to batching.

    With continuous batching, a GPU shares memory bandwidth among concurrent
    sequences. Per-request throughput degrades but sub-linearly:

        degradation = 1 / batch_size^alpha,  alpha = 0.3

    This means at batch_size=1, factor=1.0 (full speed); at bs=8, ~0.52;
    at bs=64, ~0.27. The total GPU throughput (factor * bs) still grows.

    Returns a value in (0, 1].
    """
    if batch_size <= 1.0:
        return 1.0
    alpha = 0.3
    return 1.0 / (batch_size**alpha)


def estimate_performance(
    dataset: DatasetStats,
    gpu: GPUProfile,
    config: PDConfig,
) -> PerformanceEstimate:
    """Estimate performance for a PD configuration using queuing + batching model.

    Prefill (TTFT):
        - concurrent_per_gpu = num_requests / num_prefill
        - per_request_prefill_tps = prefill_tps * batch_degradation(concurrent)
        - service_time = prompt_len_p95 / per_request_prefill_tps
        - M/M/c queue wait layered on top

    Decode (TPOT):
        - concurrent_per_gpu = num_requests / num_decode
        - per_request_decode_tps = decode_tps * batch_degradation(concurrent)
        - service_time = 1 / per_request_decode_tps
        - M/M/c queue wait layered on top
    """
    # --- Prefill (TTFT) ---
    prefill_concurrent = dataset.num_requests / config.num_prefill
    prefill_degrad = _batch_degradation(prefill_concurrent)
    per_req_prefill_tps = gpu.prefill_tokens_per_sec * prefill_degrad

    # Service time for one P95 prompt on one GPU
    prefill_service_s = dataset.prompt_len_p95 / per_req_prefill_tps
    mu_prefill = 1.0 / prefill_service_s  # service rate (requests/s per GPU)

    # Arrival rate: steady-state where total arrival = total capacity * utilization_target
    # Use 70% target utilization to model a realistic operating point
    utilization = 0.7
    lambda_prefill = utilization * config.num_prefill * mu_prefill
    rho_prefill = lambda_prefill / mu_prefill  # = utilization * c

    if rho_prefill < config.num_prefill:
        p_q = _erlang_c(config.num_prefill, rho_prefill)
        avg_wait_s = p_q / (config.num_prefill * mu_prefill - lambda_prefill)
    else:
        avg_wait_s = prefill_service_s * 5.0

    ttft_ms = (prefill_service_s + avg_wait_s) * 1000.0

    # --- Decode (TPOT) ---
    decode_concurrent = dataset.num_requests / config.num_decode
    decode_degrad = _batch_degradation(decode_concurrent)
    per_req_decode_tps = gpu.decode_tokens_per_sec * decode_degrad

    decode_service_s = 1.0 / per_req_decode_tps
    mu_decode = per_req_decode_tps

    lambda_decode = utilization * config.num_decode * mu_decode
    rho_decode = lambda_decode / mu_decode

    if rho_decode < config.num_decode:
        p_q_d = _erlang_c(config.num_decode, rho_decode)
        avg_wait_decode_s = p_q_d / (config.num_decode * mu_decode - lambda_decode)
    else:
        avg_wait_decode_s = decode_service_s * 5.0

    tpot_ms = (decode_service_s + avg_wait_decode_s) * 1000.0

    # --- Throughput ---
    # Total system tokens/s = per_request_tps * concurrent_per_gpu * num_decode
    total_system_tps = per_req_decode_tps * decode_concurrent * config.num_decode
    throughput_rps = total_system_tps / dataset.output_len_mean

    # --- Cost ---
    total_cost = config.total * gpu.cost_per_hour

    return PerformanceEstimate(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        throughput_rps=throughput_rps,
        total_cost_per_hour=total_cost,
    )

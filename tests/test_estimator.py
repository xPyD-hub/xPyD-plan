"""Tests for performance estimator."""

from __future__ import annotations

from xpyd_plan.estimator import estimate_performance
from xpyd_plan.models import DatasetStats, GPUProfile, PDConfig


def _make_dataset(**kwargs) -> DatasetStats:
    defaults = dict(
        prompt_len_mean=500,
        prompt_len_p95=2000,
        output_len_mean=200,
        output_len_p95=800,
        num_requests=1000,
    )
    defaults.update(kwargs)
    return DatasetStats(**defaults)


def _make_gpu(**kwargs) -> GPUProfile:
    defaults = dict(
        name="A100-80G",
        prefill_tokens_per_sec=50000,
        decode_tokens_per_sec=2000,
        memory_gb=80,
        cost_per_hour=2.50,
    )
    defaults.update(kwargs)
    return GPUProfile(**defaults)


class TestEstimator:
    def test_ttft_scales_with_prefill_gpus(self):
        dataset = _make_dataset()
        gpu = _make_gpu()
        perf_1p = estimate_performance(dataset, gpu, PDConfig(num_prefill=1, num_decode=7))
        perf_2p = estimate_performance(dataset, gpu, PDConfig(num_prefill=2, num_decode=6))
        # Doubling prefill GPUs should halve TTFT
        assert abs(perf_1p.ttft_ms / perf_2p.ttft_ms - 2.0) < 0.01

    def test_tpot_scales_with_decode_gpus(self):
        dataset = _make_dataset()
        gpu = _make_gpu()
        perf_2d = estimate_performance(dataset, gpu, PDConfig(num_prefill=6, num_decode=2))
        perf_4d = estimate_performance(dataset, gpu, PDConfig(num_prefill=4, num_decode=4))
        # Doubling decode GPUs should halve TPOT
        assert abs(perf_2d.tpot_ms / perf_4d.tpot_ms - 2.0) < 0.01

    def test_cost_is_proportional_to_total_gpus(self):
        dataset = _make_dataset()
        gpu = _make_gpu()
        perf = estimate_performance(dataset, gpu, PDConfig(num_prefill=3, num_decode=5))
        assert perf.total_cost_per_hour == 8 * 2.50

    def test_throughput_positive(self):
        dataset = _make_dataset()
        gpu = _make_gpu()
        perf = estimate_performance(dataset, gpu, PDConfig(num_prefill=2, num_decode=6))
        assert perf.throughput_rps > 0

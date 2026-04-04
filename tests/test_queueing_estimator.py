"""Tests for the queuing-theory based estimator (M2)."""

from __future__ import annotations

import math

from xpyd_plan.estimator import (
    _batch_degradation,
    _erlang_c,
    estimate_performance,
)
from xpyd_plan.gpu_profiles import get_gpu_profile
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


class TestErlangC:
    def test_zero_load(self):
        assert _erlang_c(4, 0.0) == 0.0

    def test_single_server_light_load(self):
        """M/M/1 with rho=0.5 → Erlang-C = rho = 0.5."""
        p = _erlang_c(1, 0.5)
        assert abs(p - 0.5) < 0.01

    def test_saturated_returns_one(self):
        assert _erlang_c(4, 4.0) == 1.0
        assert _erlang_c(4, 5.0) == 1.0

    def test_more_servers_less_queuing(self):
        p4 = _erlang_c(4, 3.0)
        p8 = _erlang_c(8, 3.0)
        assert p8 < p4

    def test_zero_servers(self):
        assert _erlang_c(0, 1.0) == 1.0


class TestBatchDegradation:
    def test_single_request(self):
        assert _batch_degradation(1.0) == 1.0

    def test_below_one(self):
        assert _batch_degradation(0.5) == 1.0

    def test_degrades_with_batch_size(self):
        assert _batch_degradation(8) < _batch_degradation(1)

    def test_monotonically_decreasing(self):
        values = [_batch_degradation(bs) for bs in [1, 4, 16, 64, 256]]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_always_positive(self):
        assert _batch_degradation(10000) > 0


class TestQueueingEstimator:
    def test_ttft_scales_with_prefill_gpus(self):
        dataset = _make_dataset()
        gpu = _make_gpu()
        perf_1p = estimate_performance(dataset, gpu, PDConfig(num_prefill=1, num_decode=7))
        perf_4p = estimate_performance(dataset, gpu, PDConfig(num_prefill=4, num_decode=4))
        assert perf_4p.ttft_ms < perf_1p.ttft_ms

    def test_tpot_scales_with_decode_gpus(self):
        dataset = _make_dataset()
        gpu = _make_gpu()
        perf_2d = estimate_performance(dataset, gpu, PDConfig(num_prefill=6, num_decode=2))
        perf_6d = estimate_performance(dataset, gpu, PDConfig(num_prefill=2, num_decode=6))
        assert perf_6d.tpot_ms < perf_2d.tpot_ms

    def test_cost_proportional(self):
        dataset = _make_dataset()
        gpu = _make_gpu()
        perf = estimate_performance(dataset, gpu, PDConfig(num_prefill=3, num_decode=5))
        assert perf.total_cost_per_hour == 8 * 2.50

    def test_throughput_positive(self):
        dataset = _make_dataset()
        gpu = _make_gpu()
        perf = estimate_performance(dataset, gpu, PDConfig(num_prefill=2, num_decode=6))
        assert perf.throughput_rps > 0

    def test_h100_faster_than_a100(self):
        dataset = _make_dataset(num_requests=100)
        config = PDConfig(num_prefill=2, num_decode=6)
        perf_a100 = estimate_performance(dataset, get_gpu_profile("A100-80G"), config)
        perf_h100 = estimate_performance(dataset, get_gpu_profile("H100-80G"), config)
        assert perf_h100.ttft_ms < perf_a100.ttft_ms
        assert perf_h100.tpot_ms < perf_a100.tpot_ms

    def test_higher_load_increases_latency(self):
        gpu = _make_gpu()
        config = PDConfig(num_prefill=4, num_decode=4)
        perf_low = estimate_performance(_make_dataset(num_requests=10), gpu, config)
        perf_high = estimate_performance(_make_dataset(num_requests=500), gpu, config)
        assert perf_high.tpot_ms > perf_low.tpot_ms
        assert perf_high.ttft_ms > perf_low.ttft_ms

    def test_all_values_finite(self):
        dataset = _make_dataset()
        gpu = _make_gpu()
        perf = estimate_performance(dataset, gpu, PDConfig(num_prefill=4, num_decode=4))
        assert math.isfinite(perf.ttft_ms)
        assert math.isfinite(perf.tpot_ms)
        assert math.isfinite(perf.throughput_rps)
        assert math.isfinite(perf.total_cost_per_hour)

    def test_all_values_positive(self):
        dataset = _make_dataset()
        gpu = _make_gpu()
        perf = estimate_performance(dataset, gpu, PDConfig(num_prefill=4, num_decode=4))
        assert perf.ttft_ms > 0
        assert perf.tpot_ms > 0
        assert perf.throughput_rps > 0
        assert perf.total_cost_per_hour > 0

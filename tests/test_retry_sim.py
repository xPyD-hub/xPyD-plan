"""Tests for request retry impact simulator."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.retry_sim import (
    BackoffType,
    RetryConfig,
    RetrySimulator,
    _backoff_delay,
    _exceeds_threshold,
    simulate_retries,
)


def _make_request(
    request_id: str = "r1",
    prompt_tokens: int = 100,
    output_tokens: int = 50,
    ttft_ms: float = 50.0,
    tpot_ms: float = 20.0,
    total_latency_ms: float = 200.0,
    timestamp: float = 1000.0,
) -> BenchmarkRequest:
    return BenchmarkRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_latency_ms=total_latency_ms,
        timestamp=timestamp,
    )


def _make_data(requests: list[BenchmarkRequest], qps: float = 10.0) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=4,
            total_instances=6,
            measured_qps=qps,
        ),
        requests=requests,
    )


class TestExceedsThreshold:
    def test_no_threshold_set(self):
        cfg = RetryConfig(
            retry_threshold_ttft_ms=None,
            retry_threshold_tpot_ms=None,
            retry_threshold_total_ms=100.0,
        )
        assert not _exceeds_threshold(50.0, 20.0, 80.0, cfg)

    def test_ttft_exceeds(self):
        cfg = RetryConfig(retry_threshold_ttft_ms=100.0)
        assert _exceeds_threshold(150.0, 20.0, 200.0, cfg)

    def test_tpot_exceeds(self):
        cfg = RetryConfig(retry_threshold_tpot_ms=30.0)
        assert _exceeds_threshold(50.0, 40.0, 200.0, cfg)

    def test_total_exceeds(self):
        cfg = RetryConfig(retry_threshold_total_ms=150.0)
        assert _exceeds_threshold(50.0, 20.0, 200.0, cfg)

    def test_none_exceed(self):
        cfg = RetryConfig(
            retry_threshold_ttft_ms=100.0,
            retry_threshold_tpot_ms=50.0,
            retry_threshold_total_ms=500.0,
        )
        assert not _exceeds_threshold(50.0, 20.0, 200.0, cfg)


class TestBackoffDelay:
    def test_constant(self):
        cfg = RetryConfig(
            backoff_ms=100.0,
            backoff_type=BackoffType.CONSTANT,
            retry_threshold_total_ms=100.0,
        )
        assert _backoff_delay(0, cfg) == 100.0
        assert _backoff_delay(1, cfg) == 100.0
        assert _backoff_delay(2, cfg) == 100.0

    def test_exponential(self):
        cfg = RetryConfig(
            backoff_ms=100.0,
            backoff_type=BackoffType.EXPONENTIAL,
            retry_threshold_total_ms=100.0,
        )
        assert _backoff_delay(0, cfg) == 100.0
        assert _backoff_delay(1, cfg) == 200.0
        assert _backoff_delay(2, cfg) == 400.0


class TestRetrySimulatorInit:
    def test_no_threshold_raises(self):
        with pytest.raises(ValueError, match="At least one retry threshold"):
            RetrySimulator(RetryConfig(
                retry_threshold_ttft_ms=None,
                retry_threshold_tpot_ms=None,
                retry_threshold_total_ms=None,
            ))

    def test_valid_config(self):
        sim = RetrySimulator(RetryConfig(retry_threshold_total_ms=100.0))
        assert sim is not None


class TestRetrySimulatorSimulate:
    def test_single_request_no_retry(self):
        cfg = RetryConfig(retry_threshold_total_ms=500.0)
        sim = RetrySimulator(cfg)
        data = _make_data([_make_request(total_latency_ms=100.0)])
        report = sim.simulate(data)
        assert report.load_amplification.original_requests == 1
        assert report.effective_goodput == 1.0
        assert report.original_goodput == 1.0

    def test_no_retries_needed(self):
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=30.0, tpot_ms=10.0, total_latency_ms=100.0)
            for i in range(20)
        ]
        cfg = RetryConfig(retry_threshold_total_ms=200.0)
        sim = RetrySimulator(cfg)
        report = sim.simulate(_make_data(requests))

        assert report.load_amplification.retry_rate == 0.0
        assert report.load_amplification.amplification_factor == 1.0
        assert report.original_goodput == 1.0
        assert report.effective_goodput == 1.0
        assert "No requests exceed" in report.recommendation

    def test_all_requests_need_retries(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=300.0, timestamp=1000.0 + i)
            for i in range(10)
        ]
        cfg = RetryConfig(
            max_retries=2,
            retry_threshold_total_ms=100.0,
        )
        sim = RetrySimulator(cfg)
        report = sim.simulate(_make_data(requests))

        assert report.load_amplification.retry_rate == 1.0
        assert report.load_amplification.amplification_factor > 1.0
        assert report.original_goodput == 0.0

    def test_some_requests_need_retries(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=50.0, timestamp=1000.0 + i)
            for i in range(5)
        ] + [
            _make_request(request_id=f"r{i+5}", total_latency_ms=300.0, timestamp=1000.0 + i + 5)
            for i in range(5)
        ]
        cfg = RetryConfig(retry_threshold_total_ms=200.0, max_retries=3)
        sim = RetrySimulator(cfg)
        report = sim.simulate(_make_data(requests))

        assert report.load_amplification.retry_rate == 0.5
        assert report.original_goodput == 0.5
        assert report.effective_goodput >= report.original_goodput

    def test_latency_impact_has_three_metrics(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=300.0, timestamp=1000.0 + i)
            for i in range(10)
        ]
        cfg = RetryConfig(retry_threshold_total_ms=200.0)
        sim = RetrySimulator(cfg)
        report = sim.simulate(_make_data(requests))

        assert len(report.latency_impact) == 3
        metrics = {i.metric for i in report.latency_impact}
        assert metrics == {"ttft", "tpot", "total_latency"}

    def test_amplification_factor_correct(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=300.0, timestamp=1000.0 + i)
            for i in range(10)
        ]
        cfg = RetryConfig(
            max_retries=2,
            retry_threshold_total_ms=100.0,
        )
        sim = RetrySimulator(cfg)
        report = sim.simulate(_make_data(requests))

        la = report.load_amplification
        expected = (la.original_requests + la.total_retry_attempts) / la.original_requests
        assert abs(la.amplification_factor - expected) < 0.001

    def test_exponential_backoff(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=300.0, timestamp=1000.0 + i)
            for i in range(5)
        ]
        cfg = RetryConfig(
            max_retries=3,
            retry_threshold_total_ms=100.0,
            backoff_ms=50.0,
            backoff_type=BackoffType.EXPONENTIAL,
        )
        sim = RetrySimulator(cfg)
        report = sim.simulate(_make_data(requests))
        assert report.load_amplification.total_retry_attempts > 0

    def test_high_amplification_warning(self):
        # All requests fail with very tight threshold, forcing max retries
        requests = [
            _make_request(
                request_id=f"r{i}", ttft_ms=500.0, tpot_ms=500.0,
                total_latency_ms=5000.0, timestamp=1000.0 + i,
            )
            for i in range(10)
        ]
        cfg = RetryConfig(
            max_retries=10,
            retry_threshold_total_ms=1.0,  # Very tight
        )
        sim = RetrySimulator(cfg)
        report = sim.simulate(_make_data(requests))
        assert report.load_amplification.amplification_factor > 2.0
        assert "amplification" in report.recommendation.lower()

    def test_report_serializable(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=300.0, timestamp=1000.0 + i)
            for i in range(5)
        ]
        cfg = RetryConfig(retry_threshold_total_ms=200.0)
        sim = RetrySimulator(cfg)
        report = sim.simulate(_make_data(requests))
        d = report.model_dump()
        assert isinstance(d, dict)
        json_str = json.dumps(d)
        assert "load_amplification" in json_str


class TestSimulateRetriesAPI:
    def test_api_function(self, tmp_path):
        data = _make_data([
            _make_request(request_id=f"r{i}", total_latency_ms=50.0 + i * 30, timestamp=1000.0 + i)
            for i in range(20)
        ])
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        result = simulate_retries(
            str(path),
            retry_threshold_total_ms=200.0,
            max_retries=2,
        )
        assert isinstance(result, dict)
        assert "load_amplification" in result
        assert "effective_goodput" in result


class TestRetryConfig:
    def test_default_values(self):
        cfg = RetryConfig(retry_threshold_total_ms=100.0)
        assert cfg.max_retries == 3
        assert cfg.backoff_ms == 100.0
        assert cfg.backoff_type == BackoffType.CONSTANT

    def test_exponential_type(self):
        cfg = RetryConfig(
            retry_threshold_total_ms=100.0,
            backoff_type=BackoffType.EXPONENTIAL,
        )
        assert cfg.backoff_type == BackoffType.EXPONENTIAL

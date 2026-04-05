"""Tests for retry policy optimizer."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.retry_optimizer import (
    PolicyCandidate,
    RetryOptimizer,
    RetryOptimizerConfig,
    _compute_pareto,
    optimize_retry_policy,
)
from xpyd_plan.retry_sim import RetryConfig


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


class TestRetryOptimizerInit:
    def test_no_sla_raises(self):
        with pytest.raises(ValueError, match="At least one SLA threshold"):
            RetryOptimizer(RetryOptimizerConfig(
                sla_ttft_ms=None, sla_tpot_ms=None, sla_total_ms=None,
            ))

    def test_valid_config(self):
        opt = RetryOptimizer(RetryOptimizerConfig(sla_total_ms=300.0))
        assert opt is not None


class TestRetryOptimizerOptimize:
    def test_single_request_passing(self):
        requests = [_make_request(total_latency_ms=50.0)]
        opt = RetryOptimizer(RetryOptimizerConfig(sla_total_ms=300.0))
        report = opt.optimize(_make_data(requests))
        assert report.candidates_evaluated > 0

    def test_all_passing_no_improvement(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=50.0, timestamp=1000.0 + i)
            for i in range(20)
        ]
        opt = RetryOptimizer(RetryOptimizerConfig(sla_total_ms=300.0))
        report = opt.optimize(_make_data(requests))
        assert report.candidates_evaluated > 0
        # All requests already pass, so goodput improvement should be minimal
        if report.best_policy:
            assert report.best_policy.effective_goodput >= 0.99

    def test_some_failing_finds_improvement(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=50.0, timestamp=1000.0 + i)
            for i in range(15)
        ] + [
            _make_request(request_id=f"r{i+15}", total_latency_ms=400.0, timestamp=1000.0 + i + 15)
            for i in range(5)
        ]
        opt = RetryOptimizer(RetryOptimizerConfig(sla_total_ms=300.0))
        report = opt.optimize(_make_data(requests))
        assert report.candidates_evaluated > 0
        assert report.candidates_within_budget > 0
        assert report.best_policy is not None
        assert report.best_policy.effective_goodput >= 0.75

    def test_amplification_budget_respected(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=500.0, timestamp=1000.0 + i)
            for i in range(20)
        ]
        opt = RetryOptimizer(RetryOptimizerConfig(
            sla_total_ms=100.0,
            max_amplification=1.5,
        ))
        report = opt.optimize(_make_data(requests))
        for c in report.all_candidates:
            if c.within_budget:
                assert c.amplification_factor <= 1.5

    def test_pareto_frontier_non_empty(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=50.0 + i * 20, timestamp=1000.0 + i)
            for i in range(20)
        ]
        opt = RetryOptimizer(RetryOptimizerConfig(sla_total_ms=200.0))
        report = opt.optimize(_make_data(requests))
        assert len(report.pareto_frontier) > 0

    def test_pareto_sorted_by_amplification(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=50.0 + i * 20, timestamp=1000.0 + i)
            for i in range(20)
        ]
        opt = RetryOptimizer(RetryOptimizerConfig(sla_total_ms=200.0))
        report = opt.optimize(_make_data(requests))
        for i in range(len(report.pareto_frontier) - 1):
            assert (
                report.pareto_frontier[i].amplification_factor
                <= report.pareto_frontier[i + 1].amplification_factor
            )

    def test_report_serializable(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=300.0, timestamp=1000.0 + i)
            for i in range(10)
        ]
        opt = RetryOptimizer(RetryOptimizerConfig(sla_total_ms=200.0))
        report = opt.optimize(_make_data(requests))
        d = report.model_dump()
        assert isinstance(d, dict)
        json_str = json.dumps(d)
        assert "candidates_evaluated" in json_str

    def test_recommendation_text(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=50.0, timestamp=1000.0 + i)
            for i in range(10)
        ] + [
            _make_request(request_id=f"r{i+10}", total_latency_ms=400.0, timestamp=1000.0 + i + 10)
            for i in range(10)
        ]
        opt = RetryOptimizer(RetryOptimizerConfig(sla_total_ms=300.0))
        report = opt.optimize(_make_data(requests))
        assert len(report.recommendation) > 0

    def test_no_budget_feasible(self):
        # Very tight budget, all requests fail
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=5000.0, timestamp=1000.0 + i)
            for i in range(20)
        ]
        opt = RetryOptimizer(RetryOptimizerConfig(
            sla_total_ms=1.0,
            max_amplification=1.01,
        ))
        report = opt.optimize(_make_data(requests))
        # Might have no candidates within budget
        assert report.candidates_evaluated > 0

    def test_multiple_sla_metrics(self):
        requests = [
            _make_request(
                request_id=f"r{i}",
                ttft_ms=50.0 + i * 5,
                tpot_ms=10.0 + i * 2,
                total_latency_ms=100.0 + i * 20,
                timestamp=1000.0 + i,
            )
            for i in range(20)
        ]
        opt = RetryOptimizer(RetryOptimizerConfig(
            sla_ttft_ms=150.0,
            sla_tpot_ms=50.0,
            sla_total_ms=300.0,
        ))
        report = opt.optimize(_make_data(requests))
        assert report.candidates_evaluated > 0

    def test_best_policy_has_highest_goodput_in_budget(self):
        requests = [
            _make_request(request_id=f"r{i}", total_latency_ms=50.0 + i * 30, timestamp=1000.0 + i)
            for i in range(20)
        ]
        opt = RetryOptimizer(RetryOptimizerConfig(sla_total_ms=300.0))
        report = opt.optimize(_make_data(requests))
        if report.best_policy:
            within = [c for c in report.all_candidates if c.within_budget]
            max_goodput = max(c.effective_goodput for c in within)
            assert report.best_policy.effective_goodput == max_goodput


class TestComputePareto:
    def test_empty(self):
        assert _compute_pareto([]) == []

    def test_single_candidate(self):
        c = PolicyCandidate(
            config=RetryConfig(retry_threshold_total_ms=100.0),
            effective_goodput=0.9,
            original_goodput=0.8,
            goodput_improvement=0.1,
            amplification_factor=1.5,
            retry_rate=0.2,
            within_budget=True,
        )
        result = _compute_pareto([c])
        assert len(result) == 1

    def test_dominated_removed(self):
        c1 = PolicyCandidate(
            config=RetryConfig(retry_threshold_total_ms=100.0),
            effective_goodput=0.9,
            original_goodput=0.8,
            goodput_improvement=0.1,
            amplification_factor=1.5,
            retry_rate=0.2,
            within_budget=True,
        )
        c2 = PolicyCandidate(
            config=RetryConfig(retry_threshold_total_ms=200.0),
            effective_goodput=0.8,
            original_goodput=0.8,
            goodput_improvement=0.0,
            amplification_factor=1.8,
            retry_rate=0.3,
            within_budget=True,
        )
        # c2 is dominated by c1 (worse goodput and worse amplification)
        result = _compute_pareto([c1, c2])
        assert len(result) == 1
        assert result[0].effective_goodput == 0.9

    def test_non_dominated_kept(self):
        c1 = PolicyCandidate(
            config=RetryConfig(retry_threshold_total_ms=100.0),
            effective_goodput=0.9,
            original_goodput=0.8,
            goodput_improvement=0.1,
            amplification_factor=2.0,
            retry_rate=0.2,
            within_budget=True,
        )
        c2 = PolicyCandidate(
            config=RetryConfig(retry_threshold_total_ms=200.0),
            effective_goodput=0.85,
            original_goodput=0.8,
            goodput_improvement=0.05,
            amplification_factor=1.3,
            retry_rate=0.1,
            within_budget=True,
        )
        # Neither dominates the other
        result = _compute_pareto([c1, c2])
        assert len(result) == 2


class TestOptimizeRetryPolicyAPI:
    def test_api_function(self, tmp_path):
        data = _make_data([
            _make_request(request_id=f"r{i}", total_latency_ms=50.0 + i * 30, timestamp=1000.0 + i)
            for i in range(20)
        ])
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        result = optimize_retry_policy(
            str(path),
            sla_total_ms=300.0,
            max_amplification=2.0,
        )
        assert isinstance(result, dict)
        assert "candidates_evaluated" in result
        assert "best_policy" in result

    def test_api_custom_retries_range(self, tmp_path):
        data = _make_data([
            _make_request(request_id=f"r{i}", total_latency_ms=50.0 + i * 30, timestamp=1000.0 + i)
            for i in range(20)
        ])
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        result = optimize_retry_policy(
            str(path),
            sla_total_ms=300.0,
            max_retries_range=[1, 2],
        )
        assert isinstance(result, dict)


class TestRetryOptimizerConfig:
    def test_defaults(self):
        cfg = RetryOptimizerConfig(sla_total_ms=100.0)
        assert cfg.max_amplification == 2.0
        assert cfg.max_retries_range == [1, 2, 3, 5]
        assert len(cfg.threshold_percentiles) == 3
        assert len(cfg.backoff_types) == 2
        assert len(cfg.backoff_ms_values) == 3

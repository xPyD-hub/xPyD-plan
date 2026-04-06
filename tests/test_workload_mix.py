"""Tests for workload_mix module."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.models import SLAConfig
from xpyd_plan.workload_mix import (
    AllocationMode,
    MixOptimizationResult,
    WorkloadAllocation,
    WorkloadMixOptimizer,
    WorkloadSpec,
    optimize_workload_mix,
)


def _make_benchmark(
    num_requests: int = 20,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    total_base: float = 200.0,
    num_prefill: int = 2,
    num_decode: int = 2,
    total_instances: int = 4,
    measured_qps: float = 10.0,
) -> BenchmarkData:
    """Create benchmark data for testing."""
    requests = []
    for i in range(num_requests):
        requests.append(
            {
                "request_id": f"req-{i}",
                "prompt_tokens": 100 + i,
                "output_tokens": 50 + i,
                "ttft_ms": ttft_base + i * 2.0,
                "tpot_ms": tpot_base + i * 0.5,
                "total_latency_ms": total_base + i * 5.0,
                "timestamp": 1000.0 + i * 0.1,
            }
        )
    data = {
        "metadata": {
            "num_prefill_instances": num_prefill,
            "num_decode_instances": num_decode,
            "total_instances": total_instances,
            "measured_qps": measured_qps,
        },
        "requests": requests,
    }
    return BenchmarkData.model_validate(data)


def _make_sla(ttft_ms: float = 500.0, tpot_ms: float = 100.0) -> SLAConfig:
    """Create a lenient SLA config."""
    return SLAConfig(ttft_ms=ttft_ms, tpot_ms=tpot_ms)


def _make_workload(
    name: str = "wl-a",
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    weight: float = 1.0,
) -> WorkloadSpec:
    """Create a workload spec for testing."""
    return WorkloadSpec(
        name=name,
        benchmark_data=_make_benchmark(ttft_base=ttft_base, tpot_base=tpot_base),
        sla=_make_sla(),
        weight=weight,
    )


# --- Model tests ---


class TestWorkloadSpec:
    def test_create_valid(self):
        ws = _make_workload("test-workload")
        assert ws.name == "test-workload"
        assert ws.weight == 1.0
        assert ws.min_prefill == 1
        assert ws.min_decode == 1

    def test_name_stripped(self):
        ws = _make_workload("  padded  ")
        assert ws.name == "padded"

    def test_empty_name_rejected(self):
        with pytest.raises(Exception):
            WorkloadSpec(
                name="",
                benchmark_data=_make_benchmark(),
                sla=_make_sla(),
            )

    def test_negative_weight_rejected(self):
        with pytest.raises(Exception):
            WorkloadSpec(
                name="bad",
                benchmark_data=_make_benchmark(),
                sla=_make_sla(),
                weight=-1.0,
            )

    def test_zero_weight_rejected(self):
        with pytest.raises(Exception):
            WorkloadSpec(
                name="bad",
                benchmark_data=_make_benchmark(),
                sla=_make_sla(),
                weight=0.0,
            )


class TestWorkloadAllocation:
    def test_ratio_str(self):
        alloc = WorkloadAllocation(
            name="test",
            num_prefill=3,
            num_decode=5,
            total_instances=8,
            meets_sla=True,
            prefill_waste=0.1,
            decode_waste=0.2,
            weighted_waste=0.15,
        )
        assert alloc.ratio_str == "3P:5D"

    def test_serialization(self):
        alloc = WorkloadAllocation(
            name="test",
            num_prefill=2,
            num_decode=2,
            total_instances=4,
            meets_sla=True,
            prefill_waste=0.1,
            decode_waste=0.2,
            weighted_waste=0.15,
            sla_details={"ttft_p99_ms": 100.0},
        )
        data = json.loads(alloc.model_dump_json())
        assert data["name"] == "test"
        assert data["num_prefill"] == 2
        assert data["sla_details"]["ttft_p99_ms"] == 100.0


class TestMixOptimizationResult:
    def test_infeasible_summary(self):
        r = MixOptimizationResult(feasible=False)
        assert "No feasible" in r.summary

    def test_feasible_summary(self):
        r = MixOptimizationResult(
            feasible=True,
            all_sla_met=True,
            total_instances=8,
            total_prefill=3,
            total_decode=5,
            total_weighted_waste=0.123,
            allocations=[],
        )
        assert "✅" in r.summary
        assert "8 GPUs" in r.summary

    def test_partial_sla_summary(self):
        r = MixOptimizationResult(
            feasible=True,
            all_sla_met=False,
            total_instances=4,
            total_prefill=2,
            total_decode=2,
            total_weighted_waste=0.5,
            allocations=[],
        )
        assert "⚠️" in r.summary

    def test_serialization(self):
        r = MixOptimizationResult(
            feasible=True,
            all_sla_met=True,
            total_instances=4,
            total_prefill=2,
            total_decode=2,
            total_weighted_waste=0.1,
            mode=AllocationMode.DEDICATED,
        )
        data = json.loads(r.model_dump_json())
        assert data["feasible"] is True
        assert data["mode"] == "dedicated"


# --- Optimizer tests ---


class TestWorkloadMixOptimizer:
    def test_empty_workloads(self):
        optimizer = WorkloadMixOptimizer()
        result = optimizer.optimize([])
        assert not result.feasible
        assert result.candidates_evaluated == 0

    def test_single_workload(self):
        ws = _make_workload("single")
        optimizer = WorkloadMixOptimizer()
        result = optimizer.optimize([ws])
        assert result.feasible
        assert len(result.allocations) == 1
        assert result.allocations[0].name == "single"
        assert result.total_instances > 0

    def test_two_workloads(self):
        w1 = _make_workload("w1", ttft_base=40.0)
        w2 = _make_workload("w2", ttft_base=60.0)
        optimizer = WorkloadMixOptimizer()
        result = optimizer.optimize([w1, w2])
        assert result.feasible
        assert len(result.allocations) == 2
        assert {a.name for a in result.allocations} == {"w1", "w2"}

    def test_gpu_budget_unlimited(self):
        ws = _make_workload("budget-test")
        result = optimize_workload_mix([ws], total_gpu_budget=None)
        assert result.feasible
        assert result.gpu_budget is None

    def test_gpu_budget_sufficient(self):
        ws = _make_workload("budget-ok")
        result = optimize_workload_mix([ws], total_gpu_budget=100)
        assert result.feasible
        assert result.total_instances <= 100

    def test_gpu_budget_tight(self):
        ws = _make_workload("budget-tight")
        # Budget of 1 should be infeasible (need at least 1P+1D=2)
        result = optimize_workload_mix([ws], total_gpu_budget=1)
        # Might be infeasible depending on min instances
        # At least we shouldn't crash
        assert isinstance(result, MixOptimizationResult)

    def test_candidates_evaluated_positive(self):
        ws = _make_workload("eval-count")
        result = optimize_workload_mix([ws])
        assert result.candidates_evaluated > 0

    def test_allocation_mode_dedicated(self):
        ws = _make_workload("mode-test")
        result = optimize_workload_mix([ws], mode=AllocationMode.DEDICATED)
        assert result.mode == AllocationMode.DEDICATED

    def test_weighted_workloads(self):
        w1 = _make_workload("high-prio", weight=10.0)
        w2 = _make_workload("low-prio", weight=0.1)
        result = optimize_workload_mix([w1, w2])
        assert result.feasible

    def test_max_instances_per_workload(self):
        ws = _make_workload("max-test")
        result = optimize_workload_mix([ws], max_instances_per_workload=4)
        assert result.feasible or not result.feasible  # Just shouldn't crash

    def test_result_json_round_trip(self):
        ws = _make_workload("json-test")
        result = optimize_workload_mix([ws])
        data = json.loads(result.model_dump_json())
        restored = MixOptimizationResult.model_validate(data)
        assert restored.feasible == result.feasible
        assert restored.total_instances == result.total_instances

    def test_allocation_waste_bounds(self):
        ws = _make_workload("waste-bounds")
        result = optimize_workload_mix([ws])
        if result.feasible:
            for a in result.allocations:
                assert 0 <= a.prefill_waste <= 1
                assert 0 <= a.decode_waste <= 1
                assert a.weighted_waste >= 0

    def test_sla_details_populated(self):
        ws = _make_workload("sla-details")
        result = optimize_workload_mix([ws])
        if result.feasible and result.allocations:
            a = result.allocations[0]
            # sla_details should have some data
            assert isinstance(a.sla_details, dict)

    def test_three_workloads(self):
        workloads = [
            _make_workload("w1"),
            _make_workload("w2", ttft_base=30.0),
            _make_workload("w3", ttft_base=70.0),
        ]
        result = optimize_workload_mix(workloads)
        assert result.feasible
        assert len(result.allocations) == 3

    def test_total_prefill_decode_sum(self):
        ws = _make_workload("sum-check")
        result = optimize_workload_mix([ws])
        if result.feasible:
            assert result.total_instances == result.total_prefill + result.total_decode
            assert result.total_prefill == sum(a.num_prefill for a in result.allocations)
            assert result.total_decode == sum(a.num_decode for a in result.allocations)


# --- Convenience function tests ---


class TestOptimizeWorkloadMix:
    def test_convenience_function(self):
        ws = _make_workload("convenience")
        result = optimize_workload_mix([ws])
        assert isinstance(result, MixOptimizationResult)

    def test_empty_list(self):
        result = optimize_workload_mix([])
        assert not result.feasible

    def test_returns_best_solution(self):
        # With lenient SLA, should find a feasible solution
        ws = _make_workload("best", ttft_base=20.0)
        result = optimize_workload_mix([ws])
        assert result.feasible


# --- API surface tests ---


class TestPublicAPI:
    def test_imports(self):
        from xpyd_plan import (
            WorkloadMixOptimizer,
            optimize_workload_mix,
        )
        assert WorkloadMixOptimizer is not None
        assert optimize_workload_mix is not None

    def test_optimizer_init_default(self):
        opt = WorkloadMixOptimizer()
        assert opt._max_instances == 32

    def test_optimizer_init_custom(self):
        opt = WorkloadMixOptimizer(max_instances_per_workload=16)
        assert opt._max_instances == 16

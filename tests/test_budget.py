"""Tests for the SLA budget allocation module (M27)."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.budget import (
    AllocationStrategy,
    BudgetAllocation,
    BudgetAllocator,
    StageBudget,
    allocate_budget,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_requests(
    n: int = 100, ttft: float = 30.0, tpot: float = 10.0
) -> list[dict]:
    """Generate benchmark request dicts with known latencies."""
    return [
        {
            "request_id": f"req-{i}",
            "prompt_tokens": 128,
            "output_tokens": 64,
            "ttft_ms": ttft + (i % 10) * 0.5,
            "tpot_ms": tpot + (i % 10) * 0.2,
            "total_latency_ms": ttft + tpot * 64 + (i % 10) * 2.0,
            "timestamp": 1700000000.0 + i,
        }
        for i in range(n)
    ]


def _make_benchmark(
    n: int = 100, ttft: float = 30.0, tpot: float = 10.0
) -> BenchmarkData:
    """Create a BenchmarkData instance."""
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=6,
            total_instances=8,
            measured_qps=10.0,
        ),
        requests=[BenchmarkRequest(**r) for r in _make_requests(n, ttft, tpot)],
    )


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestModels:
    """Test Pydantic models."""

    def test_stage_budget_creation(self):
        sb = StageBudget(
            stage="ttft",
            budget_ms=100.0,
            share=0.6,
            observed_ms=80.0,
            headroom_ms=20.0,
        )
        assert sb.stage == "ttft"
        assert sb.budget_ms == 100.0
        assert sb.headroom_ms == 20.0

    def test_allocation_strategy_values(self):
        assert AllocationStrategy.PROPORTIONAL == "proportional"
        assert AllocationStrategy.BALANCED == "balanced"
        assert AllocationStrategy.TTFT_PRIORITY == "ttft-priority"
        assert AllocationStrategy.TPOT_PRIORITY == "tpot-priority"

    def test_budget_allocation_serialization(self):
        alloc = BudgetAllocation(
            total_budget_ms=100.0,
            strategy=AllocationStrategy.PROPORTIONAL,
            percentile=95.0,
            ttft=StageBudget(
                stage="ttft", budget_ms=60.0, share=0.6,
                observed_ms=50.0, headroom_ms=10.0,
            ),
            tpot=StageBudget(
                stage="tpot", budget_ms=40.0, share=0.4,
                observed_ms=30.0, headroom_ms=10.0,
            ),
            observed_ratio=0.625,
            feasible=True,
        )
        d = alloc.model_dump()
        assert d["strategy"] == "proportional"
        assert d["feasible"] is True


# ---------------------------------------------------------------------------
# Allocator Tests
# ---------------------------------------------------------------------------


class TestBudgetAllocator:
    """Test the BudgetAllocator class."""

    def test_proportional_strategy(self):
        bench = _make_benchmark(ttft=30.0, tpot=10.0)
        allocator = BudgetAllocator(bench)
        result = allocator.allocate(100.0, AllocationStrategy.PROPORTIONAL)
        # TTFT ~30, TPOT ~10, ratio ~0.75
        assert result.ttft.share > 0.5
        assert result.tpot.share < 0.5
        assert abs(result.ttft.share + result.tpot.share - 1.0) < 1e-6

    def test_balanced_strategy(self):
        bench = _make_benchmark(ttft=30.0, tpot=10.0)
        allocator = BudgetAllocator(bench)
        result = allocator.allocate(100.0, AllocationStrategy.BALANCED)
        assert result.ttft.share == 0.5
        assert result.tpot.share == 0.5
        assert result.ttft.budget_ms == 50.0
        assert result.tpot.budget_ms == 50.0

    def test_ttft_priority_strategy(self):
        bench = _make_benchmark()
        allocator = BudgetAllocator(bench)
        result = allocator.allocate(100.0, AllocationStrategy.TTFT_PRIORITY)
        assert result.ttft.share == 0.7
        assert result.tpot.share == 0.3

    def test_tpot_priority_strategy(self):
        bench = _make_benchmark()
        allocator = BudgetAllocator(bench)
        result = allocator.allocate(100.0, AllocationStrategy.TPOT_PRIORITY)
        assert result.ttft.share == 0.3
        assert result.tpot.share == 0.7

    def test_feasibility_true(self):
        bench = _make_benchmark(ttft=20.0, tpot=10.0)
        allocator = BudgetAllocator(bench)
        result = allocator.allocate(200.0, AllocationStrategy.PROPORTIONAL)
        assert result.feasible is True
        assert result.ttft.headroom_ms > 0
        assert result.tpot.headroom_ms > 0

    def test_feasibility_false_tight_budget(self):
        bench = _make_benchmark(ttft=40.0, tpot=40.0)
        allocator = BudgetAllocator(bench)
        # Budget way too small for observed latencies
        result = allocator.allocate(10.0, AllocationStrategy.BALANCED)
        assert result.feasible is False

    def test_percentile_affects_result(self):
        bench = _make_benchmark(n=200, ttft=30.0, tpot=10.0)
        allocator = BudgetAllocator(bench)
        r50 = allocator.allocate(100.0, percentile=50.0)
        r99 = allocator.allocate(100.0, percentile=99.0)
        # P99 observed should be >= P50 observed
        assert r99.ttft.observed_ms >= r50.ttft.observed_ms

    def test_total_budget_preserved(self):
        bench = _make_benchmark()
        allocator = BudgetAllocator(bench)
        result = allocator.allocate(500.0, AllocationStrategy.PROPORTIONAL)
        total = result.ttft.budget_ms + result.tpot.budget_ms
        assert abs(total - 500.0) < 0.1

    def test_observed_ratio_range(self):
        bench = _make_benchmark(ttft=50.0, tpot=50.0)
        allocator = BudgetAllocator(bench)
        result = allocator.allocate(200.0, AllocationStrategy.PROPORTIONAL)
        assert 0.0 <= result.observed_ratio <= 1.0

    def test_headroom_calculation(self):
        bench = _make_benchmark(ttft=20.0, tpot=10.0)
        allocator = BudgetAllocator(bench)
        result = allocator.allocate(100.0, AllocationStrategy.BALANCED)
        assert result.ttft.headroom_ms == pytest.approx(
            result.ttft.budget_ms - result.ttft.observed_ms, abs=0.1
        )
        assert result.tpot.headroom_ms == pytest.approx(
            result.tpot.budget_ms - result.tpot.observed_ms, abs=0.1
        )

    def test_small_dataset(self):
        """Allocator works with minimal data."""
        bench = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=1.0,
            ),
            requests=[
                BenchmarkRequest(
                    request_id="r1",
                    prompt_tokens=10,
                    output_tokens=10,
                    ttft_ms=20.0,
                    tpot_ms=5.0,
                    total_latency_ms=70.0,
                    timestamp=1700000000.0,
                )
            ],
        )
        allocator = BudgetAllocator(bench)
        result = allocator.allocate(50.0)
        assert result.total_budget_ms == 50.0

    def test_equal_latencies(self):
        bench = _make_benchmark(ttft=25.0, tpot=25.0)
        allocator = BudgetAllocator(bench)
        result = allocator.allocate(100.0, AllocationStrategy.PROPORTIONAL)
        # Should be close to 50/50
        assert abs(result.ttft.share - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Programmatic API Tests
# ---------------------------------------------------------------------------


class TestAllocateBudgetAPI:
    """Test the allocate_budget() function."""

    def test_returns_dict(self):
        bench = _make_benchmark()
        result = allocate_budget(bench, 100.0)
        assert isinstance(result, dict)
        assert "ttft" in result
        assert "tpot" in result
        assert "feasible" in result

    def test_strategy_string(self):
        bench = _make_benchmark()
        result = allocate_budget(bench, 100.0, strategy="balanced")
        assert result["strategy"] == "balanced"
        assert result["ttft"]["share"] == 0.5

    def test_all_strategies(self):
        bench = _make_benchmark()
        for s in ["proportional", "balanced", "ttft-priority", "tpot-priority"]:
            result = allocate_budget(bench, 100.0, strategy=s)
            assert result["strategy"] == s

    def test_custom_percentile(self):
        bench = _make_benchmark()
        result = allocate_budget(bench, 100.0, percentile=99.0)
        assert result["percentile"] == 99.0

    def test_invalid_strategy_raises(self):
        bench = _make_benchmark()
        with pytest.raises(ValueError):
            allocate_budget(bench, 100.0, strategy="invalid")

    def test_json_serializable(self):
        bench = _make_benchmark()
        result = allocate_budget(bench, 200.0, strategy="proportional")
        serialized = json.dumps(result)
        assert "ttft" in serialized


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------


class TestBudgetCLI:
    """Test the budget CLI subcommand."""

    def test_budget_table_output(self, tmp_path):
        bench = _make_benchmark()
        bench_file = tmp_path / "bench.json"
        bench_file.write_text(bench.model_dump_json())


        from xpyd_plan.cli import main

        with pytest.raises(SystemExit) as exc:
            main([
                "budget",
                "--benchmark", str(bench_file),
                "--total-budget-ms", "100",
            ])
        # Should exit 0 on success
        assert exc.value.code is None or exc.value.code == 0

    def test_budget_json_output(self, tmp_path):
        bench = _make_benchmark()
        bench_file = tmp_path / "bench.json"
        bench_file.write_text(bench.model_dump_json())

        from xpyd_plan.cli import main

        with pytest.raises(SystemExit) as exc:
            main([
                "budget",
                "--benchmark", str(bench_file),
                "--total-budget-ms", "100",
                "--strategy", "balanced",
                "--output-format", "json",
            ])
        assert exc.value.code is None or exc.value.code == 0

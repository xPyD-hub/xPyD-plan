"""Tests for benchmark plan generator module."""

from __future__ import annotations

import pytest

from xpyd_plan.benchmark_models import AnalysisResult, RatioCandidate, SLACheck
from xpyd_plan.plan_generator import (
    BenchmarkPlan,
    BenchmarkPlanGenerator,
    PlannedRatio,
    RatioPriority,
    generate_benchmark_plan,
)


def _make_sla_check() -> SLACheck:
    return SLACheck(
        ttft_p95_ms=10.0,
        ttft_p99_ms=15.0,
        tpot_p95_ms=5.0,
        tpot_p99_ms=8.0,
        total_latency_p95_ms=100.0,
        total_latency_p99_ms=150.0,
        meets_ttft=True,
        meets_tpot=True,
        meets_total_latency=True,
        meets_all=True,
    )


def _make_candidate(p: int, d: int, meets: bool = True) -> RatioCandidate:
    return RatioCandidate(
        num_prefill=p,
        num_decode=d,
        prefill_utilization=0.7,
        decode_utilization=0.8,
        waste_rate=0.1,
        meets_sla=meets,
        sla_check=_make_sla_check(),
    )


class TestBenchmarkPlanGenerator:
    """Tests for BenchmarkPlanGenerator."""

    def test_minimum_instances(self) -> None:
        gen = BenchmarkPlanGenerator(total_instances=2)
        plan = gen.generate()
        assert plan.total_instances == 2
        assert len(plan.ratios) == 1  # Only 1P:1D possible
        assert plan.ratios[0].num_prefill == 1
        assert plan.ratios[0].num_decode == 1

    def test_all_ratios_enumerated(self) -> None:
        gen = BenchmarkPlanGenerator(total_instances=6)
        plan = gen.generate()
        # 5 possible ratios: 1:5, 2:4, 3:3, 4:2, 5:1
        assert len(plan.ratios) == 5
        totals = {r.num_prefill + r.num_decode for r in plan.ratios}
        assert totals == {6}

    def test_invalid_total_instances(self) -> None:
        with pytest.raises(ValueError, match="total_instances must be >= 2"):
            BenchmarkPlanGenerator(total_instances=1)

    def test_invalid_max_runs(self) -> None:
        with pytest.raises(ValueError, match="max_runs must be >= 1"):
            BenchmarkPlanGenerator(total_instances=4, max_runs=0)

    def test_max_runs_caps_output(self) -> None:
        gen = BenchmarkPlanGenerator(total_instances=10, max_runs=3)
        plan = gen.generate()
        assert len(plan.ratios) == 3

    def test_balanced_is_first(self) -> None:
        gen = BenchmarkPlanGenerator(total_instances=8)
        plan = gen.generate()
        assert plan.ratios[0].num_prefill == 4
        assert plan.ratios[0].num_decode == 4
        assert plan.ratios[0].priority == RatioPriority.CRITICAL

    def test_boundaries_included(self) -> None:
        gen = BenchmarkPlanGenerator(total_instances=6)
        plan = gen.generate()
        ratios = {(r.num_prefill, r.num_decode) for r in plan.ratios}
        assert (1, 5) in ratios
        assert (5, 1) in ratios

    def test_no_duplicates(self) -> None:
        gen = BenchmarkPlanGenerator(total_instances=8)
        plan = gen.generate()
        pairs = [(r.num_prefill, r.num_decode) for r in plan.ratios]
        assert len(pairs) == len(set(pairs))

    def test_priority_ordering(self) -> None:
        gen = BenchmarkPlanGenerator(total_instances=10)
        plan = gen.generate()
        # Critical should come before Medium
        critical_idx = [
            i for i, r in enumerate(plan.ratios)
            if r.priority == RatioPriority.CRITICAL
        ]
        medium_idx = [
            i for i, r in enumerate(plan.ratios)
            if r.priority == RatioPriority.MEDIUM
        ]
        if critical_idx and medium_idx:
            assert max(critical_idx) < min(medium_idx)

    def test_refined_plan_with_existing_result(self) -> None:
        best = _make_candidate(3, 5)
        result = AnalysisResult(
            best=best,
            candidates=[best, _make_candidate(4, 4)],
            total_instances=8,
        )
        gen = BenchmarkPlanGenerator(total_instances=8, existing_result=result)
        plan = gen.generate()
        assert plan.refined is True
        # Neighbors of 3P:5D should be critical
        critical = [r for r in plan.ratios if r.priority == RatioPriority.CRITICAL]
        critical_pairs = {(r.num_prefill, r.num_decode) for r in critical}
        assert (2, 6) in critical_pairs  # 3-1=2
        assert (4, 4) in critical_pairs or len(critical) > 0

    def test_refined_excludes_already_benchmarked(self) -> None:
        best = _make_candidate(3, 5)
        already = _make_candidate(4, 4)
        result = AnalysisResult(
            best=best,
            candidates=[best, already],
            total_instances=8,
        )
        gen = BenchmarkPlanGenerator(total_instances=8, existing_result=result)
        plan = gen.generate()
        medium_pairs = {
            (r.num_prefill, r.num_decode)
            for r in plan.ratios
            if r.priority == RatioPriority.MEDIUM
        }
        # 3:5 and 4:4 are already benchmarked, should not appear in medium
        assert (3, 5) not in medium_pairs
        assert (4, 4) not in medium_pairs

    def test_plan_model_fields(self) -> None:
        gen = BenchmarkPlanGenerator(total_instances=4)
        plan = gen.generate()
        assert isinstance(plan, BenchmarkPlan)
        assert plan.total_instances == 4
        assert plan.refined is False
        for r in plan.ratios:
            assert isinstance(r, PlannedRatio)
            assert r.num_prefill >= 1
            assert r.num_decode >= 1

    def test_ratio_str_property(self) -> None:
        r = PlannedRatio(num_prefill=3, num_decode=5, priority=RatioPriority.HIGH, reason="test")
        assert r.ratio_str == "3P:5D"
        assert r.total == 8

    def test_three_instances(self) -> None:
        gen = BenchmarkPlanGenerator(total_instances=3)
        plan = gen.generate()
        # 2 possible ratios: 1:2, 2:1
        assert len(plan.ratios) == 2

    def test_large_instance_count(self) -> None:
        gen = BenchmarkPlanGenerator(total_instances=20, max_runs=5)
        plan = gen.generate()
        assert len(plan.ratios) == 5
        # First should be balanced (10:10)
        assert plan.ratios[0].num_prefill == 10

    def test_programmatic_api(self) -> None:
        result = generate_benchmark_plan(total_instances=6, max_runs=3)
        assert isinstance(result, dict)
        assert result["total_instances"] == 6
        assert len(result["ratios"]) == 3
        assert result["refined"] is False

    def test_programmatic_api_with_existing(self) -> None:
        best = _make_candidate(2, 4)
        analysis = AnalysisResult(
            best=best,
            candidates=[best],
            total_instances=6,
        )
        result = generate_benchmark_plan(total_instances=6, existing_result=analysis)
        assert result["refined"] is True

    def test_max_runs_one(self) -> None:
        gen = BenchmarkPlanGenerator(total_instances=10, max_runs=1)
        plan = gen.generate()
        assert len(plan.ratios) == 1
        assert plan.ratios[0].priority == RatioPriority.CRITICAL

    def test_even_vs_odd_total(self) -> None:
        # Even: balanced is exactly half
        gen_even = BenchmarkPlanGenerator(total_instances=8)
        plan_even = gen_even.generate()
        assert plan_even.ratios[0].num_prefill == 4
        assert plan_even.ratios[0].num_decode == 4

        # Odd: balanced is floor(N/2)
        gen_odd = BenchmarkPlanGenerator(total_instances=7)
        plan_odd = gen_odd.generate()
        assert plan_odd.ratios[0].num_prefill == 3
        assert plan_odd.ratios[0].num_decode == 4

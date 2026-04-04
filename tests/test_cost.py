"""Tests for cost-aware optimization."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from xpyd_plan.benchmark_models import (
    AnalysisResult,
    RatioCandidate,
    SLACheck,
)
from xpyd_plan.cost import CostAnalyzer, CostConfig, CostResult

# --- Fixtures ---


def _make_sla_check(meets: bool = True) -> SLACheck:
    return SLACheck(
        ttft_p95_ms=10.0,
        ttft_p99_ms=15.0,
        tpot_p95_ms=5.0,
        tpot_p99_ms=8.0,
        total_latency_p95_ms=100.0,
        total_latency_p99_ms=120.0,
        meets_ttft=meets,
        meets_tpot=meets,
        meets_total_latency=meets,
        meets_all=meets,
    )


def _make_candidate(p: int, d: int, meets_sla: bool = True, waste: float = 0.1) -> RatioCandidate:
    return RatioCandidate(
        num_prefill=p,
        num_decode=d,
        prefill_utilization=0.8,
        decode_utilization=0.7,
        waste_rate=waste,
        meets_sla=meets_sla,
        sla_check=_make_sla_check(meets_sla),
    )


def _make_analysis(
    candidates: list[RatioCandidate], best: RatioCandidate | None = None
) -> AnalysisResult:
    return AnalysisResult(
        best=best,
        candidates=candidates,
        total_instances=candidates[0].total if candidates else 8,
    )


@pytest.fixture
def cost_config() -> CostConfig:
    return CostConfig(gpu_hourly_rate=2.50, currency="USD")


@pytest.fixture
def analyzer(cost_config: CostConfig) -> CostAnalyzer:
    return CostAnalyzer(cost_config)


# --- CostConfig tests ---


class TestCostConfig:
    def test_basic_creation(self) -> None:
        cfg = CostConfig(gpu_hourly_rate=3.0)
        assert cfg.gpu_hourly_rate == 3.0
        assert cfg.currency == "USD"

    def test_custom_currency(self) -> None:
        cfg = CostConfig(gpu_hourly_rate=1.5, currency="EUR")
        assert cfg.currency == "EUR"

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "cost.yaml"
        yaml_path.write_text(yaml.dump({"gpu_hourly_rate": 4.0, "currency": "CNY"}))
        cfg = CostConfig.from_yaml(str(yaml_path))
        assert cfg.gpu_hourly_rate == 4.0
        assert cfg.currency == "CNY"

    def test_invalid_rate(self) -> None:
        with pytest.raises(Exception):
            CostConfig(gpu_hourly_rate=-1.0)

    def test_zero_rate(self) -> None:
        with pytest.raises(Exception):
            CostConfig(gpu_hourly_rate=0.0)


# --- CostResult tests ---


class TestCostResult:
    def test_ratio_str(self) -> None:
        r = CostResult(
            num_prefill=2, num_decode=6, total_instances=8,
            hourly_cost=20.0, currency="USD",
        )
        assert r.ratio_str == "2P:6D"

    def test_cost_per_request_none(self) -> None:
        r = CostResult(
            num_prefill=2, num_decode=6, total_instances=8,
            hourly_cost=20.0,
        )
        assert r.cost_per_request is None


# --- CostAnalyzer.compute_cost tests ---


class TestComputeCost:
    def test_basic_cost(self, analyzer: CostAnalyzer) -> None:
        c = _make_candidate(2, 6)
        result = analyzer.compute_cost(c)
        assert result.hourly_cost == pytest.approx(20.0)  # 8 * 2.50
        assert result.total_instances == 8
        assert result.cost_per_request is None

    def test_cost_with_qps(self, analyzer: CostAnalyzer) -> None:
        c = _make_candidate(2, 6)
        result = analyzer.compute_cost(c, measured_qps=10.0)
        assert result.hourly_cost == pytest.approx(20.0)
        expected_cpr = 20.0 / (10.0 * 3600)
        assert result.cost_per_request == pytest.approx(expected_cpr)

    def test_cost_preserves_sla(self, analyzer: CostAnalyzer) -> None:
        c = _make_candidate(3, 5, meets_sla=False)
        result = analyzer.compute_cost(c)
        assert result.meets_sla is False

    def test_different_instance_counts(self, analyzer: CostAnalyzer) -> None:
        c1 = _make_candidate(1, 3)  # 4 total
        c2 = _make_candidate(4, 4)  # 8 total
        r1 = analyzer.compute_cost(c1)
        r2 = analyzer.compute_cost(c2)
        assert r1.hourly_cost < r2.hourly_cost


# --- CostAnalyzer.compute_all_costs tests ---


class TestComputeAllCosts:
    def test_sorted_by_cost(self, analyzer: CostAnalyzer) -> None:
        candidates = [_make_candidate(4, 4), _make_candidate(2, 2), _make_candidate(3, 5)]
        analysis = _make_analysis(candidates, best=candidates[0])
        costs = analyzer.compute_all_costs(analysis)
        assert len(costs) == 3
        assert costs[0].total_instances <= costs[1].total_instances

    def test_budget_filter(self, analyzer: CostAnalyzer) -> None:
        candidates = [_make_candidate(1, 1), _make_candidate(4, 4)]
        analysis = _make_analysis(candidates)
        costs = analyzer.compute_all_costs(analysis, budget_ceiling=10.0)
        # 2 * 2.50 = 5.0 (in), 8 * 2.50 = 20.0 (out)
        assert len(costs) == 1
        assert costs[0].total_instances == 2

    def test_empty_after_filter(self, analyzer: CostAnalyzer) -> None:
        candidates = [_make_candidate(4, 4)]
        analysis = _make_analysis(candidates)
        costs = analyzer.compute_all_costs(analysis, budget_ceiling=1.0)
        assert len(costs) == 0


# --- CostAnalyzer.find_cost_optimal tests ---


class TestFindCostOptimal:
    def test_finds_cheapest_meeting_sla(self, analyzer: CostAnalyzer) -> None:
        candidates = [
            _make_candidate(1, 1, meets_sla=True),
            _make_candidate(4, 4, meets_sla=True),
        ]
        analysis = _make_analysis(candidates, best=candidates[1])
        result = analyzer.find_cost_optimal(analysis)
        assert result is not None
        assert result.total_instances == 2

    def test_skips_non_sla(self, analyzer: CostAnalyzer) -> None:
        candidates = [
            _make_candidate(1, 1, meets_sla=False),
            _make_candidate(4, 4, meets_sla=True),
        ]
        analysis = _make_analysis(candidates)
        result = analyzer.find_cost_optimal(analysis)
        assert result is not None
        assert result.total_instances == 8

    def test_none_when_all_fail(self, analyzer: CostAnalyzer) -> None:
        candidates = [_make_candidate(2, 2, meets_sla=False)]
        analysis = _make_analysis(candidates)
        result = analyzer.find_cost_optimal(analysis)
        assert result is None

    def test_budget_ceiling(self, analyzer: CostAnalyzer) -> None:
        candidates = [
            _make_candidate(1, 1, meets_sla=True),
            _make_candidate(2, 2, meets_sla=True),
        ]
        analysis = _make_analysis(candidates)
        # ceiling=8.0 → 2*2.50=5.0 in, 4*2.50=10.0 out
        result = analyzer.find_cost_optimal(analysis, budget_ceiling=8.0)
        assert result is not None
        assert result.total_instances == 2


# --- CostAnalyzer.compare tests ---


class TestCompare:
    def test_comparison_result(self, analyzer: CostAnalyzer) -> None:
        candidates = [
            _make_candidate(1, 1, meets_sla=True, waste=0.2),
            _make_candidate(3, 5, meets_sla=True, waste=0.05),
        ]
        analysis = _make_analysis(candidates, best=candidates[1])
        cmp = analyzer.compare(analysis, measured_qps=10.0)
        assert cmp.sla_optimal is not None
        assert cmp.cost_optimal is not None
        assert cmp.cost_optimal.total_instances == 2  # cheapest
        assert cmp.sla_optimal.total_instances == 8  # best waste

    def test_same_optimal(self, analyzer: CostAnalyzer) -> None:
        candidates = [_make_candidate(2, 2, meets_sla=True)]
        analysis = _make_analysis(candidates, best=candidates[0])
        cmp = analyzer.compare(analysis)
        assert cmp.sla_optimal is not None
        assert cmp.cost_optimal is not None
        assert cmp.sla_optimal.total_instances == cmp.cost_optimal.total_instances

    def test_no_sla_optimal(self, analyzer: CostAnalyzer) -> None:
        candidates = [_make_candidate(2, 2, meets_sla=False)]
        analysis = _make_analysis(candidates, best=None)
        cmp = analyzer.compare(analysis)
        assert cmp.sla_optimal is None
        assert cmp.cost_optimal is None

    def test_budget_ceiling_in_comparison(self, analyzer: CostAnalyzer) -> None:
        candidates = [
            _make_candidate(1, 1, meets_sla=True),
            _make_candidate(4, 4, meets_sla=True),
        ]
        analysis = _make_analysis(candidates, best=candidates[1])
        cmp = analyzer.compare(analysis, budget_ceiling=6.0)
        assert cmp.budget_ceiling == 6.0
        assert len(cmp.all_costs) == 1  # only 2-instance fits

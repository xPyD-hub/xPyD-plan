"""Tests for Pareto frontier analysis."""

from __future__ import annotations

from xpyd_plan.benchmark_models import (
    AnalysisResult,
    RatioCandidate,
    SLACheck,
)
from xpyd_plan.cost import CostConfig
from xpyd_plan.pareto import (
    ParetoAnalyzer,
    ParetoCandidate,
    ParetoFrontier,
    ParetoObjective,
    _dominates,
    find_pareto_frontier,
)


def _make_sla_check(
    ttft_p95: float = 50.0,
    tpot_p95: float = 20.0,
    total_p95: float = 200.0,
) -> SLACheck:
    return SLACheck(
        ttft_p95_ms=ttft_p95,
        ttft_p99_ms=ttft_p95 * 1.2,
        tpot_p95_ms=tpot_p95,
        tpot_p99_ms=tpot_p95 * 1.2,
        total_latency_p95_ms=total_p95,
        total_latency_p99_ms=total_p95 * 1.2,
        meets_ttft=True,
        meets_tpot=True,
        meets_total_latency=True,
        meets_all=True,
    )


def _make_candidate(
    p: int,
    d: int,
    waste: float,
    meets_sla: bool = True,
    total_p95: float = 200.0,
) -> RatioCandidate:
    return RatioCandidate(
        num_prefill=p,
        num_decode=d,
        prefill_utilization=1 - waste,
        decode_utilization=1 - waste,
        waste_rate=waste,
        meets_sla=meets_sla,
        sla_check=_make_sla_check(total_p95=total_p95),
    )


def _make_analysis(candidates: list[RatioCandidate]) -> AnalysisResult:
    best = None
    sla_meeting = [c for c in candidates if c.meets_sla]
    if sla_meeting:
        best = min(sla_meeting, key=lambda c: c.waste_rate)
    total = candidates[0].total if candidates else 8
    return AnalysisResult(
        best=best,
        candidates=candidates,
        total_instances=total,
    )


class TestDominance:
    """Test Pareto dominance logic."""

    def test_dominates_all_better(self):
        a = ParetoCandidate(num_prefill=2, num_decode=6, latency_ms=100, waste_rate=0.1)
        b = ParetoCandidate(num_prefill=3, num_decode=5, latency_ms=200, waste_rate=0.3)
        assert _dominates(a, b, [ParetoObjective.LATENCY, ParetoObjective.WASTE])

    def test_not_dominates_one_worse(self):
        a = ParetoCandidate(num_prefill=2, num_decode=6, latency_ms=100, waste_rate=0.5)
        b = ParetoCandidate(num_prefill=3, num_decode=5, latency_ms=200, waste_rate=0.1)
        assert not _dominates(a, b, [ParetoObjective.LATENCY, ParetoObjective.WASTE])

    def test_not_dominates_equal(self):
        a = ParetoCandidate(num_prefill=2, num_decode=6, latency_ms=100, waste_rate=0.1)
        b = ParetoCandidate(num_prefill=3, num_decode=5, latency_ms=100, waste_rate=0.1)
        assert not _dominates(a, b, [ParetoObjective.LATENCY, ParetoObjective.WASTE])

    def test_dominates_one_equal_one_better(self):
        a = ParetoCandidate(num_prefill=2, num_decode=6, latency_ms=100, waste_rate=0.1)
        b = ParetoCandidate(num_prefill=3, num_decode=5, latency_ms=100, waste_rate=0.3)
        assert _dominates(a, b, [ParetoObjective.LATENCY, ParetoObjective.WASTE])


class TestParetoAnalyzer:
    """Test ParetoAnalyzer."""

    def test_basic_frontier(self):
        candidates = [
            _make_candidate(2, 6, 0.1, total_p95=100.0),  # low waste, low latency
            _make_candidate(3, 5, 0.3, total_p95=80.0),   # higher waste, lower latency
            _make_candidate(4, 4, 0.5, total_p95=300.0),  # dominated
        ]
        analysis = _make_analysis(candidates)
        analyzer = ParetoAnalyzer()
        result = analyzer.analyze(analysis)

        assert len(result.frontier) >= 1
        assert len(result.dominated) + len(result.frontier) == len(candidates)
        # c3 (4P:4D) should be dominated (worse latency AND worse waste)
        dominated_ratios = {c.ratio_str for c in result.dominated}
        assert "4P:4D" in dominated_ratios

    def test_empty_candidates(self):
        analysis = AnalysisResult(candidates=[], total_instances=8)
        analyzer = ParetoAnalyzer()
        result = analyzer.analyze(analysis)
        assert result.frontier == []
        assert result.best_weighted is None

    def test_single_candidate(self):
        candidates = [_make_candidate(2, 6, 0.1, total_p95=100.0)]
        analysis = _make_analysis(candidates)
        analyzer = ParetoAnalyzer()
        result = analyzer.analyze(analysis)
        assert len(result.frontier) == 1
        assert len(result.dominated) == 0

    def test_sla_only_filter(self):
        candidates = [
            _make_candidate(2, 6, 0.1, meets_sla=True, total_p95=100.0),
            _make_candidate(3, 5, 0.3, meets_sla=False, total_p95=80.0),
        ]
        analysis = _make_analysis(candidates)
        analyzer = ParetoAnalyzer()
        result = analyzer.analyze(analysis, sla_only=True)
        assert len(result.frontier) == 1
        assert result.frontier[0].ratio_str == "2P:6D"

    def test_sla_only_false_includes_all(self):
        candidates = [
            _make_candidate(2, 6, 0.1, meets_sla=True, total_p95=100.0),
            _make_candidate(3, 5, 0.3, meets_sla=False, total_p95=80.0),
        ]
        analysis = _make_analysis(candidates)
        analyzer = ParetoAnalyzer()
        result = analyzer.analyze(analysis, sla_only=False)
        assert len(result.candidates) == 2

    def test_with_cost_objective(self):
        candidates = [
            _make_candidate(2, 6, 0.1, total_p95=100.0),
            _make_candidate(3, 5, 0.3, total_p95=80.0),
        ]
        analysis = _make_analysis(candidates)
        cost_config = CostConfig(gpu_hourly_rate=2.0, currency="USD")
        analyzer = ParetoAnalyzer(cost_config=cost_config)
        result = analyzer.analyze(
            analysis,
            measured_qps=10.0,
            objectives=[ParetoObjective.LATENCY, ParetoObjective.COST, ParetoObjective.WASTE],
        )
        # Both have same total instances, so cost is equal → latency & waste determine frontier
        assert len(result.candidates) >= 1
        assert "cost" in result.objectives_used

    def test_weighted_scoring(self):
        candidates = [
            _make_candidate(2, 6, 0.1, total_p95=200.0),  # low waste, high latency
            _make_candidate(3, 5, 0.3, total_p95=80.0),   # high waste, low latency
        ]
        analysis = _make_analysis(candidates)
        analyzer = ParetoAnalyzer()

        # Weight latency heavily
        result_latency = analyzer.analyze(
            analysis, weights={"latency": 10.0, "waste": 1.0}
        )
        # Weight waste heavily
        result_waste = analyzer.analyze(
            analysis, weights={"latency": 1.0, "waste": 10.0}
        )

        # Both should be on frontier (they're non-dominated), but best_weighted should differ
        assert result_latency.best_weighted is not None
        assert result_waste.best_weighted is not None
        # With latency heavily weighted, 3P:5D (lower latency) should score better
        assert result_latency.best_weighted.ratio_str == "3P:5D"
        # With waste heavily weighted, 2P:6D (lower waste) should score better
        assert result_waste.best_weighted.ratio_str == "2P:6D"

    def test_custom_objectives(self):
        candidates = [
            _make_candidate(2, 6, 0.1, total_p95=100.0),
            _make_candidate(3, 5, 0.3, total_p95=80.0),
        ]
        analysis = _make_analysis(candidates)
        analyzer = ParetoAnalyzer()
        result = analyzer.analyze(
            analysis, objectives=[ParetoObjective.WASTE]
        )
        assert result.objectives_used == ["waste"]
        # With only waste objective, 2P:6D dominates 3P:5D
        assert len(result.frontier) == 1
        assert result.frontier[0].ratio_str == "2P:6D"

    def test_all_non_dominated(self):
        """When each candidate is best on some objective, none are dominated."""
        candidates = [
            _make_candidate(2, 6, 0.4, total_p95=80.0),   # best latency
            _make_candidate(3, 5, 0.1, total_p95=200.0),  # best waste
        ]
        analysis = _make_analysis(candidates)
        analyzer = ParetoAnalyzer()
        result = analyzer.analyze(analysis)
        assert len(result.frontier) == 2
        assert len(result.dominated) == 0

    def test_all_dominated_except_one(self):
        candidates = [
            _make_candidate(2, 6, 0.1, total_p95=80.0),   # dominates all others
            _make_candidate(3, 5, 0.3, total_p95=200.0),
            _make_candidate(4, 4, 0.5, total_p95=300.0),
        ]
        analysis = _make_analysis(candidates)
        analyzer = ParetoAnalyzer()
        result = analyzer.analyze(analysis)
        assert len(result.frontier) == 1
        assert result.frontier[0].ratio_str == "2P:6D"
        assert len(result.dominated) == 2


class TestParetoCandidate:
    """Test ParetoCandidate model."""

    def test_ratio_str(self):
        c = ParetoCandidate(num_prefill=3, num_decode=5, latency_ms=100, waste_rate=0.2)
        assert c.ratio_str == "3P:5D"

    def test_total_instances(self):
        c = ParetoCandidate(num_prefill=3, num_decode=5, latency_ms=100, waste_rate=0.2)
        assert c.total_instances == 8

    def test_defaults(self):
        c = ParetoCandidate(num_prefill=2, num_decode=6, latency_ms=100, waste_rate=0.1)
        assert c.is_dominated is False
        assert c.weighted_score is None
        assert c.meets_sla is True
        assert c.hourly_cost is None


class TestParetoFrontier:
    """Test ParetoFrontier model."""

    def test_empty_frontier(self):
        f = ParetoFrontier()
        assert f.frontier == []
        assert f.dominated == []
        assert f.best_weighted is None

    def test_serialization(self):
        c = ParetoCandidate(
            num_prefill=2, num_decode=6, latency_ms=100,
            waste_rate=0.1, weighted_score=0.25,
        )
        f = ParetoFrontier(
            candidates=[c], frontier=[c], dominated=[],
            best_weighted=c, objectives_used=["latency", "waste"],
            weights={"latency": 1.0, "waste": 1.0},
        )
        d = f.model_dump()
        assert d["objectives_used"] == ["latency", "waste"]
        assert len(d["frontier"]) == 1


class TestFindParetoFrontierAPI:
    """Test the programmatic API."""

    def test_basic_api(self):
        candidates = [
            _make_candidate(2, 6, 0.1, total_p95=100.0),
            _make_candidate(3, 5, 0.3, total_p95=80.0),
        ]
        analysis = _make_analysis(candidates)
        result = find_pareto_frontier(analysis)
        assert isinstance(result, ParetoFrontier)
        assert len(result.frontier) >= 1

    def test_api_with_cost(self):
        candidates = [
            _make_candidate(2, 6, 0.1, total_p95=100.0),
            _make_candidate(3, 5, 0.3, total_p95=80.0),
        ]
        analysis = _make_analysis(candidates)
        cost_config = CostConfig(gpu_hourly_rate=3.0)
        result = find_pareto_frontier(
            analysis,
            measured_qps=5.0,
            cost_config=cost_config,
            objectives=["latency", "cost", "waste"],
        )
        assert "cost" in result.objectives_used

    def test_api_with_custom_weights(self):
        candidates = [
            _make_candidate(2, 6, 0.1, total_p95=200.0),
            _make_candidate(3, 5, 0.3, total_p95=80.0),
        ]
        analysis = _make_analysis(candidates)
        result = find_pareto_frontier(
            analysis, weights={"latency": 5.0, "waste": 1.0}
        )
        assert result.best_weighted is not None

    def test_api_sla_only(self):
        candidates = [
            _make_candidate(2, 6, 0.1, meets_sla=True, total_p95=100.0),
            _make_candidate(3, 5, 0.3, meets_sla=False, total_p95=80.0),
        ]
        analysis = _make_analysis(candidates)
        result_sla = find_pareto_frontier(analysis, sla_only=True)
        result_all = find_pareto_frontier(analysis, sla_only=False)
        assert len(result_sla.candidates) == 1
        assert len(result_all.candidates) == 2


class TestEdgeCases:
    """Test edge cases."""

    def test_identical_candidates(self):
        """Two identical candidates — neither dominates the other."""
        candidates = [
            _make_candidate(2, 6, 0.1, total_p95=100.0),
            _make_candidate(3, 5, 0.1, total_p95=100.0),
        ]
        analysis = _make_analysis(candidates)
        analyzer = ParetoAnalyzer()
        result = analyzer.analyze(analysis)
        # Neither dominates the other (equal on all objectives)
        assert len(result.frontier) == 2
        assert len(result.dominated) == 0

    def test_three_way_pareto(self):
        """Three candidates forming a proper Pareto frontier."""
        candidates = [
            _make_candidate(1, 7, 0.5, total_p95=50.0),   # best latency, worst waste
            _make_candidate(3, 5, 0.2, total_p95=150.0),   # middle
            _make_candidate(4, 4, 0.05, total_p95=300.0),  # best waste, worst latency
        ]
        analysis = _make_analysis(candidates)
        analyzer = ParetoAnalyzer()
        result = analyzer.analyze(analysis)
        # All three form the frontier (each best on some trade-off)
        assert len(result.frontier) == 3
        assert len(result.dominated) == 0

    def test_cost_none_when_no_cost_config(self):
        candidates = [_make_candidate(2, 6, 0.1, total_p95=100.0)]
        analysis = _make_analysis(candidates)
        analyzer = ParetoAnalyzer()
        result = analyzer.analyze(analysis)
        assert result.frontier[0].hourly_cost is None
        assert "cost" not in result.objectives_used

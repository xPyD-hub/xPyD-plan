"""Tests for the recommendation engine."""

from __future__ import annotations

import json

from xpyd_plan.benchmark_models import (
    AnalysisResult,
    RatioCandidate,
    SLACheck,
    UtilizationResult,
)
from xpyd_plan.cost import CostConfig
from xpyd_plan.recommender import (
    ActionCategory,
    Recommendation,
    RecommendationEngine,
    RecommendationPriority,
    RecommendationReport,
    get_recommendations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sla_check(meets: bool, ttft: float = 50.0, tpot: float = 20.0) -> SLACheck:
    return SLACheck(
        ttft_p95_ms=ttft,
        ttft_p99_ms=ttft * 1.2,
        tpot_p95_ms=tpot,
        tpot_p99_ms=tpot * 1.2,
        total_latency_p95_ms=ttft + tpot * 50,
        total_latency_p99_ms=(ttft + tpot * 50) * 1.2,
        meets_ttft=meets,
        meets_tpot=meets,
        meets_total_latency=meets,
        meets_all=meets,
        evaluated_percentile=95.0,
        ttft_evaluated_ms=ttft,
        tpot_evaluated_ms=tpot,
        total_latency_evaluated_ms=ttft + tpot * 50,
    )


def _candidate(
    p: int, d: int, waste: float, meets_sla: bool = True
) -> RatioCandidate:
    return RatioCandidate(
        num_prefill=p,
        num_decode=d,
        prefill_utilization=1.0 - waste,
        decode_utilization=1.0 - waste,
        waste_rate=waste,
        meets_sla=meets_sla,
        sla_check=_sla_check(meets_sla),
    )


def _util(waste: float) -> UtilizationResult:
    return UtilizationResult(
        prefill_utilization=1.0 - waste,
        decode_utilization=1.0 - waste,
        waste_rate=waste,
    )


def _analysis(
    candidates: list[RatioCandidate],
    best: RatioCandidate | None = None,
    current_config: UtilizationResult | None = None,
    current_sla: SLACheck | None = None,
    total: int = 8,
) -> AnalysisResult:
    return AnalysisResult(
        candidates=candidates,
        best=best,
        total_instances=total,
        current_config=current_config,
        current_sla_check=current_sla,
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_recommendation_priority_order(self):
        r = Recommendation(
            priority=RecommendationPriority.CRITICAL,
            action=ActionCategory.SCALE_UP,
            title="t",
            detail="d",
        )
        assert r.priority_order == 0

    def test_recommendation_priority_low(self):
        r = Recommendation(
            priority=RecommendationPriority.LOW,
            action=ActionCategory.NO_ACTION,
            title="t",
            detail="d",
        )
        assert r.priority_order == 3

    def test_report_defaults(self):
        report = RecommendationReport()
        assert report.recommendations == []
        assert report.total_instances == 0
        assert report.analysis_summary == ""

    def test_action_category_values(self):
        assert ActionCategory.SCALE_UP == "scale_up"
        assert ActionCategory.SCALE_DOWN == "scale_down"
        assert ActionCategory.REBALANCE == "rebalance"
        assert ActionCategory.INVESTIGATE == "investigate"
        assert ActionCategory.NO_ACTION == "no_action"

    def test_priority_values(self):
        assert RecommendationPriority.CRITICAL == "critical"
        assert RecommendationPriority.HIGH == "high"
        assert RecommendationPriority.MEDIUM == "medium"
        assert RecommendationPriority.LOW == "low"


# ---------------------------------------------------------------------------
# Engine: SLA checks
# ---------------------------------------------------------------------------

class TestSLARecommendations:
    def test_no_candidates_meet_sla(self):
        c1 = _candidate(2, 6, 0.3, meets_sla=False)
        c2 = _candidate(4, 4, 0.2, meets_sla=False)
        analysis = _analysis([c1, c2])
        engine = RecommendationEngine()
        report = engine.analyze(analysis)
        assert any(
            r.priority == RecommendationPriority.CRITICAL
            and r.action == ActionCategory.SCALE_UP
            for r in report.recommendations
        )

    def test_current_fails_sla_with_better_option(self):
        c_good = _candidate(3, 5, 0.1, meets_sla=True)
        c_bad = _candidate(2, 6, 0.4, meets_sla=False)
        analysis = _analysis(
            [c_good, c_bad],
            best=c_good,
            current_config=_util(0.4),
            current_sla=_sla_check(False),
        )
        engine = RecommendationEngine()
        report = engine.analyze(analysis)
        critical = [
            r for r in report.recommendations
            if r.priority == RecommendationPriority.CRITICAL
        ]
        assert len(critical) >= 1
        assert critical[0].action == ActionCategory.REBALANCE
        assert critical[0].suggested_ratio == "3P:5D"


# ---------------------------------------------------------------------------
# Engine: Waste checks
# ---------------------------------------------------------------------------

class TestWasteRecommendations:
    def test_high_waste_triggers_high_priority(self):
        best = _candidate(3, 5, 0.1)
        analysis = _analysis(
            [best, _candidate(2, 6, 0.6)],
            best=best,
            current_config=_util(0.6),
            current_sla=_sla_check(True),
        )
        engine = RecommendationEngine()
        report = engine.analyze(analysis)
        high = [
            r for r in report.recommendations
            if r.priority == RecommendationPriority.HIGH
        ]
        assert len(high) >= 1
        assert high[0].action == ActionCategory.REBALANCE

    def test_moderate_waste_triggers_medium_priority(self):
        best = _candidate(3, 5, 0.1)
        analysis = _analysis(
            [best, _candidate(2, 6, 0.35)],
            best=best,
            current_config=_util(0.35),
            current_sla=_sla_check(True),
        )
        engine = RecommendationEngine()
        report = engine.analyze(analysis)
        med = [
            r for r in report.recommendations
            if r.priority == RecommendationPriority.MEDIUM
        ]
        assert len(med) >= 1

    def test_low_waste_no_rebalance(self):
        best = _candidate(3, 5, 0.1)
        analysis = _analysis(
            [best],
            best=best,
            current_config=_util(0.1),
            current_sla=_sla_check(True),
        )
        engine = RecommendationEngine()
        report = engine.analyze(analysis)
        rebalance = [
            r for r in report.recommendations
            if r.action == ActionCategory.REBALANCE
        ]
        assert len(rebalance) == 0

    def test_over_provisioning_detected(self):
        # All candidates meet SLA with very low waste
        c1 = _candidate(2, 6, 0.02)
        c2 = _candidate(3, 5, 0.03)
        c3 = _candidate(4, 4, 0.01)
        analysis = _analysis([c1, c2, c3], best=c3)
        engine = RecommendationEngine()
        report = engine.analyze(analysis)
        scale_down = [
            r for r in report.recommendations
            if r.action == ActionCategory.SCALE_DOWN
        ]
        assert len(scale_down) >= 1

    def test_custom_thresholds(self):
        best = _candidate(3, 5, 0.05)
        analysis = _analysis(
            [best, _candidate(2, 6, 0.2)],
            best=best,
            current_config=_util(0.2),
            current_sla=_sla_check(True),
        )
        # Custom low threshold to trigger rebalance at 0.2
        engine = RecommendationEngine(waste_threshold=0.15)
        report = engine.analyze(analysis)
        rebalance = [
            r for r in report.recommendations
            if r.action == ActionCategory.REBALANCE
        ]
        assert len(rebalance) >= 1


# ---------------------------------------------------------------------------
# Engine: Cost checks
# ---------------------------------------------------------------------------

class TestCostRecommendations:
    def test_cost_savings_detected(self):
        # Current uses 10 instances, cheaper option uses 8
        c_expensive = _candidate(5, 5, 0.3)
        c_cheap = _candidate(3, 5, 0.1)
        analysis = _analysis(
            [c_expensive, c_cheap],
            best=c_cheap,
            current_config=_util(0.3),
            current_sla=_sla_check(True),
            total=10,
        )
        cost_config = CostConfig(gpu_hourly_rate=10.0, currency="USD")
        engine = RecommendationEngine(cost_config=cost_config)
        report = engine.analyze(analysis, measured_qps=100.0)
        cost_recs = [
            r for r in report.recommendations
            if "cost" in r.title.lower() or "Cost" in r.title
        ]
        assert len(cost_recs) >= 1

    def test_no_cost_rec_without_config(self):
        best = _candidate(3, 5, 0.1)
        analysis = _analysis([best], best=best)
        engine = RecommendationEngine()
        report = engine.analyze(analysis, measured_qps=100.0)
        cost_recs = [
            r for r in report.recommendations
            if "cost" in r.title.lower()
        ]
        assert len(cost_recs) == 0

    def test_no_cost_rec_without_qps(self):
        best = _candidate(3, 5, 0.1)
        analysis = _analysis([best], best=best)
        cost_config = CostConfig(gpu_hourly_rate=10.0, currency="USD")
        engine = RecommendationEngine(cost_config=cost_config)
        report = engine.analyze(analysis)  # no measured_qps
        cost_recs = [
            r for r in report.recommendations
            if "cost" in r.title.lower()
        ]
        assert len(cost_recs) == 0


# ---------------------------------------------------------------------------
# Engine: No-action
# ---------------------------------------------------------------------------

class TestNoAction:
    def test_optimal_config_returns_no_action(self):
        best = _candidate(3, 5, 0.1)
        analysis = _analysis(
            [best],
            best=best,
            current_config=_util(0.1),
            current_sla=_sla_check(True),
        )
        engine = RecommendationEngine()
        report = engine.analyze(analysis)
        assert any(r.action == ActionCategory.NO_ACTION for r in report.recommendations)


# ---------------------------------------------------------------------------
# Engine: Sorting
# ---------------------------------------------------------------------------

class TestSorting:
    def test_recommendations_sorted_by_priority(self):
        # SLA failure (CRITICAL) + high waste (HIGH) should be ordered
        c_bad = _candidate(2, 6, 0.6, meets_sla=False)
        c_good = _candidate(3, 5, 0.1, meets_sla=True)
        analysis = _analysis(
            [c_bad, c_good],
            best=c_good,
            current_config=_util(0.6),
            current_sla=_sla_check(False),
        )
        engine = RecommendationEngine()
        report = engine.analyze(analysis)
        priorities = [r.priority_order for r in report.recommendations]
        assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# Engine: Report fields
# ---------------------------------------------------------------------------

class TestReportFields:
    def test_report_summary(self):
        c1 = _candidate(3, 5, 0.1)
        c2 = _candidate(2, 6, 0.3, meets_sla=False)
        analysis = _analysis([c1, c2], best=c1, total=8)
        engine = RecommendationEngine()
        report = engine.analyze(analysis)
        assert report.total_instances == 8
        assert report.optimal_ratio == "3P:5D"
        assert "1/2" in report.analysis_summary

    def test_report_serializable(self):
        best = _candidate(3, 5, 0.1)
        analysis = _analysis([best], best=best)
        report = get_recommendations(analysis)
        data = report.model_dump()
        assert isinstance(json.dumps(data), str)


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------

class TestProgrammaticAPI:
    def test_get_recommendations_basic(self):
        best = _candidate(3, 5, 0.1)
        analysis = _analysis([best], best=best)
        report = get_recommendations(analysis)
        assert isinstance(report, RecommendationReport)
        assert len(report.recommendations) >= 1

    def test_get_recommendations_with_cost(self):
        c1 = _candidate(5, 5, 0.3)
        c2 = _candidate(3, 5, 0.1)
        analysis = _analysis(
            [c1, c2],
            best=c2,
            current_config=_util(0.3),
            current_sla=_sla_check(True),
            total=10,
        )
        cost_config = CostConfig(gpu_hourly_rate=10.0, currency="USD")
        report = get_recommendations(
            analysis, measured_qps=100.0, cost_config=cost_config
        )
        assert isinstance(report, RecommendationReport)

    def test_get_recommendations_custom_thresholds(self):
        best = _candidate(3, 5, 0.1)
        analysis = _analysis([best], best=best)
        report = get_recommendations(
            analysis, waste_threshold=0.5, critical_waste_threshold=0.8
        )
        assert isinstance(report, RecommendationReport)

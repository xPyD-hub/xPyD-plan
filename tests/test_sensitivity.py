"""Tests for sensitivity analysis module."""

from __future__ import annotations

import pytest

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.models import SLAConfig
from xpyd_plan.sensitivity import (
    CliffPoint,
    SafetyRecommendation,
    SensitivityPoint,
    SensitivityResult,
    analyze_sensitivity,
)


def _make_benchmark_data(
    num_prefill: int = 2,
    num_decode: int = 6,
    num_requests: int = 50,
    qps: float = 10.0,
    base_ttft: float = 50.0,
    base_tpot: float = 15.0,
) -> dict:
    """Generate synthetic benchmark data."""
    import random

    random.seed(42)
    requests = []
    for i in range(num_requests):
        prompt_tokens = random.randint(100, 500)
        output_tokens = random.randint(50, 200)
        ttft = base_ttft + random.gauss(0, base_ttft * 0.2)
        tpot = base_tpot + random.gauss(0, base_tpot * 0.1)
        total = max(ttft + tpot * output_tokens, 1.0)
        requests.append({
            "request_id": f"req-{i:04d}",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": max(ttft, 0.1),
            "tpot_ms": max(tpot, 0.1),
            "total_latency_ms": max(total, 0.1),
            "timestamp": 1700000000.0 + i / qps,
        })
    return {
        "metadata": {
            "num_prefill_instances": num_prefill,
            "num_decode_instances": num_decode,
            "total_instances": num_prefill + num_decode,
            "measured_qps": qps,
        },
        "requests": requests,
    }


@pytest.fixture
def analyzer() -> BenchmarkAnalyzer:
    a = BenchmarkAnalyzer()
    a.load_data_from_dict(_make_benchmark_data())
    return a


@pytest.fixture
def sla() -> SLAConfig:
    return SLAConfig(ttft_ms=200.0, tpot_ms=50.0, max_latency_ms=5000.0)


class TestAnalyzeSensitivity:
    def test_basic_output_structure(self, analyzer: BenchmarkAnalyzer, sla: SLAConfig):
        result = analyze_sensitivity(analyzer, 8, sla)
        assert isinstance(result, SensitivityResult)
        assert result.total_instances == 8
        assert len(result.points) == 7

    def test_points_cover_all_splits(self, analyzer: BenchmarkAnalyzer, sla: SLAConfig):
        result = analyze_sensitivity(analyzer, 8, sla)
        prefills = [p.num_prefill for p in result.points]
        assert prefills == [1, 2, 3, 4, 5, 6, 7]
        for p in result.points:
            assert p.num_prefill + p.num_decode == 8

    def test_margins_computed_when_sla_set(self, analyzer: BenchmarkAnalyzer, sla: SLAConfig):
        result = analyze_sensitivity(analyzer, 8, sla)
        for point in result.points:
            assert point.ttft_margin_ms is not None
            assert point.tpot_margin_ms is not None
            assert point.total_latency_margin_ms is not None
            assert point.ttft_margin_pct is not None

    def test_margins_none_when_sla_unset(self, analyzer: BenchmarkAnalyzer):
        sla = SLAConfig()
        result = analyze_sensitivity(analyzer, 8, sla)
        for point in result.points:
            assert point.ttft_margin_ms is None
            assert point.tpot_margin_ms is None
            assert point.total_latency_margin_ms is None

    def test_positive_margin_means_headroom(self, analyzer: BenchmarkAnalyzer, sla: SLAConfig):
        result = analyze_sensitivity(analyzer, 8, sla)
        for point in result.points:
            if point.meets_sla:
                if point.ttft_margin_ms is not None:
                    assert point.ttft_margin_ms >= 0
                if point.tpot_margin_ms is not None:
                    assert point.tpot_margin_ms >= 0

    def test_negative_margin_means_violation(self, analyzer: BenchmarkAnalyzer):
        tight_sla = SLAConfig(ttft_ms=10.0, tpot_ms=5.0)
        result = analyze_sensitivity(analyzer, 8, tight_sla)
        failing = [p for p in result.points if not p.meets_sla]
        assert len(failing) > 0
        for p in failing:
            has_negative = False
            if p.ttft_margin_ms is not None and p.ttft_margin_ms < 0:
                has_negative = True
            if p.tpot_margin_ms is not None and p.tpot_margin_ms < 0:
                has_negative = True
            assert has_negative

    def test_too_few_instances(self, analyzer: BenchmarkAnalyzer, sla: SLAConfig):
        result = analyze_sensitivity(analyzer, 1, sla)
        assert len(result.points) == 0
        assert len(result.cliffs) == 0

    def test_ratio_str_format(self, analyzer: BenchmarkAnalyzer, sla: SLAConfig):
        result = analyze_sensitivity(analyzer, 8, sla)
        for p in result.points:
            assert "P:" in p.ratio_str
            assert "D" in p.ratio_str


class TestCliffDetection:
    def test_cliff_detected_on_transition(self, analyzer: BenchmarkAnalyzer):
        sla = SLAConfig(ttft_ms=100.0, tpot_ms=30.0)
        result = analyze_sensitivity(analyzer, 8, sla)
        passing = [p for p in result.points if p.meets_sla]
        failing = [p for p in result.points if not p.meets_sla]
        if passing and failing:
            assert len(result.cliffs) >= 1

    def test_cliff_has_last_pass_and_first_fail(self, analyzer: BenchmarkAnalyzer):
        sla = SLAConfig(ttft_ms=100.0, tpot_ms=30.0)
        result = analyze_sensitivity(analyzer, 8, sla)
        for cliff in result.cliffs:
            assert isinstance(cliff, CliffPoint)
            assert cliff.last_pass is not None or cliff.first_fail is not None
            if cliff.last_pass and cliff.first_fail:
                assert cliff.last_pass.meets_sla
                assert not cliff.first_fail.meets_sla

    def test_cliff_identifies_failing_metric(self, analyzer: BenchmarkAnalyzer):
        sla = SLAConfig(ttft_ms=100.0, tpot_ms=30.0)
        result = analyze_sensitivity(analyzer, 8, sla)
        for cliff in result.cliffs:
            if cliff.failing_metric:
                assert cliff.failing_metric in ("ttft", "tpot", "total_latency")

    def test_no_cliff_when_all_pass(self, analyzer: BenchmarkAnalyzer):
        sla = SLAConfig(ttft_ms=100000.0, tpot_ms=100000.0)
        result = analyze_sensitivity(analyzer, 8, sla)
        assert len(result.cliffs) == 0

    def test_no_cliff_when_all_fail(self, analyzer: BenchmarkAnalyzer):
        sla = SLAConfig(ttft_ms=0.001, tpot_ms=0.001)
        result = analyze_sensitivity(analyzer, 8, sla)
        assert len(result.cliffs) == 0

    def test_cliff_direction(self, analyzer: BenchmarkAnalyzer):
        sla = SLAConfig(ttft_ms=100.0, tpot_ms=30.0)
        result = analyze_sensitivity(analyzer, 8, sla)
        for cliff in result.cliffs:
            assert cliff.direction in ("prefill_heavy", "decode_heavy")


class TestSafetyRecommendation:
    def test_recommendation_exists_when_passing_ratios(
        self, analyzer: BenchmarkAnalyzer, sla: SLAConfig
    ):
        result = analyze_sensitivity(analyzer, 8, sla)
        passing = [p for p in result.points if p.meets_sla]
        if passing:
            assert result.recommendation is not None
            assert result.recommendation.recommended is not None
            assert result.recommendation.optimal is not None

    def test_recommendation_optimal_has_min_waste(
        self, analyzer: BenchmarkAnalyzer, sla: SLAConfig
    ):
        result = analyze_sensitivity(analyzer, 8, sla)
        if result.recommendation and result.recommendation.optimal:
            passing = [p for p in result.points if p.meets_sla]
            min_waste = min(p.waste_rate for p in passing)
            assert result.recommendation.optimal.waste_rate == pytest.approx(min_waste)

    def test_recommendation_cliff_distance_positive(
        self, analyzer: BenchmarkAnalyzer, sla: SLAConfig
    ):
        result = analyze_sensitivity(analyzer, 8, sla)
        if result.recommendation:
            assert result.recommendation.cliff_distance >= 0

    def test_no_recommendation_when_all_fail(self, analyzer: BenchmarkAnalyzer):
        sla = SLAConfig(ttft_ms=0.001, tpot_ms=0.001)
        result = analyze_sensitivity(analyzer, 8, sla)
        assert result.recommendation is not None
        assert result.recommendation.recommended is None

    def test_recommendation_recommended_meets_sla(
        self, analyzer: BenchmarkAnalyzer, sla: SLAConfig
    ):
        result = analyze_sensitivity(analyzer, 8, sla)
        if result.recommendation and result.recommendation.recommended:
            assert result.recommendation.recommended.meets_sla

    def test_min_margin_pct_computed(self, analyzer: BenchmarkAnalyzer, sla: SLAConfig):
        result = analyze_sensitivity(analyzer, 8, sla)
        if result.recommendation and result.recommendation.recommended:
            assert result.recommendation.min_margin_pct is not None
            assert result.recommendation.min_margin_pct >= 0


class TestSensitivityModels:
    def test_sensitivity_point_serialization(self):
        from xpyd_plan.benchmark_models import SLACheck

        sla_check = SLACheck(
            ttft_p95_ms=50.0, ttft_p99_ms=60.0,
            tpot_p95_ms=15.0, tpot_p99_ms=18.0,
            total_latency_p95_ms=2000.0, total_latency_p99_ms=2500.0,
            meets_ttft=True, meets_tpot=True, meets_total_latency=True, meets_all=True,
        )
        point = SensitivityPoint(
            num_prefill=2, num_decode=6, ratio_str="2P:6D", meets_sla=True,
            sla_check=sla_check, ttft_margin_ms=150.0, tpot_margin_ms=35.0,
            total_latency_margin_ms=3000.0, ttft_margin_pct=0.75,
            tpot_margin_pct=0.70, total_latency_margin_pct=0.60, waste_rate=0.15,
        )
        data = point.model_dump()
        assert data["num_prefill"] == 2
        assert data["ttft_margin_pct"] == 0.75

    def test_sensitivity_result_serialization(self):
        result = SensitivityResult(points=[], cliffs=[], total_instances=8)
        data = result.model_dump()
        assert data["total_instances"] == 8

    def test_cliff_point_serialization(self):
        cliff = CliffPoint(
            last_pass=None, first_fail=None,
            failing_metric="ttft", direction="prefill_heavy",
        )
        data = cliff.model_dump()
        assert data["failing_metric"] == "ttft"

    def test_safety_recommendation_empty(self):
        rec = SafetyRecommendation()
        assert rec.recommended is None
        assert rec.cliff_distance == 0

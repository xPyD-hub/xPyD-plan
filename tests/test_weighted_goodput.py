"""Tests for weighted goodput analysis module."""

from __future__ import annotations

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.weighted_goodput import (
    ScoringConfig,
    WeightedGoodputAnalyzer,
    _score_metric,
    analyze_weighted_goodput,
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


class TestScoreMetric:
    def test_within_threshold(self):
        assert _score_metric(50.0, 100.0, 0.5) == 1.0

    def test_at_threshold(self):
        assert _score_metric(100.0, 100.0, 0.5) == 1.0

    def test_beyond_threshold_partial(self):
        # threshold=100, grace=0.5 -> grace_window=50
        # measured=125 -> (125-100)/50 = 0.5 -> score=0.5
        assert _score_metric(125.0, 100.0, 0.5) == pytest.approx(0.5)

    def test_beyond_threshold_full(self):
        # measured=150 -> (150-100)/50 = 1.0 -> score=0.0
        assert _score_metric(150.0, 100.0, 0.5) == 0.0

    def test_far_beyond_threshold(self):
        assert _score_metric(999.0, 100.0, 0.5) == 0.0

    def test_zero_grace_binary(self):
        assert _score_metric(99.0, 100.0, 0.0) == 1.0
        assert _score_metric(100.0, 100.0, 0.0) == 1.0
        assert _score_metric(100.1, 100.0, 0.0) == 0.0

    def test_large_grace(self):
        # threshold=100, grace=1.0 -> grace_window=100
        # measured=150 -> (150-100)/100 = 0.5
        assert _score_metric(150.0, 100.0, 1.0) == pytest.approx(0.5)


class TestScoringConfig:
    def test_default_values(self):
        c = ScoringConfig(sla_ttft_ms=100.0)
        assert c.grace_factor == 0.5
        assert c.aggregation == "min"

    def test_custom_values(self):
        c = ScoringConfig(
            sla_ttft_ms=100.0,
            sla_tpot_ms=50.0,
            grace_factor=0.3,
            aggregation="mean",
        )
        assert c.grace_factor == 0.3
        assert c.aggregation == "mean"


class TestWeightedGoodputAnalyzerInit:
    def test_no_sla_raises(self):
        with pytest.raises(ValueError, match="At least one SLA"):
            WeightedGoodputAnalyzer(ScoringConfig())

    def test_invalid_aggregation(self):
        with pytest.raises(ValueError, match="aggregation"):
            WeightedGoodputAnalyzer(
                ScoringConfig(sla_ttft_ms=100.0, aggregation="bad")
            )


class TestWeightedGoodputAnalyzer:
    def test_all_passing(self):
        reqs = [_make_request(request_id=f"r{i}", ttft_ms=50.0) for i in range(10)]
        config = ScoringConfig(sla_ttft_ms=100.0)
        analyzer = WeightedGoodputAnalyzer(config)
        report = analyzer.analyze(_make_data(reqs))
        assert report.weighted_goodput == 1.0
        assert report.binary_goodput_ratio == 1.0
        assert report.near_miss_count == 0

    def test_all_failing_hard(self):
        reqs = [
            _make_request(request_id=f"r{i}", ttft_ms=200.0) for i in range(10)
        ]
        config = ScoringConfig(sla_ttft_ms=100.0)
        analyzer = WeightedGoodputAnalyzer(config)
        report = analyzer.analyze(_make_data(reqs))
        assert report.weighted_goodput == 0.0
        assert report.binary_goodput_ratio == 0.0

    def test_near_miss_requests(self):
        # ttft=110, threshold=100, grace=0.5 -> score = 1 - 10/50 = 0.8
        reqs = [_make_request(request_id="r1", ttft_ms=110.0)]
        config = ScoringConfig(sla_ttft_ms=100.0)
        analyzer = WeightedGoodputAnalyzer(config)
        report = analyzer.analyze(_make_data(reqs, qps=1.0))
        assert report.weighted_goodput == pytest.approx(0.8)
        assert report.binary_goodput_ratio == 0.0
        assert report.near_miss_count == 1

    def test_mixed_scores(self):
        reqs = [
            _make_request(request_id="r1", ttft_ms=50.0),  # score=1.0
            _make_request(request_id="r2", ttft_ms=125.0),  # score=0.5
            _make_request(request_id="r3", ttft_ms=200.0),  # score=0.0
        ]
        config = ScoringConfig(sla_ttft_ms=100.0)
        analyzer = WeightedGoodputAnalyzer(config)
        report = analyzer.analyze(_make_data(reqs, qps=3.0))
        assert report.weighted_goodput == pytest.approx(0.5)
        assert report.binary_goodput_ratio == pytest.approx(1 / 3)
        assert report.near_miss_count == 1

    def test_min_aggregation_multi_metric(self):
        # ttft=110 -> score=0.8, tpot=20 -> score=1.0
        # min -> 0.8
        reqs = [_make_request(request_id="r1", ttft_ms=110.0, tpot_ms=20.0)]
        config = ScoringConfig(sla_ttft_ms=100.0, sla_tpot_ms=50.0, aggregation="min")
        analyzer = WeightedGoodputAnalyzer(config)
        report = analyzer.analyze(_make_data(reqs, qps=1.0))
        assert report.weighted_goodput == pytest.approx(0.8)

    def test_mean_aggregation_multi_metric(self):
        # ttft=110 -> score=0.8, tpot=20 -> score=1.0
        # mean -> 0.9
        reqs = [_make_request(request_id="r1", ttft_ms=110.0, tpot_ms=20.0)]
        config = ScoringConfig(sla_ttft_ms=100.0, sla_tpot_ms=50.0, aggregation="mean")
        analyzer = WeightedGoodputAnalyzer(config)
        report = analyzer.analyze(_make_data(reqs, qps=1.0))
        assert report.weighted_goodput == pytest.approx(0.9)

    def test_grace_zero_is_binary(self):
        reqs = [
            _make_request(request_id="r1", ttft_ms=99.0),
            _make_request(request_id="r2", ttft_ms=101.0),
        ]
        config = ScoringConfig(sla_ttft_ms=100.0, grace_factor=0.0)
        analyzer = WeightedGoodputAnalyzer(config)
        report = analyzer.analyze(_make_data(reqs, qps=2.0))
        assert report.weighted_goodput == pytest.approx(0.5)
        assert report.binary_goodput_ratio == pytest.approx(0.5)

    def test_weighted_goodput_qps(self):
        reqs = [_make_request(request_id=f"r{i}", ttft_ms=50.0) for i in range(5)]
        config = ScoringConfig(sla_ttft_ms=100.0)
        analyzer = WeightedGoodputAnalyzer(config)
        report = analyzer.analyze(_make_data(reqs, qps=20.0))
        assert report.weighted_goodput_qps == pytest.approx(20.0)

    def test_score_distribution_buckets(self):
        reqs = [
            _make_request(request_id="r1", ttft_ms=50.0),   # score=1.0
            _make_request(request_id="r2", ttft_ms=125.0),  # score=0.5
            _make_request(request_id="r3", ttft_ms=140.0),  # score=0.2
        ]
        config = ScoringConfig(sla_ttft_ms=100.0)
        analyzer = WeightedGoodputAnalyzer(config)
        report = analyzer.analyze(_make_data(reqs, qps=3.0))
        dist = report.score_distribution
        assert len(dist.buckets) == 5
        total_count = sum(b.count for b in dist.buckets)
        assert total_count == 3

    def test_recommendation_excellent(self):
        reqs = [_make_request(request_id=f"r{i}", ttft_ms=50.0) for i in range(10)]
        config = ScoringConfig(sla_ttft_ms=100.0)
        report = WeightedGoodputAnalyzer(config).analyze(_make_data(reqs))
        assert "Excellent" in report.recommendation

    def test_recommendation_poor(self):
        reqs = [_make_request(request_id=f"r{i}", ttft_ms=200.0) for i in range(10)]
        config = ScoringConfig(sla_ttft_ms=100.0)
        report = WeightedGoodputAnalyzer(config).analyze(_make_data(reqs))
        assert "Poor" in report.recommendation


class TestAnalyzeWeightedGoodputAPI:
    def test_returns_dict(self):
        reqs = [_make_request(request_id="r1", ttft_ms=50.0)]
        data = _make_data(reqs, qps=1.0)
        result = analyze_weighted_goodput(data, sla_ttft_ms=100.0)
        assert isinstance(result, dict)
        assert result["weighted_goodput"] == 1.0

    def test_custom_grace(self):
        reqs = [_make_request(request_id="r1", ttft_ms=110.0)]
        data = _make_data(reqs, qps=1.0)
        # grace=1.0 -> window=100 -> (110-100)/100 = 0.1 -> score=0.9
        result = analyze_weighted_goodput(
            data, sla_ttft_ms=100.0, grace_factor=1.0
        )
        assert result["weighted_goodput"] == pytest.approx(0.9)

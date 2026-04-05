"""Tests for goodput analysis module."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.goodput import (
    GoodputAnalyzer,
    GoodputGrade,
    GoodputWindow,
    _grade_from_ratio,
    _request_passes_sla,
    analyze_goodput,
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


class TestGradeFromRatio:
    def test_excellent(self):
        assert _grade_from_ratio(0.99) == GoodputGrade.EXCELLENT
        assert _grade_from_ratio(1.0) == GoodputGrade.EXCELLENT

    def test_good(self):
        assert _grade_from_ratio(0.95) == GoodputGrade.GOOD
        assert _grade_from_ratio(0.98) == GoodputGrade.GOOD

    def test_fair(self):
        assert _grade_from_ratio(0.80) == GoodputGrade.FAIR
        assert _grade_from_ratio(0.94) == GoodputGrade.FAIR

    def test_poor(self):
        assert _grade_from_ratio(0.79) == GoodputGrade.POOR
        assert _grade_from_ratio(0.0) == GoodputGrade.POOR


class TestRequestPassesSla:
    def test_all_pass(self):
        req = _make_request(ttft_ms=50, tpot_ms=20, total_latency_ms=200)
        ok, failures = _request_passes_sla(req, 100, 50, 500)
        assert ok is True
        assert failures == []

    def test_ttft_fail(self):
        req = _make_request(ttft_ms=150)
        ok, failures = _request_passes_sla(req, 100, None, None)
        assert ok is False
        assert len(failures) == 1

    def test_multi_fail(self):
        req = _make_request(ttft_ms=150, tpot_ms=60, total_latency_ms=600)
        ok, failures = _request_passes_sla(req, 100, 50, 500)
        assert ok is False
        assert len(failures) == 3

    def test_no_thresholds_passes(self):
        req = _make_request()
        ok, failures = _request_passes_sla(req, None, None, None)
        assert ok is True


class TestGoodputAnalyzerInit:
    def test_no_sla_raises(self):
        with pytest.raises(ValueError, match="At least one SLA"):
            GoodputAnalyzer()

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match="window_size must be positive"):
            GoodputAnalyzer(sla_ttft_ms=100, window_size=-1)


class TestGoodputAnalyzer:
    def test_all_passing(self):
        reqs = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000 + i)
            for i in range(20)
        ]
        data = _make_data(reqs)
        analyzer = GoodputAnalyzer(sla_ttft_ms=100)
        report = analyzer.analyze(data)

        assert report.total_requests == 20
        assert report.passing_requests == 20
        assert report.failing_requests == 0
        assert report.goodput_ratio == 1.0
        assert report.grade == GoodputGrade.EXCELLENT
        assert report.failure_breakdown.ttft_failures == 0

    def test_some_failing(self):
        reqs = []
        for i in range(10):
            reqs.append(_make_request(request_id=f"pass{i}", ttft_ms=50, timestamp=1000 + i))
        for i in range(5):
            reqs.append(_make_request(request_id=f"fail{i}", ttft_ms=150, timestamp=1010 + i))
        data = _make_data(reqs, qps=15.0)
        analyzer = GoodputAnalyzer(sla_ttft_ms=100)
        report = analyzer.analyze(data)

        assert report.total_requests == 15
        assert report.passing_requests == 10
        assert report.failing_requests == 5
        assert abs(report.goodput_ratio - 10 / 15) < 1e-6
        assert report.grade == GoodputGrade.POOR  # ~66.7%
        assert report.failure_breakdown.ttft_failures == 5

    def test_goodput_qps(self):
        reqs = []
        for i in range(8):
            reqs.append(_make_request(request_id=f"p{i}", ttft_ms=50, timestamp=1000 + i))
        for i in range(2):
            reqs.append(_make_request(request_id=f"f{i}", ttft_ms=150, timestamp=1008 + i))
        data = _make_data(reqs, qps=10.0)
        analyzer = GoodputAnalyzer(sla_ttft_ms=100)
        report = analyzer.analyze(data)

        assert report.raw_qps == 10.0
        assert abs(report.goodput_qps - 8.0) < 1e-6

    def test_multi_metric_failures(self):
        reqs = [
            _make_request(
                request_id="r1", ttft_ms=150, tpot_ms=60, total_latency_ms=600, timestamp=1000
            ),
            _make_request(
                request_id="r2", ttft_ms=50, tpot_ms=20, total_latency_ms=200, timestamp=1001
            ),
        ]
        data = _make_data(reqs, qps=2.0)
        analyzer = GoodputAnalyzer(sla_ttft_ms=100, sla_tpot_ms=50, sla_total_latency_ms=500)
        report = analyzer.analyze(data)

        assert report.failing_requests == 1
        assert report.failure_breakdown.ttft_failures == 1
        assert report.failure_breakdown.tpot_failures == 1
        assert report.failure_breakdown.total_latency_failures == 1
        assert report.failure_breakdown.multi_metric_failures == 1

    def test_windows_generated(self):
        reqs = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000 + i)
            for i in range(20)
        ]
        data = _make_data(reqs)
        analyzer = GoodputAnalyzer(sla_ttft_ms=100, window_size=5.0)
        report = analyzer.analyze(data)

        assert len(report.windows) > 0
        for w in report.windows:
            assert isinstance(w, GoodputWindow)
            assert w.goodput_ratio >= 0
            assert w.goodput_ratio <= 1

    def test_worst_window_tracked(self):
        # First 10 requests pass, next 10 fail — different time windows
        reqs = []
        for i in range(10):
            reqs.append(_make_request(request_id=f"p{i}", ttft_ms=50, timestamp=1000 + i))
        for i in range(10):
            reqs.append(_make_request(request_id=f"f{i}", ttft_ms=150, timestamp=1010 + i))
        data = _make_data(reqs)
        analyzer = GoodputAnalyzer(sla_ttft_ms=100, window_size=5.0)
        report = analyzer.analyze(data)

        assert report.worst_window_goodput < 1.0

    def test_sla_thresholds_recorded(self):
        reqs = [_make_request(timestamp=1000)]
        data = _make_data(reqs)
        analyzer = GoodputAnalyzer(sla_ttft_ms=100, sla_tpot_ms=50)
        report = analyzer.analyze(data)

        assert report.sla_ttft_ms == 100
        assert report.sla_tpot_ms == 50
        assert report.sla_total_latency_ms is None

    def test_same_timestamp_single_window(self):
        reqs = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000)
            for i in range(5)
        ]
        data = _make_data(reqs)
        analyzer = GoodputAnalyzer(sla_ttft_ms=100)
        report = analyzer.analyze(data)

        assert len(report.windows) == 1
        assert report.windows[0].total_requests == 5

    def test_recommendation_excellent(self):
        reqs = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000 + i)
            for i in range(10)
        ]
        data = _make_data(reqs)
        analyzer = GoodputAnalyzer(sla_ttft_ms=100)
        report = analyzer.analyze(data)
        assert "excellent" in report.recommendation.lower()

    def test_recommendation_poor(self):
        reqs = [
            _make_request(request_id=f"r{i}", ttft_ms=150, timestamp=1000 + i)
            for i in range(10)
        ]
        data = _make_data(reqs)
        analyzer = GoodputAnalyzer(sla_ttft_ms=100)
        report = analyzer.analyze(data)
        assert "poor" in report.recommendation.lower()

    def test_grade_good(self):
        # 96% pass rate
        reqs = []
        for i in range(96):
            reqs.append(_make_request(request_id=f"p{i}", ttft_ms=50, timestamp=1000 + i * 0.1))
        for i in range(4):
            reqs.append(_make_request(request_id=f"f{i}", ttft_ms=150, timestamp=1010 + i * 0.1))
        data = _make_data(reqs)
        analyzer = GoodputAnalyzer(sla_ttft_ms=100)
        report = analyzer.analyze(data)
        assert report.grade == GoodputGrade.GOOD

    def test_grade_fair(self):
        # 85% pass rate
        reqs = []
        for i in range(85):
            reqs.append(_make_request(request_id=f"p{i}", ttft_ms=50, timestamp=1000 + i * 0.1))
        for i in range(15):
            reqs.append(_make_request(request_id=f"f{i}", ttft_ms=150, timestamp=1009 + i * 0.1))
        data = _make_data(reqs)
        analyzer = GoodputAnalyzer(sla_ttft_ms=100)
        report = analyzer.analyze(data)
        assert report.grade == GoodputGrade.FAIR


class TestAnalyzeGoodputAPI:
    def test_programmatic_api(self, tmp_path):
        reqs = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000 + i)
            for i in range(10)
        ]
        data = _make_data(reqs)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        result = analyze_goodput(str(path), sla_ttft_ms=100)
        assert isinstance(result, dict)
        assert result["total_requests"] == 10
        assert result["goodput_ratio"] == 1.0

    def test_programmatic_api_with_failures(self, tmp_path):
        reqs = [
            _make_request(request_id="pass", ttft_ms=50, timestamp=1000),
            _make_request(request_id="fail", ttft_ms=150, timestamp=1001),
        ]
        data = _make_data(reqs, qps=2.0)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        result = analyze_goodput(str(path), sla_ttft_ms=100)
        assert result["passing_requests"] == 1
        assert result["failing_requests"] == 1
        assert result["failure_breakdown"]["ttft_failures"] == 1

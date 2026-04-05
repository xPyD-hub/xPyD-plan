"""Tests for cold start detection (M78)."""

from __future__ import annotations

import json
import random

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.cold_start import (
    ColdStartDetector,
    ColdStartReport,
    ColdStartSeverity,
    _classify_severity,
    _find_stabilization,
    _percentile,
    detect_cold_start,
)


def _make_data(
    n: int = 100,
    warmup_ttft: float = 100.0,
    steady_ttft: float = 100.0,
    warmup_count: int = 10,
) -> BenchmarkData:
    """Generate benchmark data with optional cold start effect."""
    rng = random.Random(42)
    requests = []
    for i in range(n):
        if i < warmup_count:
            ttft = warmup_ttft + rng.uniform(-5, 5)
        else:
            ttft = steady_ttft + rng.uniform(-5, 5)
        tpot = 20.0 + rng.uniform(-1, 1)
        total = ttft + tpot * 50
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=round(ttft, 2),
                tpot_ms=round(tpot, 2),
                total_latency_ms=round(total, 2),
                timestamp=1000.0 + i * 0.1,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=100.0,
        ),
        requests=requests,
    )


class TestClassifySeverity:
    def test_none(self):
        assert _classify_severity(1.5, 2.0) == ColdStartSeverity.NONE

    def test_mild(self):
        assert _classify_severity(3.0, 2.0) == ColdStartSeverity.MILD

    def test_moderate(self):
        assert _classify_severity(6.0, 2.0) == ColdStartSeverity.MODERATE

    def test_severe(self):
        assert _classify_severity(10.0, 2.0) == ColdStartSeverity.SEVERE

    def test_boundary_none(self):
        assert _classify_severity(2.0, 2.0) == ColdStartSeverity.NONE

    def test_boundary_mild(self):
        assert _classify_severity(4.0, 2.0) == ColdStartSeverity.MILD


class TestPercentile:
    def test_single(self):
        assert _percentile([5.0], 50.0) == 5.0

    def test_empty(self):
        assert _percentile([], 50.0) == 0.0

    def test_median(self):
        assert _percentile([1.0, 2.0, 3.0], 50.0) == 2.0


class TestFindStabilization:
    def test_already_stable(self):
        values = [10.0] * 20
        assert _find_stabilization(values, 10.0, 2.0) == 0

    def test_cold_start_then_stable(self):
        values = [500.0] * 5 + [100.0] * 20
        idx = _find_stabilization(values, 100.0, 2.0)
        assert idx <= 6  # should stabilize around index 5-6


class TestColdStartDetector:
    def test_no_cold_start(self):
        data = _make_data(warmup_ttft=100.0, steady_ttft=100.0)
        detector = ColdStartDetector()
        report = detector.detect(data)

        assert isinstance(report, ColdStartReport)
        assert not report.has_cold_start
        for m in report.metrics:
            assert m.severity == ColdStartSeverity.NONE

    def test_cold_start_detected(self):
        data = _make_data(warmup_ttft=500.0, steady_ttft=100.0)
        detector = ColdStartDetector()
        report = detector.detect(data)

        assert report.has_cold_start
        ttft = next(m for m in report.metrics if m.metric == "ttft_ms")
        assert ttft.severity != ColdStartSeverity.NONE
        assert ttft.ratio > 2.0

    def test_severe_cold_start(self):
        data = _make_data(warmup_ttft=2000.0, steady_ttft=100.0)
        detector = ColdStartDetector()
        report = detector.detect(data)

        ttft = next(m for m in report.metrics if m.metric == "ttft_ms")
        assert ttft.severity == ColdStartSeverity.SEVERE

    def test_metric_count(self):
        data = _make_data()
        report = ColdStartDetector().detect(data)
        assert len(report.metrics) == 3

    def test_metric_names(self):
        data = _make_data()
        report = ColdStartDetector().detect(data)
        names = [m.metric for m in report.metrics]
        assert names == ["ttft_ms", "tpot_ms", "total_latency_ms"]

    def test_total_requests(self):
        data = _make_data(n=50)
        report = ColdStartDetector().detect(data, warmup_window=10)
        assert report.total_requests == 50
        assert report.warmup_size == 10

    def test_custom_warmup_window(self):
        data = _make_data(n=100, warmup_ttft=500.0, warmup_count=20)
        report = ColdStartDetector().detect(data, warmup_window=20)
        assert report.warmup_size == 20

    def test_custom_threshold(self):
        data = _make_data(warmup_ttft=200.0, steady_ttft=100.0)
        # With high threshold, should not detect cold start
        report = ColdStartDetector().detect(data, threshold=5.0)
        assert not report.has_cold_start

    def test_too_few_requests(self):
        data = _make_data(n=15)
        with pytest.raises(ValueError, match="Need at least"):
            ColdStartDetector().detect(data, warmup_window=10)

    def test_recommendation_no_cold_start(self):
        data = _make_data(warmup_ttft=100.0, steady_ttft=100.0)
        report = ColdStartDetector().detect(data)
        assert "no significant" in report.recommendation.lower()

    def test_recommendation_severe(self):
        data = _make_data(warmup_ttft=2000.0, steady_ttft=100.0)
        report = ColdStartDetector().detect(data)
        assert "severe" in report.recommendation.lower()

    def test_worst_metric(self):
        data = _make_data(warmup_ttft=500.0, steady_ttft=100.0)
        report = ColdStartDetector().detect(data)
        assert report.worst_metric in ("ttft_ms", "tpot_ms", "total_latency_ms")

    def test_stabilization_index(self):
        data = _make_data(warmup_ttft=500.0, steady_ttft=100.0, warmup_count=10)
        report = ColdStartDetector().detect(data)
        ttft = next(m for m in report.metrics if m.metric == "ttft_ms")
        assert ttft.stabilization_index >= 0


class TestDetectColdStartAPI:
    def test_returns_dict(self):
        data = _make_data()
        result = detect_cold_start(data)
        assert isinstance(result, dict)
        assert "metrics" in result
        assert "has_cold_start" in result
        assert "worst_metric" in result

    def test_custom_params(self):
        data = _make_data(warmup_ttft=500.0, steady_ttft=100.0)
        result = detect_cold_start(data, warmup_window=15, threshold=3.0)
        assert result["warmup_size"] == 15
        assert result["threshold"] == 3.0


class TestCLIColdStart:
    def test_json_output(self):
        """Test CLI JSON output via programmatic API."""
        data = _make_data(warmup_ttft=500.0, steady_ttft=100.0)
        result = detect_cold_start(data)
        # Validate JSON serialization roundtrip
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["has_cold_start"] is True
        assert len(parsed["metrics"]) == 3

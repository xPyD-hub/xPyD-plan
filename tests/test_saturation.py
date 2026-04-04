"""Tests for saturation point detection."""

from __future__ import annotations

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.saturation import SaturationDetector, detect_saturation


def _make_benchmark(
    qps: float,
    num_prefill: int = 2,
    num_decode: int = 2,
    num_requests: int = 100,
    base_ttft: float = 50.0,
    base_tpot: float = 10.0,
    base_total: float = 200.0,
    ttft_scale: float = 1.0,
    tpot_scale: float = 1.0,
    total_scale: float = 1.0,
) -> BenchmarkData:
    """Create a synthetic benchmark with controllable latency scaling."""
    import random

    random.seed(42)
    requests = []
    for i in range(num_requests):
        noise = random.uniform(0.8, 1.2)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{qps}-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=base_ttft * ttft_scale * noise,
                tpot_ms=base_tpot * tpot_scale * noise,
                total_latency_ms=base_total * total_scale * noise,
                timestamp=1000.0 + i * (1.0 / qps),
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=qps,
        ),
        requests=requests,
    )


class TestSaturationDetector:
    """Tests for SaturationDetector."""

    def test_init_default(self) -> None:
        detector = SaturationDetector()
        assert detector.increase_threshold == 0.5

    def test_init_custom_threshold(self) -> None:
        detector = SaturationDetector(increase_threshold=0.3)
        assert detector.increase_threshold == 0.3

    def test_init_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match="increase_threshold must be > 0"):
            SaturationDetector(increase_threshold=0)
        with pytest.raises(ValueError, match="increase_threshold must be > 0"):
            SaturationDetector(increase_threshold=-0.1)

    def test_too_few_benchmarks(self) -> None:
        detector = SaturationDetector()
        bench = _make_benchmark(qps=10.0)
        with pytest.raises(ValueError, match="at least 2"):
            detector.analyze([bench])

    def test_same_qps(self) -> None:
        detector = SaturationDetector()
        b1 = _make_benchmark(qps=10.0)
        b2 = _make_benchmark(qps=10.0)
        with pytest.raises(ValueError, match="at least 2 distinct"):
            detector.analyze([b1, b2])

    def test_no_saturation(self) -> None:
        """When latency stays stable across QPS levels, no saturation detected."""
        detector = SaturationDetector()
        benchmarks = [
            _make_benchmark(qps=10.0, ttft_scale=1.0, tpot_scale=1.0, total_scale=1.0),
            _make_benchmark(qps=20.0, ttft_scale=1.05, tpot_scale=1.05, total_scale=1.05),
            _make_benchmark(qps=30.0, ttft_scale=1.1, tpot_scale=1.1, total_scale=1.1),
        ]
        report = detector.analyze(benchmarks)
        assert len(report.thresholds) == 0
        assert report.overall_safe_qps is None
        assert "No saturation" in report.recommendation

    def test_saturation_detected(self) -> None:
        """When latency spikes, saturation is detected."""
        detector = SaturationDetector(increase_threshold=0.5)
        benchmarks = [
            _make_benchmark(qps=10.0, ttft_scale=1.0, tpot_scale=1.0, total_scale=1.0),
            _make_benchmark(qps=20.0, ttft_scale=1.2, tpot_scale=1.2, total_scale=1.2),
            _make_benchmark(qps=30.0, ttft_scale=3.0, tpot_scale=3.0, total_scale=3.0),
        ]
        report = detector.analyze(benchmarks)
        assert len(report.thresholds) > 0
        assert report.overall_safe_qps == 20.0
        # All thresholds should have saturated_qps == 30.0
        for t in report.thresholds:
            assert t.saturated_qps == 30.0
            assert t.safe_qps == 20.0
            assert t.increase_pct > 0

    def test_points_sorted_by_qps(self) -> None:
        """Points should be sorted by QPS regardless of input order."""
        detector = SaturationDetector()
        benchmarks = [
            _make_benchmark(qps=30.0),
            _make_benchmark(qps=10.0),
            _make_benchmark(qps=20.0),
        ]
        report = detector.analyze(benchmarks)
        qps_values = [p.measured_qps for p in report.points]
        assert qps_values == sorted(qps_values)

    def test_report_has_correct_point_count(self) -> None:
        detector = SaturationDetector()
        benchmarks = [
            _make_benchmark(qps=5.0),
            _make_benchmark(qps=15.0),
            _make_benchmark(qps=25.0),
            _make_benchmark(qps=35.0),
        ]
        report = detector.analyze(benchmarks)
        assert len(report.points) == 4

    def test_early_saturation(self) -> None:
        """Saturation at the first QPS transition."""
        detector = SaturationDetector(increase_threshold=0.3)
        benchmarks = [
            _make_benchmark(qps=5.0, ttft_scale=1.0),
            _make_benchmark(qps=10.0, ttft_scale=2.0),
        ]
        report = detector.analyze(benchmarks)
        assert len(report.thresholds) > 0
        assert report.overall_safe_qps == 5.0

    def test_partial_metric_saturation(self) -> None:
        """Only some metrics saturate."""
        detector = SaturationDetector(increase_threshold=0.5)
        benchmarks = [
            _make_benchmark(qps=10.0, ttft_scale=1.0, tpot_scale=1.0, total_scale=1.0),
            _make_benchmark(qps=20.0, ttft_scale=2.0, tpot_scale=1.1, total_scale=1.1),
        ]
        report = detector.analyze(benchmarks)
        # TTFT should saturate, TPOT and total should not
        saturated_metrics = {t.metric for t in report.thresholds}
        assert "ttft_p95_ms" in saturated_metrics
        assert "tpot_p95_ms" not in saturated_metrics

    def test_overall_safe_qps_is_minimum(self) -> None:
        """Overall safe QPS is the minimum across all saturated metrics."""
        detector = SaturationDetector(increase_threshold=0.5)
        benchmarks = [
            _make_benchmark(qps=10.0, ttft_scale=1.0, tpot_scale=1.0, total_scale=1.0),
            _make_benchmark(qps=20.0, ttft_scale=2.0, tpot_scale=2.0, total_scale=2.0),
        ]
        report = detector.analyze(benchmarks)
        if report.thresholds:
            min_safe = min(t.safe_qps for t in report.thresholds)
            assert report.overall_safe_qps == min_safe

    def test_recommendation_includes_safe_qps(self) -> None:
        detector = SaturationDetector(increase_threshold=0.5)
        benchmarks = [
            _make_benchmark(qps=10.0, ttft_scale=1.0),
            _make_benchmark(qps=20.0, ttft_scale=3.0),
        ]
        report = detector.analyze(benchmarks)
        assert "10.0" in report.recommendation

    def test_point_request_count(self) -> None:
        detector = SaturationDetector()
        benchmarks = [
            _make_benchmark(qps=10.0, num_requests=50),
            _make_benchmark(qps=20.0, num_requests=200),
        ]
        report = detector.analyze(benchmarks)
        assert report.points[0].request_count == 50
        assert report.points[1].request_count == 200

    def test_point_pd_config(self) -> None:
        detector = SaturationDetector()
        benchmarks = [
            _make_benchmark(qps=10.0, num_prefill=3, num_decode=5),
            _make_benchmark(qps=20.0, num_prefill=3, num_decode=5),
        ]
        report = detector.analyze(benchmarks)
        for p in report.points:
            assert p.num_prefill == 3
            assert p.num_decode == 5

    def test_high_threshold_no_saturation(self) -> None:
        """With very high threshold, moderate increases don't trigger."""
        detector = SaturationDetector(increase_threshold=5.0)
        benchmarks = [
            _make_benchmark(qps=10.0, ttft_scale=1.0),
            _make_benchmark(qps=20.0, ttft_scale=3.0),
        ]
        report = detector.analyze(benchmarks)
        assert len(report.thresholds) == 0

    def test_low_threshold_sensitive(self) -> None:
        """Low threshold detects even minor increases."""
        detector = SaturationDetector(increase_threshold=0.1)
        benchmarks = [
            _make_benchmark(qps=10.0, ttft_scale=1.0),
            _make_benchmark(qps=20.0, ttft_scale=1.15),
        ]
        report = detector.analyze(benchmarks)
        assert len(report.thresholds) > 0

    def test_increase_pct_correct(self) -> None:
        """Verify the reported increase percentage is approximately correct."""
        detector = SaturationDetector(increase_threshold=0.5)
        benchmarks = [
            _make_benchmark(qps=10.0, ttft_scale=1.0, tpot_scale=1.0, total_scale=1.0),
            _make_benchmark(qps=20.0, ttft_scale=2.0, tpot_scale=1.0, total_scale=1.0),
        ]
        report = detector.analyze(benchmarks)
        ttft_thresholds = [t for t in report.thresholds if "ttft" in t.metric]
        assert len(ttft_thresholds) > 0
        for t in ttft_thresholds:
            # ~100% increase (2x)
            assert t.increase_pct > 50


class TestDetectSaturationAPI:
    """Tests for the programmatic API."""

    def test_returns_dict(self) -> None:
        benchmarks = [
            _make_benchmark(qps=10.0),
            _make_benchmark(qps=20.0),
        ]
        result = detect_saturation(benchmarks)
        assert isinstance(result, dict)
        assert "points" in result
        assert "thresholds" in result
        assert "overall_safe_qps" in result
        assert "recommendation" in result

    def test_custom_threshold(self) -> None:
        benchmarks = [
            _make_benchmark(qps=10.0, ttft_scale=1.0),
            _make_benchmark(qps=20.0, ttft_scale=2.0),
        ]
        result = detect_saturation(benchmarks, increase_threshold=0.5)
        assert len(result["thresholds"]) > 0

    def test_matches_class_output(self) -> None:
        benchmarks = [
            _make_benchmark(qps=10.0),
            _make_benchmark(qps=20.0),
            _make_benchmark(qps=30.0),
        ]
        detector = SaturationDetector()
        report = detector.analyze(benchmarks)
        api_result = detect_saturation(benchmarks)
        assert api_result == report.model_dump()

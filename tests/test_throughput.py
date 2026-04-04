"""Tests for throughput percentile analysis (M60)."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.throughput import (
    ThroughputAnalyzer,
    ThroughputStability,
    analyze_throughput,
)


def _make_requests(
    count: int,
    start_ts: float = 1000.0,
    interval: float = 0.1,
    total_latency_ms: float = 50.0,
) -> list[BenchmarkRequest]:
    """Create evenly spaced requests."""
    return [
        BenchmarkRequest(
            request_id=f"req-{i}",
            prompt_tokens=100,
            output_tokens=50,
            ttft_ms=20.0,
            tpot_ms=5.0,
            total_latency_ms=total_latency_ms,
            timestamp=start_ts + i * interval,
        )
        for i in range(count)
    ]


def _make_data(requests: list[BenchmarkRequest], qps: float = 10.0) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=qps,
        ),
        requests=requests,
    )


class TestThroughputAnalyzer:
    """Test ThroughputAnalyzer class."""

    def test_single_request(self) -> None:
        reqs = _make_requests(1)
        data = _make_data(reqs, qps=1.0)
        analyzer = ThroughputAnalyzer()
        report = analyzer.analyze(data)
        assert report.total_requests == 1

    def test_uniform_requests_stable(self) -> None:
        """Evenly spaced requests over 10 seconds should be stable."""
        # 100 requests, one every 0.1s = 10 req/s over 10s
        reqs = _make_requests(100, interval=0.1, total_latency_ms=0.0)
        data = _make_data(reqs, qps=10.0)
        analyzer = ThroughputAnalyzer(bucket_size=1.0)
        report = analyzer.analyze(data)

        assert report.total_requests == 100
        assert report.stats.stability == ThroughputStability.STABLE
        assert report.stats.mean_rps > 0
        assert report.stats.cv <= 0.15

    def test_bursty_requests_unstable(self) -> None:
        """Requests clustered in bursts should be variable/unstable."""
        reqs = []
        # Burst: 50 requests in first 0.5s, then 10 requests over next 9.5s
        for i in range(50):
            reqs.append(
                BenchmarkRequest(
                    request_id=f"burst-{i}",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=20.0,
                    tpot_ms=5.0,
                    total_latency_ms=0.0,
                    timestamp=1000.0 + i * 0.01,
                )
            )
        for i in range(10):
            reqs.append(
                BenchmarkRequest(
                    request_id=f"tail-{i}",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=20.0,
                    tpot_ms=5.0,
                    total_latency_ms=0.0,
                    timestamp=1001.0 + i * 1.0,
                )
            )
        data = _make_data(reqs)
        analyzer = ThroughputAnalyzer(bucket_size=1.0)
        report = analyzer.analyze(data)
        assert report.stats.stability in (
            ThroughputStability.VARIABLE,
            ThroughputStability.UNSTABLE,
        )
        assert report.stats.cv > 0.15

    def test_bucket_size_custom(self) -> None:
        reqs = _make_requests(100, interval=0.1, total_latency_ms=0.0)
        data = _make_data(reqs, qps=10.0)
        analyzer = ThroughputAnalyzer(bucket_size=2.0)
        report = analyzer.analyze(data)
        assert report.bucket_size == 2.0
        # Fewer buckets with larger bucket size
        assert report.total_buckets <= 6  # ~10s / 2s = 5 buckets

    def test_invalid_bucket_size(self) -> None:
        with pytest.raises(ValueError, match="bucket_size must be positive"):
            ThroughputAnalyzer(bucket_size=0)

    def test_negative_bucket_size(self) -> None:
        with pytest.raises(ValueError, match="bucket_size must be positive"):
            ThroughputAnalyzer(bucket_size=-1.0)

    def test_stats_ordering(self) -> None:
        """min <= p5 <= p50 <= mean (approximately) <= p95 <= p99 <= max."""
        reqs = _make_requests(200, interval=0.05, total_latency_ms=0.0)
        data = _make_data(reqs)
        analyzer = ThroughputAnalyzer(bucket_size=1.0)
        report = analyzer.analyze(data)
        s = report.stats
        assert s.min_rps <= s.p5_rps
        assert s.p5_rps <= s.p50_rps
        assert s.p50_rps <= s.p95_rps
        assert s.p95_rps <= s.p99_rps
        assert s.p99_rps <= s.max_rps

    def test_bottleneck_buckets_below_p5(self) -> None:
        """Bottleneck buckets should have throughput below sustainable (P5)."""
        reqs = _make_requests(100, interval=0.1, total_latency_ms=0.0)
        data = _make_data(reqs)
        analyzer = ThroughputAnalyzer(bucket_size=1.0)
        report = analyzer.analyze(data)
        for b in report.bottleneck_buckets:
            assert b.request_count / report.bucket_size < report.sustainable_rps

    def test_sustainable_rps_equals_p5(self) -> None:
        reqs = _make_requests(100, interval=0.1, total_latency_ms=0.0)
        data = _make_data(reqs)
        analyzer = ThroughputAnalyzer()
        report = analyzer.analyze(data)
        assert report.sustainable_rps == report.stats.p5_rps

    def test_report_model_dump(self) -> None:
        reqs = _make_requests(50, interval=0.1, total_latency_ms=0.0)
        data = _make_data(reqs)
        analyzer = ThroughputAnalyzer()
        report = analyzer.analyze(data)
        d = report.model_dump()
        assert "stats" in d
        assert "bottleneck_buckets" in d
        assert "recommendation" in d
        assert d["stats"]["stability"] in ("STABLE", "VARIABLE", "UNSTABLE")

    def test_latency_affects_completion_time(self) -> None:
        """Requests with high latency complete later, shifting throughput."""
        # All requests start at same time but have different latencies
        reqs = []
        for i in range(20):
            reqs.append(
                BenchmarkRequest(
                    request_id=f"req-{i}",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=20.0,
                    tpot_ms=5.0,
                    total_latency_ms=i * 500.0,  # 0ms to 9500ms
                    timestamp=1000.0,
                )
            )
        data = _make_data(reqs)
        analyzer = ThroughputAnalyzer(bucket_size=1.0)
        report = analyzer.analyze(data)
        # Should have multiple buckets since completions are spread out
        assert report.total_buckets > 1

    def test_all_same_timestamp(self) -> None:
        """All requests at same timestamp with same latency."""
        reqs = [
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=20.0,
                tpot_ms=5.0,
                total_latency_ms=100.0,
                timestamp=1000.0,
            )
            for i in range(10)
        ]
        data = _make_data(reqs)
        analyzer = ThroughputAnalyzer()
        report = analyzer.analyze(data)
        assert report.total_buckets == 1
        assert report.total_requests == 10

    def test_recommendation_contains_cv(self) -> None:
        reqs = _make_requests(100, interval=0.1, total_latency_ms=0.0)
        data = _make_data(reqs)
        analyzer = ThroughputAnalyzer()
        report = analyzer.analyze(data)
        assert "CV=" in report.recommendation

    def test_duration_seconds(self) -> None:
        reqs = _make_requests(100, interval=0.1, total_latency_ms=0.0)
        data = _make_data(reqs)
        analyzer = ThroughputAnalyzer()
        report = analyzer.analyze(data)
        # 100 requests at 0.1s interval = ~9.9s
        assert 9.0 < report.duration_seconds < 10.5

    def test_small_bucket_size(self) -> None:
        """Small bucket size creates more buckets."""
        reqs = _make_requests(100, interval=0.1, total_latency_ms=0.0)
        data = _make_data(reqs)
        analyzer = ThroughputAnalyzer(bucket_size=0.5)
        report = analyzer.analyze(data)
        assert report.total_buckets > 10  # ~9.9s / 0.5s = ~20 buckets

    def test_cv_classification_stable(self) -> None:
        reqs = _make_requests(100, interval=0.1, total_latency_ms=0.0)
        data = _make_data(reqs)
        analyzer = ThroughputAnalyzer()
        report = analyzer.analyze(data)
        if report.stats.cv <= 0.15:
            assert report.stats.stability == ThroughputStability.STABLE

    def test_cv_classification_boundaries(self) -> None:
        """Verify stability classification thresholds in the model."""
        # This is more of a unit test on the enum values
        assert ThroughputStability.STABLE.value == "STABLE"
        assert ThroughputStability.VARIABLE.value == "VARIABLE"
        assert ThroughputStability.UNSTABLE.value == "UNSTABLE"


class TestAnalyzeThroughputAPI:
    """Test the programmatic API."""

    def test_analyze_throughput_returns_dict(self, tmp_path) -> None:
        reqs = _make_requests(50, interval=0.1, total_latency_ms=0.0)
        data = _make_data(reqs)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))
        result = analyze_throughput(str(path))
        assert isinstance(result, dict)
        assert "stats" in result
        assert "sustainable_rps" in result

    def test_analyze_throughput_custom_bucket(self, tmp_path) -> None:
        reqs = _make_requests(50, interval=0.1, total_latency_ms=0.0)
        data = _make_data(reqs)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))
        result = analyze_throughput(str(path), bucket_size=2.0)
        assert result["bucket_size"] == 2.0

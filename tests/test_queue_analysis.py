"""Tests for request queuing time analysis (M62)."""

from __future__ import annotations

import json

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.queue_analysis import (
    CongestionLevel,
    QueueAnalyzer,
    analyze_queue,
)


def _make_requests(
    count: int,
    start_ts: float = 1000.0,
    interval: float = 0.1,
    total_latency_ms: float = 50.0,
    ttft_ms: float = 20.0,
    tpot_ms: float = 5.0,
) -> list[BenchmarkRequest]:
    """Create evenly spaced requests."""
    return [
        BenchmarkRequest(
            request_id=f"req-{i}",
            prompt_tokens=100,
            output_tokens=50,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=total_latency_ms,
            timestamp=start_ts + i * interval,
        )
        for i in range(count)
    ]


def _make_data(
    requests: list[BenchmarkRequest],
    qps: float = 10.0,
    prefill: int = 2,
    decode: int = 2,
) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=prefill,
            num_decode_instances=decode,
            total_instances=prefill + decode,
            measured_qps=qps,
        ),
        requests=requests,
    )


class TestQueueAnalyzer:
    """Test QueueAnalyzer class."""

    def test_single_request(self) -> None:
        reqs = _make_requests(1)
        data = _make_data(reqs, qps=1.0)
        analyzer = QueueAnalyzer()
        report = analyzer.analyze(data)
        assert report.total_requests == 1
        assert report.queue_stats.min_queue_ms == 0.0
        assert report.queue_stats.max_queue_ms == 0.0

    def test_uniform_low_concurrency(self) -> None:
        """Evenly spaced requests with low concurrency should have minimal queue."""
        reqs = _make_requests(20, interval=1.0, total_latency_ms=50.0)
        data = _make_data(reqs, qps=1.0)
        analyzer = QueueAnalyzer()
        report = analyzer.analyze(data)
        assert report.congestion_level == CongestionLevel.LOW
        assert report.queue_stats.queue_ratio < 0.1

    def test_burst_creates_queuing(self) -> None:
        """Many requests at the same time should create queue delays."""
        # 20 requests all starting at same timestamp, varying latency
        reqs = []
        for i in range(20):
            reqs.append(
                BenchmarkRequest(
                    request_id=f"req-{i}",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=20.0,
                    tpot_ms=5.0,
                    total_latency_ms=50.0 + i * 20.0,  # Later requests take longer (queued)
                    timestamp=1000.0,
                )
            )
        data = _make_data(reqs, qps=20.0)
        analyzer = QueueAnalyzer()
        report = analyzer.analyze(data)
        assert report.concurrency.peak_concurrency >= 15
        assert report.queue_stats.max_queue_ms > 0

    def test_peak_concurrency(self) -> None:
        """Peak concurrency should reflect overlapping requests."""
        reqs = _make_requests(10, interval=0.01, total_latency_ms=100.0)
        data = _make_data(reqs, qps=100.0)
        analyzer = QueueAnalyzer()
        report = analyzer.analyze(data)
        assert report.concurrency.peak_concurrency >= 5

    def test_concurrency_profile_points(self) -> None:
        reqs = _make_requests(50, interval=0.1)
        data = _make_data(reqs, qps=10.0)
        analyzer = QueueAnalyzer()
        report = analyzer.analyze(data)
        assert len(report.concurrency.points) > 0

    def test_max_profile_points(self) -> None:
        reqs = _make_requests(500, interval=0.01)
        data = _make_data(reqs, qps=100.0)
        analyzer = QueueAnalyzer(max_profile_points=50)
        report = analyzer.analyze(data)
        assert len(report.concurrency.points) <= 55  # Allow slight overshoot from step

    def test_queue_ratio_bounded(self) -> None:
        reqs = _make_requests(10)
        data = _make_data(reqs, qps=10.0)
        analyzer = QueueAnalyzer()
        report = analyzer.analyze(data)
        assert 0.0 <= report.queue_stats.queue_ratio <= 1.0

    def test_queue_stats_ordering(self) -> None:
        """Percentile ordering: min <= p50 <= p95 <= p99 <= max."""
        reqs = _make_requests(100, interval=0.05, total_latency_ms=80.0)
        data = _make_data(reqs, qps=20.0)
        analyzer = QueueAnalyzer()
        report = analyzer.analyze(data)
        q = report.queue_stats
        assert q.min_queue_ms <= q.p50_queue_ms
        assert q.p50_queue_ms <= q.p95_queue_ms
        assert q.p95_queue_ms <= q.p99_queue_ms
        assert q.p99_queue_ms <= q.max_queue_ms

    def test_duration_calculation(self) -> None:
        reqs = _make_requests(10, start_ts=1000.0, interval=1.0)
        data = _make_data(reqs, qps=1.0)
        analyzer = QueueAnalyzer()
        report = analyzer.analyze(data)
        assert abs(report.duration_seconds - 9.0) < 0.01

    def test_congestion_levels(self) -> None:
        """Different queue ratios should yield different congestion levels."""
        # LOW: well-spaced requests
        reqs = _make_requests(10, interval=2.0, total_latency_ms=50.0)
        data = _make_data(reqs, qps=0.5)
        report = QueueAnalyzer().analyze(data)
        assert report.congestion_level == CongestionLevel.LOW

    def test_recommendation_not_empty(self) -> None:
        reqs = _make_requests(10)
        data = _make_data(reqs, qps=10.0)
        report = QueueAnalyzer().analyze(data)
        assert len(report.recommendation) > 0

    def test_mean_concurrency_positive(self) -> None:
        reqs = _make_requests(20, interval=0.05, total_latency_ms=100.0)
        data = _make_data(reqs, qps=20.0)
        report = QueueAnalyzer().analyze(data)
        assert report.concurrency.mean_concurrency > 0

    def test_all_same_latency_no_queue(self) -> None:
        """If all requests have identical latency and are well-spaced, no queuing."""
        reqs = _make_requests(10, interval=1.0, total_latency_ms=50.0)
        data = _make_data(reqs, qps=1.0)
        report = QueueAnalyzer().analyze(data)
        assert report.queue_stats.mean_queue_ms == 0.0

    def test_mixed_latency_requests(self) -> None:
        """Requests with varying latency."""
        reqs = []
        for i in range(20):
            reqs.append(
                BenchmarkRequest(
                    request_id=f"req-{i}",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=20.0,
                    tpot_ms=5.0,
                    total_latency_ms=30.0 + (i % 5) * 40.0,
                    timestamp=1000.0 + i * 0.5,
                )
            )
        data = _make_data(reqs, qps=2.0)
        report = QueueAnalyzer().analyze(data)
        assert report.total_requests == 20

    def test_report_model_dump(self) -> None:
        """Report should be JSON serializable."""
        reqs = _make_requests(10)
        data = _make_data(reqs, qps=10.0)
        report = QueueAnalyzer().analyze(data)
        d = report.model_dump()
        assert "queue_stats" in d
        assert "concurrency" in d
        assert "congestion_level" in d
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_two_requests_same_time(self) -> None:
        reqs = [
            BenchmarkRequest(
                request_id="a",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=20.0,
                tpot_ms=5.0,
                total_latency_ms=100.0,
                timestamp=1000.0,
            ),
            BenchmarkRequest(
                request_id="b",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=20.0,
                tpot_ms=5.0,
                total_latency_ms=200.0,
                timestamp=1000.0,
            ),
        ]
        data = _make_data(reqs, qps=2.0)
        report = QueueAnalyzer().analyze(data)
        assert report.concurrency.peak_concurrency == 2

    def test_high_concurrency_detection(self) -> None:
        """Overlapping requests creating high concurrency."""
        reqs = _make_requests(50, interval=0.01, total_latency_ms=500.0)
        data = _make_data(reqs, qps=100.0)
        report = QueueAnalyzer().analyze(data)
        assert report.concurrency.peak_concurrency >= 20

    def test_concurrency_percentiles_ordered(self) -> None:
        reqs = _make_requests(100, interval=0.05, total_latency_ms=100.0)
        data = _make_data(reqs, qps=20.0)
        report = QueueAnalyzer().analyze(data)
        c = report.concurrency
        assert c.p50_concurrency <= c.p95_concurrency
        assert c.p95_concurrency <= c.p99_concurrency
        assert c.p99_concurrency <= c.peak_concurrency


class TestAnalyzeQueueAPI:
    """Test programmatic analyze_queue() API."""

    def test_analyze_queue_returns_dict(self, tmp_path) -> None:
        """analyze_queue() should return a dict."""
        reqs = _make_requests(10)
        data = _make_data(reqs, qps=10.0)
        path = tmp_path / "bench.json"
        path.write_text(data.model_dump_json(indent=2))

        result = analyze_queue(str(path))
        assert isinstance(result, dict)
        assert "queue_stats" in result
        assert "concurrency" in result
        assert "congestion_level" in result

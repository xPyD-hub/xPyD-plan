"""Tests for batch_analysis module."""

from __future__ import annotations

import json
import tempfile

import pytest

from xpyd_plan.batch_analysis import (
    BatchAnalyzer,
    BatchEfficiency,
    analyze_batch_impact,
)
from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest


def _make_request(
    request_id: str,
    timestamp: float,
    prompt_tokens: int = 100,
    output_tokens: int = 50,
    ttft_ms: float = 20.0,
    tpot_ms: float = 10.0,
    total_latency_ms: float = 200.0,
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


def _make_benchmark(requests: list[BenchmarkRequest], qps: float = 10.0) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=qps,
        ),
        requests=requests if requests else [_make_request("dummy", 0.0)],
    )


def _make_benchmark_file(data: BenchmarkData) -> str:
    """Write benchmark data to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data.model_dump(), f, default=str)
    f.close()
    return f.name


class TestBatchAnalyzerInit:
    def test_default_window(self):
        analyzer = BatchAnalyzer()
        assert analyzer.window_ms == 100.0

    def test_custom_window(self):
        analyzer = BatchAnalyzer(window_ms=50.0)
        assert analyzer.window_ms == 50.0

    def test_invalid_window_zero(self):
        with pytest.raises(ValueError, match="positive"):
            BatchAnalyzer(window_ms=0)

    def test_invalid_window_negative(self):
        with pytest.raises(ValueError, match="positive"):
            BatchAnalyzer(window_ms=-10)


class TestBatchAnalyzerSingleRequest:
    def test_single_request(self):
        req = _make_request("r1", timestamp=1000.0)
        data = _make_benchmark([req])
        analyzer = BatchAnalyzer()
        report = analyzer.analyze(data)
        assert report.total_requests == 1
        assert report.total_batches == 1
        assert report.min_batch_size == 1
        assert report.max_batch_size == 1
        assert len(report.buckets) == 1
        assert report.buckets[0].batch_size == 1


class TestBatchGrouping:
    def test_two_separate_batches(self):
        """Requests far apart should form separate batches."""
        reqs = [
            _make_request("r1", timestamp=1.0),
            _make_request("r2", timestamp=1.05),  # 50ms apart -> same batch
            _make_request("r3", timestamp=5.0),  # 4950ms apart -> new batch
            _make_request("r4", timestamp=5.03),  # 30ms apart -> same batch
        ]
        data = _make_benchmark(reqs)
        analyzer = BatchAnalyzer(window_ms=100.0)
        report = analyzer.analyze(data)
        assert report.total_batches == 2
        assert report.min_batch_size == 2
        assert report.max_batch_size == 2

    def test_all_one_batch(self):
        """Requests within window should group into one batch."""
        reqs = [
            _make_request("r1", timestamp=1.000),
            _make_request("r2", timestamp=1.010),
            _make_request("r3", timestamp=1.020),
        ]
        data = _make_benchmark(reqs)
        analyzer = BatchAnalyzer(window_ms=50.0)
        report = analyzer.analyze(data)
        assert report.total_batches == 1
        assert report.max_batch_size == 3

    def test_each_request_separate(self):
        """Requests far apart with small window should each be their own batch."""
        reqs = [
            _make_request("r1", timestamp=1.0),
            _make_request("r2", timestamp=2.0),
            _make_request("r3", timestamp=3.0),
        ]
        data = _make_benchmark(reqs)
        analyzer = BatchAnalyzer(window_ms=10.0)  # 10ms window, requests 1s apart
        report = analyzer.analyze(data)
        assert report.total_batches == 3
        assert report.min_batch_size == 1
        assert report.max_batch_size == 1

    def test_mixed_batch_sizes(self):
        """Mix of batch sizes."""
        reqs = [
            # Batch 1: single
            _make_request("r1", timestamp=1.0),
            # Batch 2: triple
            _make_request("r2", timestamp=5.0),
            _make_request("r3", timestamp=5.05),
            _make_request("r4", timestamp=5.08),
            # Batch 3: double
            _make_request("r5", timestamp=10.0),
            _make_request("r6", timestamp=10.05),
        ]
        data = _make_benchmark(reqs)
        analyzer = BatchAnalyzer(window_ms=100.0)
        report = analyzer.analyze(data)
        assert report.total_batches == 3
        assert report.min_batch_size == 1
        assert report.max_batch_size == 3
        # Should have 3 bucket sizes: 1, 2, 3
        sizes = {b.batch_size for b in report.buckets}
        assert sizes == {1, 2, 3}


class TestBatchBucketStats:
    def test_latency_stats_computed(self):
        reqs = [
            _make_request("r1", timestamp=1.0, ttft_ms=10.0, tpot_ms=5.0, total_latency_ms=100.0),
            _make_request("r2", timestamp=1.01, ttft_ms=20.0, tpot_ms=10.0, total_latency_ms=200.0),
        ]
        data = _make_benchmark(reqs)
        analyzer = BatchAnalyzer(window_ms=100.0)
        report = analyzer.analyze(data)
        assert len(report.buckets) == 1
        b = report.buckets[0]
        assert b.batch_size == 2
        assert b.count == 1
        assert b.mean_ttft_ms == pytest.approx(15.0)
        assert b.mean_tpot_ms == pytest.approx(7.5)
        assert b.mean_total_latency_ms == pytest.approx(150.0)

    def test_throughput_calculated(self):
        reqs = [
            _make_request("r1", timestamp=1.0, total_latency_ms=1000.0),
            _make_request("r2", timestamp=1.01, total_latency_ms=1000.0),
        ]
        data = _make_benchmark(reqs)
        analyzer = BatchAnalyzer(window_ms=100.0)
        report = analyzer.analyze(data)
        b = report.buckets[0]
        # 2 requests / (1000ms / 1000) = 2.0 rps
        assert b.throughput_rps == pytest.approx(2.0)


class TestEfficiencyClassification:
    def test_optimal_is_best(self):
        score = BatchAnalyzer._classify_efficiency(1.0, 1.0)
        assert score == BatchEfficiency.OPTIMAL

    def test_good(self):
        score = BatchAnalyzer._classify_efficiency(0.75, 1.0)
        assert score == BatchEfficiency.GOOD

    def test_acceptable(self):
        score = BatchAnalyzer._classify_efficiency(0.5, 1.0)
        assert score == BatchEfficiency.ACCEPTABLE

    def test_poor(self):
        score = BatchAnalyzer._classify_efficiency(0.3, 1.0)
        assert score == BatchEfficiency.POOR

    def test_zero_best_score(self):
        score = BatchAnalyzer._classify_efficiency(0.0, 0.0)
        assert score == BatchEfficiency.ACCEPTABLE


class TestOptimalBatchSize:
    def test_optimal_selection(self):
        """Optimal batch size should have the best throughput/latency ratio."""
        reqs = []
        # Batch size 1 requests: low throughput, low latency
        for i in range(5):
            reqs.append(_make_request(
                f"s{i}", timestamp=float(i * 10),
                total_latency_ms=100.0, ttft_ms=10.0, tpot_ms=5.0,
            ))
        # Batch size 3 requests: higher throughput, moderate latency
        for i in range(3):
            base = 100.0 + i * 10
            for j in range(3):
                reqs.append(_make_request(
                    f"b{i}_{j}", timestamp=base + j * 0.01,
                    total_latency_ms=150.0, ttft_ms=15.0, tpot_ms=8.0,
                ))

        data = _make_benchmark(reqs)
        analyzer = BatchAnalyzer(window_ms=100.0)
        report = analyzer.analyze(data)
        assert report.optimal_batch_size >= 1


class TestBatchReportModel:
    def test_report_serialization(self):
        reqs = [
            _make_request("r1", timestamp=1.0),
            _make_request("r2", timestamp=1.01),
        ]
        data = _make_benchmark(reqs)
        analyzer = BatchAnalyzer()
        report = analyzer.analyze(data)
        d = report.model_dump()
        assert "total_requests" in d
        assert "buckets" in d
        assert "optimal_batch_size" in d
        assert "recommendation" in d

    def test_report_recommendation_text(self):
        reqs = [_make_request("r1", timestamp=1.0)]
        data = _make_benchmark(reqs)
        analyzer = BatchAnalyzer()
        report = analyzer.analyze(data)
        assert "Optimal" in report.recommendation


class TestAnalyzeBatchImpactAPI:
    def test_programmatic_api(self):
        reqs = [
            _make_request("r1", timestamp=1.0),
            _make_request("r2", timestamp=1.05),
        ]
        data = _make_benchmark(reqs)
        path = _make_benchmark_file(data)
        result = analyze_batch_impact(path, window_ms=100.0)
        assert isinstance(result, dict)
        assert result["total_requests"] == 2
        assert "buckets" in result

    def test_programmatic_api_custom_window(self):
        reqs = [
            _make_request("r1", timestamp=1.0),
            _make_request("r2", timestamp=1.5),  # 500ms apart
        ]
        data = _make_benchmark(reqs)
        path = _make_benchmark_file(data)

        # Window 100ms -> 2 batches
        result1 = analyze_batch_impact(path, window_ms=100.0)
        assert result1["total_batches"] == 2

        # Window 1000ms -> 1 batch
        result2 = analyze_batch_impact(path, window_ms=1000.0)
        assert result2["total_batches"] == 1


class TestBatchAnalyzerUnsorted:
    def test_unsorted_timestamps(self):
        """Analyzer should handle unsorted input."""
        reqs = [
            _make_request("r3", timestamp=5.0),
            _make_request("r1", timestamp=1.0),
            _make_request("r2", timestamp=1.05),
        ]
        data = _make_benchmark(reqs)
        analyzer = BatchAnalyzer(window_ms=100.0)
        report = analyzer.analyze(data)
        assert report.total_batches == 2

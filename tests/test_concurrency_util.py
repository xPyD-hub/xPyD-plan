"""Tests for concurrency utilization analysis."""

from __future__ import annotations

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.concurrency_util import (
    ConcurrencyUtilizationAnalyzer,
    UtilizationLevel,
    analyze_concurrency_util,
)


def _make_data(
    n: int = 100,
    seed: int = 42,
    total_instances: int = 4,
    prefill: int = 2,
    decode: int = 2,
    qps: float = 50.0,
    spread: float = 10.0,
) -> BenchmarkData:
    """Create benchmark data with n requests spread over time."""
    import random as _random

    rng = _random.Random(seed)
    requests = []
    base_ts = 1700000000.0
    for i in range(n):
        pt = rng.randint(50, 2000)
        ot = rng.randint(10, 500)
        ttft = pt * 0.05 + rng.uniform(5, 30)
        tpot = ot * 0.03 + rng.uniform(2, 15)
        total = ttft + tpot * ot + rng.uniform(10, 200)
        ts = base_ts + (i / qps) * spread + rng.uniform(0, 0.1)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=pt,
                output_tokens=ot,
                ttft_ms=ttft,
                tpot_ms=tpot,
                total_latency_ms=total,
                timestamp=ts,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=prefill,
            num_decode_instances=decode,
            total_instances=total_instances,
            measured_qps=qps,
        ),
        requests=requests,
    )


class TestConcurrencyUtilizationAnalyzer:
    def test_basic_analyze(self) -> None:
        data = _make_data(100)
        analyzer = ConcurrencyUtilizationAnalyzer(window_size=2.0)
        report = analyzer.analyze(data)
        assert report.total_requests == 100
        assert report.total_instances == 4
        assert len(report.windows) > 0
        assert report.peak_concurrent >= 1

    def test_window_count(self) -> None:
        data = _make_data(50)
        analyzer = ConcurrencyUtilizationAnalyzer(window_size=1.0)
        report = analyzer.analyze(data)
        assert len(report.windows) >= 1

    def test_utilization_levels(self) -> None:
        data = _make_data(100)
        analyzer = ConcurrencyUtilizationAnalyzer(window_size=2.0)
        report = analyzer.analyze(data)
        for w in report.windows:
            assert isinstance(w.level, UtilizationLevel)
            if w.utilization_pct < 20:
                assert w.level == UtilizationLevel.IDLE
            elif w.utilization_pct < 50:
                assert w.level == UtilizationLevel.LOW
            elif w.utilization_pct < 80:
                assert w.level == UtilizationLevel.MODERATE
            else:
                assert w.level == UtilizationLevel.HIGH

    def test_idle_and_high_counts(self) -> None:
        data = _make_data(100)
        analyzer = ConcurrencyUtilizationAnalyzer(window_size=2.0)
        report = analyzer.analyze(data)
        idle = sum(1 for w in report.windows if w.level == UtilizationLevel.IDLE)
        high = sum(1 for w in report.windows if w.level == UtilizationLevel.HIGH)
        assert report.idle_windows == idle
        assert report.high_windows == high

    def test_avg_utilization_range(self) -> None:
        data = _make_data(100)
        report = analyze_concurrency_util(data, window_size=1.0)
        assert 0 <= report.avg_utilization_pct <= 100

    def test_recommendation_present(self) -> None:
        data = _make_data(100)
        report = analyze_concurrency_util(data)
        assert report.recommendation is not None
        rec = report.recommendation
        assert rec.current_instances == 4
        assert rec.recommended_min >= 2
        assert rec.recommended_target >= 2

    def test_over_provisioned_detection(self) -> None:
        # Very few requests with many instances → over-provisioned
        data = _make_data(5, total_instances=100, prefill=50, decode=50, qps=1.0)
        report = analyze_concurrency_util(data, window_size=1.0)
        assert report.recommendation is not None
        assert report.recommendation.over_provisioned is True

    def test_small_window_size(self) -> None:
        data = _make_data(50)
        analyzer = ConcurrencyUtilizationAnalyzer(window_size=0.5)
        report = analyzer.analyze(data)
        assert len(report.windows) >= 1

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError, match="window_size must be > 0"):
            ConcurrencyUtilizationAnalyzer(window_size=0)

    def test_invalid_window_size_negative(self) -> None:
        with pytest.raises(ValueError, match="window_size must be > 0"):
            ConcurrencyUtilizationAnalyzer(window_size=-1.0)

    def test_window_fields(self) -> None:
        data = _make_data(100)
        report = analyze_concurrency_util(data, window_size=2.0)
        for w in report.windows:
            assert w.window_end > w.window_start
            assert w.concurrent_requests >= 0
            assert w.avg_concurrent >= 0
            assert 0 <= w.utilization_pct <= 100
            assert w.requests_started >= 0
            assert w.requests_completed >= 0

    def test_duration(self) -> None:
        data = _make_data(100)
        report = analyze_concurrency_util(data)
        assert report.duration_seconds > 0

    def test_programmatic_api(self) -> None:
        data = _make_data(100)
        report = analyze_concurrency_util(data, window_size=2.0)
        assert isinstance(report.total_requests, int)
        assert report.total_requests == 100

    def test_single_request(self) -> None:
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=1.0,
            ),
            requests=[
                BenchmarkRequest(
                    request_id="req-0",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=10.0,
                    tpot_ms=5.0,
                    total_latency_ms=200.0,
                    timestamp=1700000000.0,
                )
            ],
        )
        report = analyze_concurrency_util(data, window_size=1.0)
        assert report.total_requests == 1
        assert report.peak_concurrent >= 1

    def test_json_serialization(self) -> None:
        data = _make_data(50)
        report = analyze_concurrency_util(data)
        d = report.model_dump()
        assert "windows" in d
        assert "recommendation" in d
        assert isinstance(d["windows"], list)

    def test_large_window(self) -> None:
        data = _make_data(50)
        report = analyze_concurrency_util(data, window_size=100.0)
        # Should produce a single window
        assert len(report.windows) >= 1

    def test_many_requests_high_util(self) -> None:
        # Many overlapping requests → should show high utilization
        requests = []
        base_ts = 1700000000.0
        for i in range(200):
            requests.append(
                BenchmarkRequest(
                    request_id=f"req-{i}",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=10.0,
                    tpot_ms=5.0,
                    total_latency_ms=5000.0,  # 5 seconds each
                    timestamp=base_ts + i * 0.01,  # very close together
                )
            )
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=100.0,
            ),
            requests=requests,
        )
        report = analyze_concurrency_util(data, window_size=1.0)
        assert report.peak_concurrent > 10
        assert report.high_windows > 0

    def test_recommendation_min_instances(self) -> None:
        data = _make_data(10, total_instances=2, prefill=1, decode=1)
        report = analyze_concurrency_util(data)
        assert report.recommendation is not None
        assert report.recommendation.recommended_min >= 2
        assert report.recommendation.recommended_target >= 2

    def test_window_size_property(self) -> None:
        data = _make_data(50)
        report = analyze_concurrency_util(data, window_size=3.0)
        assert report.window_size_seconds == 3.0

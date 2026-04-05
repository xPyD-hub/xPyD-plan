"""Tests for arrival pattern analysis."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

from xpyd_plan.arrival_pattern import (
    ArrivalPattern,
    ArrivalPatternAnalyzer,
    BurstInfo,
    InterArrivalStats,
    analyze_arrival_pattern,
)
from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.cli import main


def _make_requests(
    n: int = 100,
    base_ts: float = 1000.0,
    interval_s: float = 0.5,
    jitter: float = 0.0,
    seed: int = 42,
) -> list[BenchmarkRequest]:
    """Generate requests with configurable inter-arrival times."""
    rng = np.random.default_rng(seed)
    requests = []
    ts = base_ts
    for i in range(n):
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i:04d}",
                prompt_tokens=100 + i,
                output_tokens=50 + i,
                ttft_ms=10.0 + rng.normal(0, 1),
                tpot_ms=5.0 + rng.normal(0, 0.5),
                total_latency_ms=100.0 + rng.normal(0, 5),
                timestamp=ts,
            )
        )
        ts += interval_s + jitter * rng.random()
    return requests


def _make_bursty_requests(
    n_bursts: int = 10,
    burst_size: int = 10,
    burst_gap_s: float = 5.0,
    intra_burst_gap_s: float = 0.01,
    seed: int = 42,
) -> list[BenchmarkRequest]:
    """Generate bursty arrival pattern."""
    rng = np.random.default_rng(seed)
    requests = []
    ts = 1000.0
    idx = 0
    for _ in range(n_bursts):
        for _ in range(burst_size):
            requests.append(
                BenchmarkRequest(
                    request_id=f"req-{idx:04d}",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=10.0,
                    tpot_ms=5.0,
                    total_latency_ms=100.0,
                    timestamp=ts,
                )
            )
            ts += intra_burst_gap_s + rng.random() * 0.005
            idx += 1
        ts += burst_gap_s
    return requests


def _make_data(requests: list[BenchmarkRequest]) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=6,
            total_instances=8,
            measured_qps=50.0,
        ),
        requests=requests,
    )


class TestArrivalPatternAnalyzer:
    def test_uniform_pattern(self):
        """Evenly spaced requests → uniform."""
        reqs = _make_requests(200, interval_s=0.5, jitter=0.0)
        data = _make_data(reqs)
        analyzer = ArrivalPatternAnalyzer()
        report = analyzer.analyze(data)

        assert report.pattern == ArrivalPattern.UNIFORM
        assert report.confidence > 0.5
        assert report.inter_arrival.cv < 0.3

    def test_bursty_pattern(self):
        """Bursty requests → bursty."""
        reqs = _make_bursty_requests(n_bursts=10, burst_size=15, burst_gap_s=5.0)
        data = _make_data(reqs)
        analyzer = ArrivalPatternAnalyzer()
        report = analyzer.analyze(data)

        assert report.pattern == ArrivalPattern.BURSTY
        assert report.burst_info is not None
        assert report.burst_info.burst_count > 0

    def test_poisson_like_pattern(self):
        """Exponential inter-arrivals → Poisson."""
        rng = np.random.default_rng(42)
        ts = 1000.0
        reqs = []
        for i in range(500):
            reqs.append(
                BenchmarkRequest(
                    request_id=f"req-{i:04d}",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=10.0,
                    tpot_ms=5.0,
                    total_latency_ms=100.0,
                    timestamp=ts,
                )
            )
            ts += rng.exponential(scale=0.5)
        data = _make_data(reqs)
        analyzer = ArrivalPatternAnalyzer()
        report = analyzer.analyze(data)

        assert report.pattern == ArrivalPattern.POISSON
        assert report.poisson_fit_p_value is not None
        assert report.inter_arrival.cv > 0.5

    def test_small_dataset(self):
        """<2 requests → unknown."""
        reqs = _make_requests(1)
        data = _make_data(reqs)
        analyzer = ArrivalPatternAnalyzer()
        report = analyzer.analyze(data)

        assert report.pattern == ArrivalPattern.UNKNOWN
        assert report.confidence == 0.0
        assert report.request_count == 1

    def test_two_requests(self):
        """Two requests should still work."""
        reqs = _make_requests(2, interval_s=1.0)
        data = _make_data(reqs)
        analyzer = ArrivalPatternAnalyzer()
        report = analyzer.analyze(data)

        assert report.request_count == 2
        assert report.inter_arrival.count == 1

    def test_inter_arrival_stats(self):
        """Stats should be computed correctly."""
        reqs = _make_requests(100, interval_s=0.5, jitter=0.0)
        data = _make_data(reqs)
        analyzer = ArrivalPatternAnalyzer()
        report = analyzer.analyze(data)

        ia = report.inter_arrival
        assert ia.count == 99
        assert ia.mean_ms > 0
        assert ia.p50_ms > 0
        assert ia.min_ms <= ia.p50_ms <= ia.p95_ms <= ia.max_ms

    def test_burst_threshold_validation(self):
        """Invalid burst threshold should raise."""
        import pytest

        with pytest.raises(ValueError):
            ArrivalPatternAnalyzer(burst_threshold=0.0)
        with pytest.raises(ValueError):
            ArrivalPatternAnalyzer(burst_threshold=1.0)
        with pytest.raises(ValueError):
            ArrivalPatternAnalyzer(burst_threshold=-0.1)

    def test_report_json_serializable(self):
        """Report should be JSON-serializable."""
        reqs = _make_requests(100, interval_s=0.5)
        data = _make_data(reqs)
        analyzer = ArrivalPatternAnalyzer()
        report = analyzer.analyze(data)

        dumped = report.model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)
        assert "pattern" in parsed
        assert "inter_arrival" in parsed

    def test_burst_info_model(self):
        """BurstInfo model fields."""
        bi = BurstInfo(burst_count=5, avg_burst_size=10.0, burst_fraction=0.5)
        assert bi.burst_count == 5
        assert bi.avg_burst_size == 10.0

    def test_inter_arrival_stats_model(self):
        """InterArrivalStats model fields."""
        ias = InterArrivalStats(
            count=99,
            mean_ms=500.0,
            std_ms=50.0,
            min_ms=400.0,
            p50_ms=500.0,
            p95_ms=580.0,
            p99_ms=600.0,
            max_ms=650.0,
            cv=0.1,
        )
        assert ias.count == 99

    def test_duration_computed(self):
        """Duration should reflect timestamp span."""
        reqs = _make_requests(100, interval_s=1.0, jitter=0.0)
        data = _make_data(reqs)
        analyzer = ArrivalPatternAnalyzer()
        report = analyzer.analyze(data)

        assert report.duration_s > 90.0

    def test_no_burst_when_uniform(self):
        """Uniform arrivals should have no bursts."""
        reqs = _make_requests(100, interval_s=0.5, jitter=0.0)
        data = _make_data(reqs)
        analyzer = ArrivalPatternAnalyzer()
        report = analyzer.analyze(data)

        assert report.burst_info is None

    def test_custom_burst_threshold(self):
        """Custom burst threshold should affect detection."""
        reqs = _make_bursty_requests(n_bursts=5, burst_size=10)
        data = _make_data(reqs)

        # Strict threshold
        analyzer1 = ArrivalPatternAnalyzer(burst_threshold=0.05)
        report1 = analyzer1.analyze(data)

        # Loose threshold
        analyzer2 = ArrivalPatternAnalyzer(burst_threshold=0.5)
        report2 = analyzer2.analyze(data)

        # Both should detect bursts but may differ in details
        assert report1.inter_arrival.count == report2.inter_arrival.count

    def test_measured_qps_preserved(self):
        """Report should preserve original measured QPS."""
        reqs = _make_requests(50)
        data = _make_data(reqs)
        analyzer = ArrivalPatternAnalyzer()
        report = analyzer.analyze(data)

        assert report.measured_qps == 50.0

    def test_identical_timestamps(self):
        """All identical timestamps should handle gracefully."""
        reqs = []
        for i in range(50):
            reqs.append(
                BenchmarkRequest(
                    request_id=f"req-{i:04d}",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=10.0,
                    tpot_ms=5.0,
                    total_latency_ms=100.0,
                    timestamp=1000.0,  # all same
                )
            )
        data = _make_data(reqs)
        analyzer = ArrivalPatternAnalyzer()
        report = analyzer.analyze(data)

        assert report.request_count == 50
        assert report.duration_s == 0.0


class TestAnalyzeArrivalPattern:
    def test_programmatic_api(self, tmp_path):
        reqs = _make_requests(100, interval_s=0.5)
        data = _make_data(reqs)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        result = analyze_arrival_pattern(str(path))

        assert isinstance(result, dict)
        assert "pattern" in result
        assert "inter_arrival" in result
        assert "request_count" in result


class TestCLIArrivalPattern:
    def test_cli_table_output(self, tmp_path, capsys):
        reqs = _make_requests(100, interval_s=0.5)
        data = _make_data(reqs)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        with patch("sys.argv", [
            "xpyd-plan", "arrival-pattern",
            "--benchmark", str(path),
        ]):
            main()

        captured = capsys.readouterr()
        assert "Arrival Pattern" in captured.out

    def test_cli_json_output(self, tmp_path, capsys):
        reqs = _make_requests(100, interval_s=0.5)
        data = _make_data(reqs)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        with patch("sys.argv", [
            "xpyd-plan", "arrival-pattern",
            "--benchmark", str(path),
            "--output-format", "json",
        ]):
            main()

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "pattern" in result

    def test_cli_custom_threshold(self, tmp_path, capsys):
        reqs = _make_requests(100, interval_s=0.5)
        data = _make_data(reqs)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        with patch("sys.argv", [
            "xpyd-plan", "arrival-pattern",
            "--benchmark", str(path),
            "--burst-threshold", "0.2",
            "--output-format", "json",
        ]):
            main()

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "pattern" in result

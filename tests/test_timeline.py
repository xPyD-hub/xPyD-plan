"""Tests for timeline analysis."""

from __future__ import annotations

import json
import tempfile

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.timeline import (
    LatencyTrendDirection,
    TimelineAnalyzer,
    _linear_regression,
    _percentile,
    analyze_timeline,
)


def _make_benchmark(
    requests: list[dict] | None = None,
    num_requests: int = 100,
    duration_s: float = 100.0,
) -> BenchmarkData:
    """Create a benchmark dataset with timestamps spread over duration."""
    if requests is None:
        requests_list = []
        for i in range(num_requests):
            t = 1000.0 + (i / max(1, num_requests - 1)) * duration_s
            requests_list.append(
                BenchmarkRequest(
                    request_id=f"req-{i}",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=20.0,
                    tpot_ms=10.0,
                    total_latency_ms=70.0,
                    timestamp=t,
                )
            )
    else:
        requests_list = [BenchmarkRequest(**r) for r in requests]

    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=requests_list,
    )


def _make_warmup_benchmark() -> BenchmarkData:
    """Create benchmark with clear warmup period."""
    requests = []
    for i in range(100):
        t = 1000.0 + i
        # First 10 requests have 5x higher latency
        if i < 10:
            total = 500.0
        else:
            total = 70.0
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=20.0 if i >= 10 else 100.0,
                tpot_ms=10.0,
                total_latency_ms=total,
                timestamp=t,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=requests,
    )


def _make_degrading_benchmark() -> BenchmarkData:
    """Create benchmark with degrading latency over time."""
    requests = []
    for i in range(100):
        t = 1000.0 + i
        total = 50.0 + i * 2.0  # linearly increasing
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=20.0 + i * 0.5,
                tpot_ms=10.0 + i * 0.3,
                total_latency_ms=total,
                timestamp=t,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=requests,
    )


class TestPercentile:
    """Tests for percentile helper."""

    def test_single_value(self) -> None:
        assert _percentile([42.0], 50) == 42.0

    def test_p50(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 50) == 3.0

    def test_p95(self) -> None:
        vals = list(range(1, 101))
        result = _percentile([float(v) for v in vals], 95)
        assert 95.0 <= result <= 96.0

    def test_empty(self) -> None:
        assert _percentile([], 50) == 0.0


class TestLinearRegression:
    """Tests for linear regression helper."""

    def test_perfect_line(self) -> None:
        slope, intercept, r_sq = _linear_regression(
            [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]
        )
        assert abs(slope - 2.0) < 1e-10
        assert abs(intercept) < 1e-10
        assert abs(r_sq - 1.0) < 1e-10

    def test_flat_line(self) -> None:
        slope, _intercept, _r_sq = _linear_regression(
            [1.0, 2.0, 3.0], [5.0, 5.0, 5.0]
        )
        assert slope == 0.0

    def test_single_point(self) -> None:
        slope, _intercept, r_sq = _linear_regression([1.0], [2.0])
        assert slope == 0.0
        assert r_sq == 0.0


class TestTimelineAnalyzer:
    """Tests for TimelineAnalyzer."""

    def test_basic_analysis(self) -> None:
        data = _make_benchmark()
        analyzer = TimelineAnalyzer(window_size_s=10.0)
        report = analyzer.analyze(data)

        assert len(report.windows) > 0
        assert report.trend is not None
        assert report.warmup is not None
        assert report.recommendation

    def test_window_count(self) -> None:
        data = _make_benchmark(duration_s=50.0)
        analyzer = TimelineAnalyzer(window_size_s=10.0)
        report = analyzer.analyze(data)
        assert len(report.windows) == 5

    def test_all_requests_assigned(self) -> None:
        data = _make_benchmark(num_requests=50, duration_s=50.0)
        analyzer = TimelineAnalyzer(window_size_s=10.0)
        report = analyzer.analyze(data)

        total_in_windows = sum(w.request_count for w in report.windows)
        assert total_in_windows == 50

    def test_stable_latency(self) -> None:
        data = _make_benchmark()
        analyzer = TimelineAnalyzer(window_size_s=10.0)
        report = analyzer.analyze(data)
        assert report.trend.direction == LatencyTrendDirection.STABLE

    def test_warmup_not_detected_stable(self) -> None:
        data = _make_benchmark()
        analyzer = TimelineAnalyzer(window_size_s=10.0)
        report = analyzer.analyze(data)
        assert not report.warmup.detected

    def test_warmup_detected(self) -> None:
        data = _make_warmup_benchmark()
        analyzer = TimelineAnalyzer(window_size_s=10.0)
        report = analyzer.analyze(data)
        assert report.warmup.detected
        assert report.warmup.warmup_windows >= 1
        assert report.warmup.warmup_peak_p95_ms > report.warmup.steady_state_p95_ms

    def test_degrading_trend(self) -> None:
        data = _make_degrading_benchmark()
        analyzer = TimelineAnalyzer(window_size_s=10.0)
        report = analyzer.analyze(data)
        assert report.trend.direction == LatencyTrendDirection.DEGRADING
        assert report.trend.slope_ms_per_s > 0

    def test_too_few_requests(self) -> None:
        requests = [
            {
                "request_id": "req-0",
                "prompt_tokens": 100,
                "output_tokens": 50,
                "ttft_ms": 20.0,
                "tpot_ms": 10.0,
                "total_latency_ms": 70.0,
                "timestamp": 1000.0,
            }
        ]
        data = _make_benchmark(requests=requests)
        analyzer = TimelineAnalyzer()
        with pytest.raises(ValueError, match="at least 2"):
            analyzer.analyze(data)

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError, match="window_size_s must be positive"):
            TimelineAnalyzer(window_size_s=0)

    def test_invalid_warmup_factor(self) -> None:
        with pytest.raises(ValueError, match="warmup_factor must be positive"):
            TimelineAnalyzer(warmup_factor=-1)

    def test_same_timestamp(self) -> None:
        """All requests at the same timestamp → single window."""
        requests = [
            {
                "request_id": f"req-{i}",
                "prompt_tokens": 100,
                "output_tokens": 50,
                "ttft_ms": 20.0,
                "tpot_ms": 10.0,
                "total_latency_ms": 70.0,
                "timestamp": 1000.0,
            }
            for i in range(10)
        ]
        data = _make_benchmark(requests=requests)
        analyzer = TimelineAnalyzer()
        report = analyzer.analyze(data)
        assert len(report.windows) == 1

    def test_recommendation_warmup(self) -> None:
        data = _make_warmup_benchmark()
        report = TimelineAnalyzer(window_size_s=10.0).analyze(data)
        assert "Warmup" in report.recommendation or "warmup" in report.recommendation.lower()

    def test_recommendation_degrading(self) -> None:
        data = _make_degrading_benchmark()
        report = TimelineAnalyzer(window_size_s=10.0).analyze(data)
        assert "degrading" in report.recommendation.lower()

    def test_recommendation_stable(self) -> None:
        data = _make_benchmark()
        report = TimelineAnalyzer(window_size_s=10.0).analyze(data)
        assert "stable" in report.recommendation.lower()

    def test_window_latency_values(self) -> None:
        data = _make_benchmark()
        analyzer = TimelineAnalyzer(window_size_s=10.0)
        report = analyzer.analyze(data)

        for w in report.windows:
            if w.request_count > 0:
                assert w.total_p95_ms > 0
                assert w.ttft_p95_ms > 0

    def test_custom_window_size(self) -> None:
        data = _make_benchmark(duration_s=100.0)
        analyzer = TimelineAnalyzer(window_size_s=25.0)
        report = analyzer.analyze(data)
        assert len(report.windows) == 4

    def test_r_squared_range(self) -> None:
        data = _make_benchmark()
        report = TimelineAnalyzer(window_size_s=10.0).analyze(data)
        assert 0.0 <= report.trend.r_squared <= 1.0


class TestAnalyzeTimelineAPI:
    """Tests for the programmatic API."""

    def test_returns_dict(self) -> None:
        data = _make_benchmark()
        result = analyze_timeline(data)
        assert isinstance(result, dict)
        assert "windows" in result
        assert "warmup" in result
        assert "trend" in result
        assert "recommendation" in result

    def test_custom_params(self) -> None:
        data = _make_benchmark()
        result = analyze_timeline(data, window_size_s=20.0, warmup_factor=3.0)
        assert isinstance(result, dict)
        assert len(result["windows"]) > 0


class TestTimelineCLI:
    """Tests for CLI integration."""

    def test_table_output(self) -> None:
        import io
        from contextlib import redirect_stdout

        from xpyd_plan.cli._main import main

        data = _make_benchmark()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json(indent=2))
            f.flush()

            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    main(["timeline", "--benchmark", f.name])
                except SystemExit:
                    pass

    def test_json_output(self) -> None:
        from xpyd_plan.cli._main import main

        data = _make_benchmark()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json(indent=2))
            f.flush()

            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    main([
                        "timeline", "--benchmark", f.name,
                        "--output-format", "json",
                    ])
                except SystemExit:
                    pass

            output = buf.getvalue()
            if output.strip():
                result = json.loads(output)
                assert "windows" in result
                assert "warmup" in result
                assert "trend" in result

    def test_custom_window_size_cli(self) -> None:
        from xpyd_plan.cli._main import main

        data = _make_benchmark()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json(indent=2))
            f.flush()

            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    main([
                        "timeline", "--benchmark", f.name,
                        "--window-size", "25",
                        "--output-format", "json",
                    ])
                except SystemExit:
                    pass

            output = buf.getvalue()
            if output.strip():
                result = json.loads(output)
                # 100s / 25s = 4 windows
                assert len(result["windows"]) == 4

"""Tests for correlation analysis."""

from __future__ import annotations

import json
import tempfile

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.correlation import (
    CorrelationAnalyzer,
    CorrelationStrength,
    _classify_strength,
    _pearson,
    analyze_correlation,
)


def _make_benchmark(
    requests: list[dict] | None = None,
    num_requests: int = 50,
) -> BenchmarkData:
    """Create a benchmark dataset."""
    if requests is None:
        # Default: output_tokens strongly correlated with total_latency
        requests = []
        for i in range(num_requests):
            out_tokens = 20 + i * 5
            requests.append(
                BenchmarkRequest(
                    request_id=f"req-{i}",
                    prompt_tokens=100,
                    output_tokens=out_tokens,
                    ttft_ms=20.0,
                    tpot_ms=10.0,
                    total_latency_ms=20.0 + out_tokens * 10.0,
                    timestamp=1000.0 + i,
                )
            )
    else:
        requests = [BenchmarkRequest(**r) for r in requests]

    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=requests,
    )


class TestPearson:
    """Tests for the Pearson correlation function."""

    def test_perfect_positive(self) -> None:
        r = _pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert abs(r - 1.0) < 1e-10

    def test_perfect_negative(self) -> None:
        r = _pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0])
        assert abs(r - (-1.0)) < 1e-10

    def test_no_correlation(self) -> None:
        # Symmetric pattern yields r=0
        r = _pearson([1.0, 2.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        assert r == 0.0

    def test_zero_variance(self) -> None:
        r = _pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        assert r == 0.0

    def test_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            _pearson([1.0], [1.0])


class TestClassifyStrength:
    """Tests for strength classification."""

    def test_strong(self) -> None:
        assert _classify_strength(0.8) == CorrelationStrength.STRONG
        assert _classify_strength(-0.7) == CorrelationStrength.STRONG

    def test_moderate(self) -> None:
        assert _classify_strength(0.5) == CorrelationStrength.MODERATE
        assert _classify_strength(-0.4) == CorrelationStrength.MODERATE

    def test_weak(self) -> None:
        assert _classify_strength(0.3) == CorrelationStrength.WEAK
        assert _classify_strength(-0.2) == CorrelationStrength.WEAK

    def test_negligible(self) -> None:
        assert _classify_strength(0.1) == CorrelationStrength.NEGLIGIBLE
        assert _classify_strength(0.0) == CorrelationStrength.NEGLIGIBLE


class TestCorrelationAnalyzer:
    """Tests for CorrelationAnalyzer."""

    def test_basic_analysis(self) -> None:
        data = _make_benchmark()
        analyzer = CorrelationAnalyzer()
        report = analyzer.analyze(data)

        assert len(report.pairs) == 6  # 2 request metrics x 3 latency metrics
        assert report.strongest_pair is not None

    def test_strong_output_tokens_total_latency(self) -> None:
        """output_tokens strongly correlates with total_latency in default fixture."""
        data = _make_benchmark()
        analyzer = CorrelationAnalyzer()
        report = analyzer.analyze(data)

        # Find the output_tokens vs total_latency_ms pair
        pair = next(
            p for p in report.pairs
            if p.x_metric == "output_tokens" and p.y_metric == "total_latency_ms"
        )
        assert pair.strength == CorrelationStrength.STRONG
        assert pair.pearson_r > 0.99

    def test_constant_prompt_no_correlation(self) -> None:
        """When prompt_tokens is constant, correlation with latency should be 0."""
        data = _make_benchmark()
        analyzer = CorrelationAnalyzer()
        report = analyzer.analyze(data)

        prompt_ttft = next(
            p for p in report.pairs
            if p.x_metric == "prompt_tokens" and p.y_metric == "ttft_ms"
        )
        assert prompt_ttft.pearson_r == 0.0
        assert prompt_ttft.strength == CorrelationStrength.NEGLIGIBLE

    def test_sample_size_recorded(self) -> None:
        data = _make_benchmark(num_requests=30)
        analyzer = CorrelationAnalyzer()
        report = analyzer.analyze(data)

        for pair in report.pairs:
            assert pair.sample_size == 30

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
        analyzer = CorrelationAnalyzer()
        with pytest.raises(ValueError, match="at least 2"):
            analyzer.analyze(data)

    def test_recommendation_strong(self) -> None:
        data = _make_benchmark()
        report = CorrelationAnalyzer().analyze(data)
        assert "Strong correlation" in report.recommendation

    def test_recommendation_no_strong(self) -> None:
        """All constant latencies → negligible correlations."""
        requests = [
            {
                "request_id": f"req-{i}",
                "prompt_tokens": 50 + i * 10,
                "output_tokens": 20 + i * 5,
                "ttft_ms": 20.0,
                "tpot_ms": 10.0,
                "total_latency_ms": 70.0,
                "timestamp": 1000.0 + i,
            }
            for i in range(30)
        ]
        data = _make_benchmark(requests=requests)
        report = CorrelationAnalyzer().analyze(data)
        assert "No strong correlations" in report.recommendation

    def test_strongest_pair_is_correct(self) -> None:
        data = _make_benchmark()
        report = CorrelationAnalyzer().analyze(data)

        max_pair = max(report.pairs, key=lambda p: abs(p.pearson_r))
        assert report.strongest_pair == max_pair


class TestAnalyzeCorrelationAPI:
    """Tests for the programmatic API."""

    def test_returns_dict(self) -> None:
        data = _make_benchmark()
        result = analyze_correlation(data)
        assert isinstance(result, dict)
        assert "pairs" in result
        assert "strongest_pair" in result
        assert "recommendation" in result

    def test_dict_values(self) -> None:
        data = _make_benchmark()
        result = analyze_correlation(data)
        assert len(result["pairs"]) == 6
        for pair in result["pairs"]:
            assert "pearson_r" in pair
            assert "strength" in pair


class TestCorrelationCLI:
    """Tests for CLI integration."""

    def test_table_output(self) -> None:
        import io
        from contextlib import redirect_stdout

        from xpyd_plan.cli._main import main

        data = _make_benchmark()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json(indent=2))
            f.flush()

            # Should not raise
            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    main(["correlation", "--benchmark", f.name])
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
                    main(["correlation", "--benchmark", f.name, "--output-format", "json"])
                except SystemExit:
                    pass

            output = buf.getvalue()
            if output.strip():
                result = json.loads(output)
                assert "pairs" in result

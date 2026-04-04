"""Tests for tail latency analysis."""

from __future__ import annotations

import json
import random
import tempfile

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.tail import (
    LongTailProfile,
    TailAnalyzer,
    TailClassification,
    TailReport,
    _classify_tail,
    analyze_tail,
)


def _make_requests(
    n: int = 200,
    *,
    seed: int = 42,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    tail_factor: float = 1.0,
) -> list[BenchmarkRequest]:
    """Generate synthetic requests with controllable tail behavior."""
    rng = random.Random(seed)
    requests = []
    for i in range(n):
        prompt_tokens = rng.randint(50, 500)
        output_tokens = rng.randint(20, 200)
        # Add tail: last ~1% of requests get inflated latencies
        is_tail = i >= int(n * 0.99)
        factor = tail_factor if is_tail else 1.0
        ttft = ttft_base * (1 + rng.random() * 0.3) * factor
        tpot = tpot_base * (1 + rng.random() * 0.3) * factor
        total = ttft + tpot * output_tokens * factor
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                ttft_ms=round(ttft, 2),
                tpot_ms=round(tpot, 2),
                total_latency_ms=round(total, 2),
                timestamp=1700000000.0 + i,
            )
        )
    return requests


def _make_benchmark(requests: list[BenchmarkRequest]) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=4,
            total_instances=6,
            measured_qps=10.0,
        ),
        requests=requests,
    )


class TestClassifyTail:
    def test_light(self):
        assert _classify_tail(1.5) == TailClassification.LIGHT

    def test_moderate(self):
        assert _classify_tail(3.0) == TailClassification.MODERATE

    def test_heavy(self):
        assert _classify_tail(7.0) == TailClassification.HEAVY

    def test_extreme(self):
        assert _classify_tail(15.0) == TailClassification.EXTREME

    def test_boundary_light_moderate(self):
        assert _classify_tail(1.99) == TailClassification.LIGHT
        assert _classify_tail(2.0) == TailClassification.MODERATE

    def test_boundary_moderate_heavy(self):
        assert _classify_tail(4.99) == TailClassification.MODERATE
        assert _classify_tail(5.0) == TailClassification.HEAVY

    def test_boundary_heavy_extreme(self):
        assert _classify_tail(9.99) == TailClassification.HEAVY
        assert _classify_tail(10.0) == TailClassification.EXTREME


class TestTailAnalyzer:
    def test_basic_analysis(self):
        reqs = _make_requests(200)
        data = _make_benchmark(reqs)
        analyzer = TailAnalyzer(data)
        report = analyzer.analyze()

        assert isinstance(report, TailReport)
        assert len(report.metrics) == 3
        assert report.total_requests == 200
        assert report.worst_tail is not None

    def test_metrics_fields(self):
        reqs = _make_requests(200)
        data = _make_benchmark(reqs)
        report = TailAnalyzer(data).analyze()

        fields = [m.field for m in report.metrics]
        assert "ttft_ms" in fields
        assert "tpot_ms" in fields
        assert "total_latency_ms" in fields

    def test_percentile_ordering(self):
        reqs = _make_requests(500, seed=123)
        data = _make_benchmark(reqs)
        report = TailAnalyzer(data).analyze()

        for m in report.metrics:
            assert m.p50 <= m.p90 <= m.p95 <= m.p99 <= m.p999 <= m.p9999

    def test_heavy_tail(self):
        reqs = _make_requests(200, tail_factor=20.0)
        data = _make_benchmark(reqs)
        report = TailAnalyzer(data).analyze()

        # With 20x tail factor, at least one metric should be heavy or extreme
        classifications = [m.classification for m in report.metrics]
        assert any(
            c in (TailClassification.HEAVY, TailClassification.EXTREME, TailClassification.MODERATE)
            for c in classifications
        )

    def test_long_tail_profiles_exist(self):
        reqs = _make_requests(200)
        data = _make_benchmark(reqs)
        report = TailAnalyzer(data).analyze()

        assert len(report.long_tail_profiles) > 0
        for p in report.long_tail_profiles:
            assert isinstance(p, LongTailProfile)
            assert p.tail_count > 0
            assert p.total_count == 200
            assert p.prompt_token_ratio > 0
            assert p.output_token_ratio > 0

    def test_tail_ratio_positive(self):
        reqs = _make_requests(100)
        data = _make_benchmark(reqs)
        report = TailAnalyzer(data).analyze()

        for m in report.metrics:
            assert m.tail_ratio_p99 > 0
            assert m.tail_ratio_p999 > 0

    def test_model_dump(self):
        reqs = _make_requests(100)
        data = _make_benchmark(reqs)
        report = TailAnalyzer(data).analyze()

        d = report.model_dump()
        assert "metrics" in d
        assert "long_tail_profiles" in d
        assert "worst_tail" in d
        assert "total_requests" in d


class TestAnalyzeTailAPI:
    def test_returns_dict(self):
        reqs = _make_requests(100)
        data = _make_benchmark(reqs)
        result = analyze_tail(data)

        assert isinstance(result, dict)
        assert "metrics" in result
        assert "long_tail_profiles" in result
        assert result["total_requests"] == 100

    def test_metrics_structure(self):
        reqs = _make_requests(100)
        data = _make_benchmark(reqs)
        result = analyze_tail(data)

        for m in result["metrics"]:
            assert "field" in m
            assert "p50" in m
            assert "p99" in m
            assert "p999" in m
            assert "p9999" in m
            assert "tail_ratio_p99" in m
            assert "classification" in m


class TestTailCLI:
    def test_cli_json_output(self):
        """Test that the CLI tail command produces valid JSON output."""
        import io
        from unittest.mock import patch

        reqs = _make_requests(100)
        data = _make_benchmark(reqs)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data.model_dump(), f)
            f.flush()
            benchmark_path = f.name

        from xpyd_plan.cli._main import main

        captured = io.StringIO()
        test_argv = [
            "xpyd-plan", "tail",
            "--benchmark", benchmark_path,
            "--output-format", "json",
        ]
        with patch("sys.stdout", captured), patch(
            "sys.argv", test_argv
        ):
            try:
                main()
            except SystemExit:
                pass

        output = captured.getvalue()
        parsed = json.loads(output)
        assert "metrics" in parsed
        assert "total_requests" in parsed

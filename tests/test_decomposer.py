"""Tests for latency decomposition analysis (M50)."""

from __future__ import annotations

import json
import tempfile

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.decomposer import (
    BottleneckType,
    LatencyDecomposer,
    _classify_bottleneck,
    _decompose_request,
    decompose_latency,
)


def _make_request(
    request_id: str = "r1",
    prompt_tokens: int = 100,
    output_tokens: int = 50,
    ttft_ms: float = 100.0,
    tpot_ms: float = 10.0,
    total_latency_ms: float = 700.0,
    timestamp: float = 1000.0,
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


def _make_benchmark(requests: list[BenchmarkRequest]) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=requests,
    )


class TestDecomposeRequest:
    """Unit tests for _decompose_request."""

    def test_basic_decomposition(self):
        d = _decompose_request(
            "r1", ttft_ms=100, tpot_ms=10, output_tokens=50, total_latency_ms=700
        )
        assert d.prefill_ms == 100.0
        assert d.decode_ms == 500.0  # 50 * 10
        assert d.overhead_ms == 100.0  # 700 - 100 - 500
        assert abs(d.prefill_fraction + d.decode_fraction + d.overhead_fraction - 1.0) < 1e-9

    def test_negative_overhead_clamped_to_zero(self):
        # total < prefill + decode → overhead should be 0
        d = _decompose_request(
            "r1", ttft_ms=100, tpot_ms=10, output_tokens=50, total_latency_ms=500
        )
        assert d.overhead_ms == 0.0
        assert d.overhead_fraction == 0.0
        # Fractions should still sum to 1
        assert abs(d.prefill_fraction + d.decode_fraction + d.overhead_fraction - 1.0) < 1e-9

    def test_zero_total_latency(self):
        d = _decompose_request("r1", ttft_ms=0, tpot_ms=0, output_tokens=1, total_latency_ms=0)
        assert d.prefill_fraction == 0.0
        assert d.decode_fraction == 0.0
        assert d.overhead_fraction == 0.0

    def test_all_prefill(self):
        d = _decompose_request(
            "r1", ttft_ms=1000, tpot_ms=0, output_tokens=1, total_latency_ms=1000
        )
        assert d.prefill_fraction == 1.0
        assert d.decode_fraction == 0.0
        assert d.overhead_fraction == 0.0


class TestClassifyBottleneck:
    """Tests for _classify_bottleneck."""

    def test_prefill_bound(self):
        assert _classify_bottleneck(0.6, 0.3, 0.1, 0.5) == BottleneckType.PREFILL_BOUND

    def test_decode_bound(self):
        assert _classify_bottleneck(0.2, 0.7, 0.1, 0.5) == BottleneckType.DECODE_BOUND

    def test_overhead_bound(self):
        assert _classify_bottleneck(0.1, 0.2, 0.7, 0.5) == BottleneckType.OVERHEAD_BOUND

    def test_balanced(self):
        assert _classify_bottleneck(0.35, 0.35, 0.3, 0.5) == BottleneckType.BALANCED

    def test_custom_threshold(self):
        assert _classify_bottleneck(0.45, 0.35, 0.2, 0.4) == BottleneckType.PREFILL_BOUND


class TestLatencyDecomposer:
    """Tests for LatencyDecomposer class."""

    def test_prefill_heavy_benchmark(self):
        """Requests where TTFT dominates → PREFILL_BOUND."""
        requests = [
            _make_request(
                request_id=f"r{i}",
                ttft_ms=800.0,
                tpot_ms=1.0,
                output_tokens=10,
                total_latency_ms=820.0,
            )
            for i in range(20)
        ]
        data = _make_benchmark(requests)
        decomposer = LatencyDecomposer(data)
        report = decomposer.analyze()

        assert report.bottleneck == BottleneckType.PREFILL_BOUND
        assert report.total_requests == 20
        assert len(report.phase_stats) == 3
        assert report.phase_stats[0].phase == "prefill"
        assert report.phase_stats[0].mean_fraction > 0.5

    def test_decode_heavy_benchmark(self):
        """Requests where decode dominates → DECODE_BOUND."""
        requests = [
            _make_request(
                request_id=f"r{i}",
                ttft_ms=10.0,
                tpot_ms=20.0,
                output_tokens=100,
                total_latency_ms=2050.0,
            )
            for i in range(20)
        ]
        data = _make_benchmark(requests)
        decomposer = LatencyDecomposer(data)
        report = decomposer.analyze()

        assert report.bottleneck == BottleneckType.DECODE_BOUND

    def test_balanced_benchmark(self):
        """Requests with balanced phases → BALANCED."""
        requests = [
            _make_request(
                request_id=f"r{i}",
                ttft_ms=300.0,
                tpot_ms=10.0,
                output_tokens=30,
                total_latency_ms=700.0,
            )
            for i in range(20)
        ]
        data = _make_benchmark(requests)
        decomposer = LatencyDecomposer(data)
        report = decomposer.analyze()

        assert report.bottleneck == BottleneckType.BALANCED

    def test_overhead_bound(self):
        """Requests with large overhead → OVERHEAD_BOUND."""
        requests = [
            _make_request(
                request_id=f"r{i}",
                ttft_ms=50.0,
                tpot_ms=1.0,
                output_tokens=10,
                total_latency_ms=1000.0,
            )
            for i in range(20)
        ]
        data = _make_benchmark(requests)
        decomposer = LatencyDecomposer(data)
        report = decomposer.analyze()

        assert report.bottleneck == BottleneckType.OVERHEAD_BOUND

    def test_include_requests(self):
        requests = [_make_request(request_id=f"r{i}") for i in range(5)]
        data = _make_benchmark(requests)
        decomposer = LatencyDecomposer(data)

        report_without = decomposer.analyze(include_requests=False)
        assert len(report_without.decomposed_requests) == 0

        report_with = decomposer.analyze(include_requests=True)
        assert len(report_with.decomposed_requests) == 5

    def test_single_request(self):
        data = _make_benchmark([_make_request()])
        decomposer = LatencyDecomposer(data)
        report = decomposer.analyze()
        assert report.total_requests == 1
        assert len(report.phase_stats) == 3

    def test_custom_threshold(self):
        requests = [
            _make_request(
                request_id=f"r{i}",
                ttft_ms=400.0,
                tpot_ms=5.0,
                output_tokens=20,
                total_latency_ms=600.0,
            )
            for i in range(20)
        ]
        data = _make_benchmark(requests)
        # With threshold=0.6, prefill at ~66% should be classified as PREFILL_BOUND
        decomposer = LatencyDecomposer(data, bottleneck_threshold=0.6)
        report = decomposer.analyze()
        assert report.bottleneck == BottleneckType.PREFILL_BOUND
        assert report.bottleneck_threshold == 0.6

    def test_recommendation_not_empty(self):
        data = _make_benchmark([_make_request()])
        decomposer = LatencyDecomposer(data)
        report = decomposer.analyze()
        assert len(report.recommendation) > 0

    def test_phase_stats_structure(self):
        data = _make_benchmark([_make_request(request_id=f"r{i}") for i in range(10)])
        decomposer = LatencyDecomposer(data)
        report = decomposer.analyze()

        phases = {ps.phase for ps in report.phase_stats}
        assert phases == {"prefill", "decode", "overhead"}

        for ps in report.phase_stats:
            assert ps.mean_fraction >= 0
            assert ps.p50_fraction >= 0
            assert ps.p95_fraction >= 0
            assert ps.mean_ms >= 0
            assert ps.p50_ms >= 0
            assert ps.p95_ms >= 0


class TestDecomposeLatencyAPI:
    """Tests for the programmatic decompose_latency() API."""

    def test_returns_dict(self):
        data = _make_benchmark([_make_request()])
        result = decompose_latency(data)
        assert isinstance(result, dict)
        assert "bottleneck" in result
        assert "phase_stats" in result
        assert "total_requests" in result
        assert "recommendation" in result

    def test_with_include_requests(self):
        data = _make_benchmark([_make_request()])
        result = decompose_latency(data, include_requests=True)
        assert len(result["decomposed_requests"]) == 1

    def test_custom_threshold(self):
        data = _make_benchmark([_make_request()])
        result = decompose_latency(data, bottleneck_threshold=0.3)
        assert result["bottleneck_threshold"] == 0.3


class TestCLIDecompose:
    """Tests for CLI decompose subcommand output."""

    def _write_benchmark(self, tmp_dir: str, requests: list[dict] | None = None) -> str:
        import os

        if requests is None:
            requests = [
                {
                    "request_id": f"r{i}",
                    "prompt_tokens": 100,
                    "output_tokens": 50,
                    "ttft_ms": 800.0,
                    "tpot_ms": 1.0,
                    "total_latency_ms": 860.0,
                    "timestamp": 1000.0 + i,
                }
                for i in range(10)
            ]

        benchmark = {
            "metadata": {
                "num_prefill_instances": 2,
                "num_decode_instances": 2,
                "total_instances": 4,
                "measured_qps": 10.0,
            },
            "requests": requests,
        }

        path = os.path.join(tmp_dir, "bench.json")
        with open(path, "w") as f:
            json.dump(benchmark, f)
        return path

    def test_json_output(self):
        from io import StringIO
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_benchmark(tmp)

            from xpyd_plan.cli._main import main

            captured = StringIO()
            with patch("sys.stdout", captured):
                main(["decompose", "--benchmark", path, "--output-format", "json"])

            output = json.loads(captured.getvalue())
            assert output["bottleneck"] == "prefill_bound"
            assert len(output["phase_stats"]) == 3

    def test_table_output_runs(self):
        """Table output should not crash."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_benchmark(tmp)

            from xpyd_plan.cli._main import main

            # Just ensure it doesn't raise
            main(["decompose", "--benchmark", path, "--output-format", "table"])

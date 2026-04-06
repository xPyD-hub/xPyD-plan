"""Tests for backend_compare module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.backend_compare import (
    BackendComparator,
    BackendComparisonConfig,
    BackendComparisonReport,
    BackendFormat,
    BackendMetrics,
    RankCriteria,
    _compute_metrics,
    _detect_format,
    _load_benchmark,
    _rank_backends,
    compare_backends,
)
from xpyd_plan.benchmark_models import BenchmarkData


def _make_benchmark(
    num_requests: int = 20,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    total_base: float = 200.0,
    num_prefill: int = 2,
    num_decode: int = 2,
    total_instances: int = 4,
    measured_qps: float = 10.0,
) -> dict:
    """Create a native benchmark data dict."""
    requests = []
    for i in range(num_requests):
        requests.append(
            {
                "request_id": f"req-{i}",
                "prompt_tokens": 100 + i,
                "output_tokens": 50 + i,
                "ttft_ms": ttft_base + i * 2.0,
                "tpot_ms": tpot_base + i * 0.5,
                "total_latency_ms": total_base + i * 5.0,
                "timestamp": 1000.0 + i * 0.1,
            }
        )
    return {
        "metadata": {
            "num_prefill_instances": num_prefill,
            "num_decode_instances": num_decode,
            "total_instances": total_instances,
            "measured_qps": measured_qps,
        },
        "requests": requests,
    }


def _write_json(data: dict, path: Path) -> str:
    """Write JSON to a temp file and return path string."""
    path.write_text(json.dumps(data))
    return str(path)


class TestDetectFormat:
    """Tests for format auto-detection."""

    def test_detect_native(self):
        data = {"metadata": {}, "requests": []}
        assert _detect_format(data) == BackendFormat.NATIVE

    def test_detect_vllm(self):
        data = [{"request_latency": 1.0, "prompt_len": 10, "output_len": 5}]
        assert _detect_format(data) == BackendFormat.VLLM

    def test_detect_sglang(self):
        data = [
            {"latency": 1.0, "itl": [0.01, 0.02], "success": True, "ttft": 0.1, "prompt_len": 10}
        ]
        assert _detect_format(data) == BackendFormat.SGLANG

    def test_detect_trtllm(self):
        data = [
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "first_token_latency": 50.0,
                "inter_token_latencies": [10.0, 12.0],
                "end_to_end_latency": 200.0,
            }
        ]
        assert _detect_format(data) == BackendFormat.TRTLLM

    def test_detect_empty_list_native(self):
        data = []
        assert _detect_format(data) == BackendFormat.NATIVE


class TestLoadBenchmark:
    """Tests for loading benchmark data."""

    def test_load_native(self, tmp_path):
        data = _make_benchmark()
        path = _write_json(data, tmp_path / "bench.json")
        bd, fmt = _load_benchmark(path, BackendFormat.NATIVE)
        assert fmt == "native"
        assert len(bd.requests) == 20

    def test_load_auto_native(self, tmp_path):
        data = _make_benchmark()
        path = _write_json(data, tmp_path / "bench.json")
        bd, fmt = _load_benchmark(path)
        assert fmt == "native"

    def test_load_invalid_format(self, tmp_path):
        data = _make_benchmark()
        path = _write_json(data, tmp_path / "bench.json")
        with pytest.raises(ValueError, match="Unsupported format"):
            _load_benchmark(path, "invalid_format")


class TestComputeMetrics:
    """Tests for metrics computation."""

    def test_basic_metrics(self):
        data = _make_benchmark(num_requests=10, ttft_base=100.0, tpot_base=20.0)
        bd = BenchmarkData.model_validate(data)
        m = _compute_metrics(bd, "test-backend", "native")
        assert m.backend_label == "test-backend"
        assert m.format_detected == "native"
        assert m.request_count == 10
        assert m.ttft_p50_ms > 0
        assert m.tpot_p50_ms > 0
        assert m.total_latency_p50_ms > 0
        assert m.throughput_rps == 10.0

    def test_metrics_percentiles_ordered(self):
        data = _make_benchmark(num_requests=100)
        bd = BenchmarkData.model_validate(data)
        m = _compute_metrics(bd, "backend", "native")
        assert m.ttft_p50_ms <= m.ttft_p95_ms <= m.ttft_p99_ms
        assert m.tpot_p50_ms <= m.tpot_p95_ms <= m.tpot_p99_ms
        assert m.total_latency_p50_ms <= m.total_latency_p95_ms <= m.total_latency_p99_ms


class TestSLACheck:
    """Tests for SLA compliance checking."""

    def test_sla_all_pass(self, tmp_path):
        data = _make_benchmark(ttft_base=10.0, tpot_base=5.0, total_base=50.0)
        path1 = _write_json(data, tmp_path / "b1.json")
        path2 = _write_json(data, tmp_path / "b2.json")
        result = compare_backends(
            [path1, path2],
            ["a", "b"],
            sla_ttft_p99_ms=999.0,
            sla_tpot_p99_ms=999.0,
            sla_total_latency_p99_ms=9999.0,
        )
        for s in result.sla_results:
            assert s.meets_all is True

    def test_sla_fail(self, tmp_path):
        data = _make_benchmark(ttft_base=100.0)
        path1 = _write_json(data, tmp_path / "b1.json")
        path2 = _write_json(data, tmp_path / "b2.json")
        result = compare_backends(
            [path1, path2],
            ["a", "b"],
            sla_ttft_p99_ms=1.0,  # impossibly low
        )
        for s in result.sla_results:
            assert s.ttft_p99_pass is False
            assert s.meets_all is False

    def test_sla_no_thresholds_all_pass(self, tmp_path):
        data = _make_benchmark()
        path1 = _write_json(data, tmp_path / "b1.json")
        path2 = _write_json(data, tmp_path / "b2.json")
        result = compare_backends([path1, path2], ["a", "b"])
        for s in result.sla_results:
            assert s.meets_all is True


class TestRankBackends:
    """Tests for backend ranking."""

    def test_rank_by_ttft(self):
        m1 = BackendMetrics(
            backend_label="fast",
            format_detected="native",
            num_prefill_instances=1,
            num_decode_instances=1,
            total_instances=2,
            measured_qps=10.0,
            request_count=100,
            ttft_p50_ms=10.0,
            ttft_p95_ms=20.0,
            ttft_p99_ms=30.0,
            tpot_p50_ms=5.0,
            tpot_p95_ms=10.0,
            tpot_p99_ms=15.0,
            total_latency_p50_ms=100.0,
            total_latency_p95_ms=200.0,
            total_latency_p99_ms=300.0,
            throughput_rps=10.0,
            avg_prompt_tokens=100.0,
            avg_output_tokens=50.0,
        )
        m2 = m1.model_copy(
            update={"backend_label": "slow", "ttft_p99_ms": 100.0}
        )
        rankings = _rank_backends([m1, m2], RankCriteria.TTFT_P99)
        assert rankings[0].backend_label == "fast"
        assert rankings[1].backend_label == "slow"
        assert rankings[0].rank == 1
        assert rankings[1].rank == 2

    def test_rank_by_throughput(self):
        m1 = BackendMetrics(
            backend_label="high-tp",
            format_detected="native",
            num_prefill_instances=1,
            num_decode_instances=1,
            total_instances=2,
            measured_qps=100.0,
            request_count=100,
            ttft_p50_ms=10.0,
            ttft_p95_ms=20.0,
            ttft_p99_ms=30.0,
            tpot_p50_ms=5.0,
            tpot_p95_ms=10.0,
            tpot_p99_ms=15.0,
            total_latency_p50_ms=100.0,
            total_latency_p95_ms=200.0,
            total_latency_p99_ms=300.0,
            throughput_rps=100.0,
            avg_prompt_tokens=100.0,
            avg_output_tokens=50.0,
        )
        m2 = m1.model_copy(
            update={"backend_label": "low-tp", "throughput_rps": 10.0, "measured_qps": 10.0}
        )
        rankings = _rank_backends([m1, m2], RankCriteria.THROUGHPUT)
        assert rankings[0].backend_label == "high-tp"

    def test_rank_by_total_latency(self):
        m1 = BackendMetrics(
            backend_label="a",
            format_detected="native",
            num_prefill_instances=1,
            num_decode_instances=1,
            total_instances=2,
            measured_qps=10.0,
            request_count=50,
            ttft_p50_ms=10.0,
            ttft_p95_ms=20.0,
            ttft_p99_ms=30.0,
            tpot_p50_ms=5.0,
            tpot_p95_ms=10.0,
            tpot_p99_ms=15.0,
            total_latency_p50_ms=100.0,
            total_latency_p95_ms=200.0,
            total_latency_p99_ms=300.0,
            throughput_rps=10.0,
            avg_prompt_tokens=100.0,
            avg_output_tokens=50.0,
        )
        m2 = m1.model_copy(
            update={"backend_label": "b", "total_latency_p99_ms": 150.0}
        )
        rankings = _rank_backends([m2, m1], RankCriteria.TOTAL_LATENCY_P99)
        assert rankings[0].backend_label == "b"


class TestBackendComparator:
    """Tests for BackendComparator class."""

    def test_compare_two_backends(self, tmp_path):
        fast = _make_benchmark(ttft_base=20.0, tpot_base=5.0, total_base=100.0)
        slow = _make_benchmark(ttft_base=80.0, tpot_base=15.0, total_base=400.0)
        p1 = _write_json(fast, tmp_path / "fast.json")
        p2 = _write_json(slow, tmp_path / "slow.json")

        comparator = BackendComparator()
        report = comparator.compare([p1, p2], ["fast-backend", "slow-backend"])

        assert isinstance(report, BackendComparisonReport)
        assert len(report.metrics) == 2
        assert len(report.rankings) == 2
        assert report.best_backend == "fast-backend"

    def test_compare_three_backends(self, tmp_path):
        b1 = _make_benchmark(ttft_base=10.0)
        b2 = _make_benchmark(ttft_base=50.0)
        b3 = _make_benchmark(ttft_base=30.0)
        p1 = _write_json(b1, tmp_path / "b1.json")
        p2 = _write_json(b2, tmp_path / "b2.json")
        p3 = _write_json(b3, tmp_path / "b3.json")

        report = compare_backends([p1, p2, p3], ["a", "b", "c"])
        assert len(report.metrics) == 3
        assert report.best_backend == "a"

    def test_compare_with_explicit_formats(self, tmp_path):
        b1 = _make_benchmark()
        b2 = _make_benchmark()
        p1 = _write_json(b1, tmp_path / "b1.json")
        p2 = _write_json(b2, tmp_path / "b2.json")

        report = compare_backends(
            [p1, p2], ["a", "b"], formats=["native", "native"]
        )
        assert len(report.metrics) == 2
        for m in report.metrics:
            assert m.format_detected == "native"

    def test_mismatched_counts_error(self, tmp_path):
        b = _make_benchmark()
        p = _write_json(b, tmp_path / "b.json")
        with pytest.raises(ValueError, match="must match"):
            compare_backends([p], ["a", "b"])

    def test_single_benchmark_error(self, tmp_path):
        b = _make_benchmark()
        p = _write_json(b, tmp_path / "b.json")
        with pytest.raises(ValueError, match="At least 2"):
            compare_backends([p], ["a"])

    def test_mismatched_formats_error(self, tmp_path):
        b1 = _make_benchmark()
        b2 = _make_benchmark()
        p1 = _write_json(b1, tmp_path / "b1.json")
        p2 = _write_json(b2, tmp_path / "b2.json")
        with pytest.raises(ValueError, match="must match"):
            compare_backends([p1, p2], ["a", "b"], formats=["native"])

    def test_rank_by_tpot(self, tmp_path):
        fast = _make_benchmark(tpot_base=2.0)
        slow = _make_benchmark(tpot_base=30.0)
        p1 = _write_json(fast, tmp_path / "fast.json")
        p2 = _write_json(slow, tmp_path / "slow.json")

        report = compare_backends([p1, p2], ["fast", "slow"], rank_by="tpot_p99")
        assert report.best_backend == "fast"
        assert report.rank_criteria == "tpot_p99"

    def test_rank_by_throughput(self, tmp_path):
        high = _make_benchmark(measured_qps=100.0)
        low = _make_benchmark(measured_qps=5.0)
        p1 = _write_json(high, tmp_path / "high.json")
        p2 = _write_json(low, tmp_path / "low.json")

        report = compare_backends([p1, p2], ["high", "low"], rank_by="throughput")
        assert report.best_backend == "high"


class TestCompareBackendsAPI:
    """Tests for the compare_backends() programmatic API."""

    def test_basic_api(self, tmp_path):
        b1 = _make_benchmark()
        b2 = _make_benchmark(ttft_base=200.0)
        p1 = _write_json(b1, tmp_path / "b1.json")
        p2 = _write_json(b2, tmp_path / "b2.json")

        result = compare_backends([p1, p2], ["backend-a", "backend-b"])
        assert isinstance(result, BackendComparisonReport)
        assert result.best_backend == "backend-a"

    def test_api_with_sla(self, tmp_path):
        b1 = _make_benchmark(ttft_base=10.0)
        b2 = _make_benchmark(ttft_base=500.0)
        p1 = _write_json(b1, tmp_path / "b1.json")
        p2 = _write_json(b2, tmp_path / "b2.json")

        result = compare_backends(
            [p1, p2],
            ["a", "b"],
            sla_ttft_p99_ms=100.0,
        )
        # First should pass, second should fail
        assert result.sla_results[0].ttft_p99_pass is True
        assert result.sla_results[1].ttft_p99_pass is False

    def test_report_serializable(self, tmp_path):
        b1 = _make_benchmark()
        b2 = _make_benchmark()
        p1 = _write_json(b1, tmp_path / "b1.json")
        p2 = _write_json(b2, tmp_path / "b2.json")

        result = compare_backends([p1, p2], ["a", "b"])
        j = result.model_dump_json()
        parsed = json.loads(j)
        assert "metrics" in parsed
        assert "rankings" in parsed
        assert "best_backend" in parsed


class TestBackendComparisonConfig:
    """Tests for BackendComparisonConfig model."""

    def test_defaults(self):
        config = BackendComparisonConfig()
        assert config.rank_by == RankCriteria.TTFT_P99
        assert config.sla_ttft_p99_ms is None

    def test_custom_config(self):
        config = BackendComparisonConfig(
            rank_by=RankCriteria.THROUGHPUT,
            sla_ttft_p99_ms=100.0,
            sla_tpot_p99_ms=50.0,
        )
        assert config.rank_by == RankCriteria.THROUGHPUT
        assert config.sla_ttft_p99_ms == 100.0


class TestBackendFormat:
    """Tests for BackendFormat enum."""

    def test_values(self):
        assert BackendFormat.AUTO == "auto"
        assert BackendFormat.NATIVE == "native"
        assert BackendFormat.VLLM == "vllm"
        assert BackendFormat.SGLANG == "sglang"
        assert BackendFormat.TRTLLM == "trtllm"


class TestRankCriteria:
    """Tests for RankCriteria enum."""

    def test_values(self):
        assert RankCriteria.TTFT_P99 == "ttft_p99"
        assert RankCriteria.THROUGHPUT == "throughput"

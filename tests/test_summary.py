"""Tests for benchmark quick summary."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.cli import main
from xpyd_plan.summary import (
    LatencyOverview,
    SummaryGenerator,
    SummaryReport,
    TokenStats,
    summarize_benchmark,
)


def _make_requests(n: int = 50, base_ts: float = 1000.0) -> list[BenchmarkRequest]:
    """Generate n benchmark requests with varying characteristics."""
    requests = []
    for i in range(n):
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i:04d}",
                prompt_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
                ttft_ms=10.0 + i * 0.5,
                tpot_ms=5.0 + i * 0.2,
                total_latency_ms=100.0 + i * 2.0,
                timestamp=base_ts + i * 0.1,
            )
        )
    return requests


def _make_data(n: int = 50) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=6,
            total_instances=8,
            measured_qps=100.0,
        ),
        requests=_make_requests(n),
    )


class TestSummaryGenerator:
    def test_basic_summary(self):
        data = _make_data(50)
        gen = SummaryGenerator(data)
        report = gen.generate()

        assert report.request_count == 50
        assert report.measured_qps == 100.0
        assert report.num_prefill_instances == 2
        assert report.num_decode_instances == 6
        assert report.pd_ratio == "2:6"

    def test_duration_calculated(self):
        data = _make_data(50)
        report = SummaryGenerator(data).generate()
        # 50 requests, 0.1s apart -> ~4.9s duration
        assert report.duration_seconds > 0

    def test_single_request_duration_zero(self):
        data = _make_data(1)
        report = SummaryGenerator(data).generate()
        assert report.duration_seconds == 0.0

    def test_prompt_token_stats(self):
        data = _make_data(50)
        report = SummaryGenerator(data).generate()
        ts = report.prompt_tokens
        # min = 100, max = 100 + 49*10 = 590
        assert ts.min == 100
        assert ts.max == 590
        assert ts.mean > 0
        assert ts.p50 > 0
        assert ts.p95 > 0

    def test_output_token_stats(self):
        data = _make_data(50)
        report = SummaryGenerator(data).generate()
        ts = report.output_tokens
        assert ts.min == 50
        assert ts.max == 50 + 49 * 5

    def test_ttft_latency_overview(self):
        data = _make_data(50)
        report = SummaryGenerator(data).generate()
        lat = report.ttft
        assert lat.min_ms == 10.0
        assert lat.max_ms == pytest.approx(10.0 + 49 * 0.5, abs=0.01)
        assert lat.mean_ms > 0
        assert lat.p50_ms > 0
        assert lat.p95_ms > 0
        assert lat.p99_ms > 0
        assert lat.p50_ms <= lat.p95_ms <= lat.p99_ms <= lat.max_ms

    def test_tpot_latency_overview(self):
        data = _make_data(50)
        report = SummaryGenerator(data).generate()
        lat = report.tpot
        assert lat.min_ms == 5.0
        assert lat.p95_ms >= lat.p50_ms

    def test_total_latency_overview(self):
        data = _make_data(50)
        report = SummaryGenerator(data).generate()
        lat = report.total_latency
        assert lat.min_ms == 100.0
        assert lat.max_ms == pytest.approx(100.0 + 49 * 2.0, abs=0.01)

    def test_report_serializable(self):
        data = _make_data(50)
        report = SummaryGenerator(data).generate()
        d = report.model_dump()
        assert isinstance(d, dict)
        assert "request_count" in d
        assert "prompt_tokens" in d
        assert "ttft" in d

    def test_report_json_roundtrip(self):
        data = _make_data(50)
        report = SummaryGenerator(data).generate()
        j = report.model_dump_json()
        restored = SummaryReport.model_validate_json(j)
        assert restored.request_count == report.request_count
        assert restored.pd_ratio == report.pd_ratio


class TestSummarizeBenchmarkAPI:
    def test_api_returns_report(self):
        data = _make_data(20)
        report = summarize_benchmark(data)
        assert isinstance(report, SummaryReport)
        assert report.request_count == 20

    def test_api_matches_class(self):
        data = _make_data(30)
        api_report = summarize_benchmark(data)
        class_report = SummaryGenerator(data).generate()
        assert api_report.request_count == class_report.request_count
        assert api_report.pd_ratio == class_report.pd_ratio


class TestTokenStats:
    def test_token_stats_model(self):
        ts = TokenStats(min=10, max=100, mean=55.0, p50=50.0, p95=95.0)
        assert ts.min == 10
        assert ts.max == 100

    def test_latency_overview_model(self):
        lo = LatencyOverview(
            min_ms=1.0, max_ms=100.0, mean_ms=50.0,
            p50_ms=45.0, p95_ms=90.0, p99_ms=98.0,
        )
        assert lo.p99_ms == 98.0


class TestSummaryCLI:
    def _write_benchmark(self, path: Path, data: BenchmarkData) -> None:
        path.write_text(json.dumps(data.model_dump()))

    def test_cli_table_output(self, tmp_path, capsys):
        data = _make_data(20)
        bench_file = tmp_path / "bench.json"
        self._write_benchmark(bench_file, data)

        main(["summary", "--benchmark", str(bench_file)])
        captured = capsys.readouterr()
        assert "Benchmark Summary" in captured.out

    def test_cli_json_output(self, tmp_path):
        data = _make_data(20)
        bench_file = tmp_path / "bench.json"
        self._write_benchmark(bench_file, data)

        buf = StringIO()
        with patch("sys.stdout", buf):
            main(["summary", "--benchmark", str(bench_file), "--output-format", "json"])
        parsed = json.loads(buf.getvalue())
        assert parsed["request_count"] == 20
        assert parsed["pd_ratio"] == "2:6"
        assert "ttft" in parsed
        assert "prompt_tokens" in parsed

    def test_cli_json_has_all_fields(self, tmp_path):
        data = _make_data(30)
        bench_file = tmp_path / "bench.json"
        self._write_benchmark(bench_file, data)

        buf = StringIO()
        with patch("sys.stdout", buf):
            main(["summary", "--benchmark", str(bench_file), "--output-format", "json"])
        parsed = json.loads(buf.getvalue())
        assert "duration_seconds" in parsed
        assert "measured_qps" in parsed
        assert "output_tokens" in parsed
        assert "tpot" in parsed
        assert "total_latency" in parsed


class TestEdgeCases:
    def test_two_requests(self):
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=10.0,
            ),
            requests=[
                BenchmarkRequest(
                    request_id="r0", prompt_tokens=100, output_tokens=50,
                    ttft_ms=10.0, tpot_ms=5.0, total_latency_ms=60.0, timestamp=1000.0,
                ),
                BenchmarkRequest(
                    request_id="r1", prompt_tokens=200, output_tokens=100,
                    ttft_ms=20.0, tpot_ms=10.0, total_latency_ms=120.0, timestamp=1001.0,
                ),
            ],
        )
        report = summarize_benchmark(data)
        assert report.request_count == 2
        assert report.pd_ratio == "1:1"
        assert report.duration_seconds == 1.0

    def test_uniform_data(self):
        """All requests identical — stats should be uniform."""
        reqs = [
            BenchmarkRequest(
                request_id=f"r{i}", prompt_tokens=100, output_tokens=50,
                ttft_ms=10.0, tpot_ms=5.0, total_latency_ms=60.0, timestamp=1000.0 + i,
            )
            for i in range(10)
        ]
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=3, num_decode_instances=5,
                total_instances=8, measured_qps=50.0,
            ),
            requests=reqs,
        )
        report = summarize_benchmark(data)
        assert report.prompt_tokens.min == report.prompt_tokens.max == 100
        assert report.ttft.min_ms == report.ttft.max_ms == 10.0

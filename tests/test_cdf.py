"""Tests for latency CDF export."""

from __future__ import annotations

import csv
import io
import json
import subprocess
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.cdf import CDFGenerator, CDFReport, generate_cdf


def _make_benchmark(num_requests: int = 200) -> BenchmarkData:
    """Create benchmark data with known latency distribution."""
    requests = []
    for i in range(num_requests):
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=50 + i * 5,
                output_tokens=20 + i * 3,
                ttft_ms=10.0 + i * 0.5,
                tpot_ms=5.0 + i * 0.3,
                total_latency_ms=100.0 + i * 2.0,
                timestamp=1000.0 + i * 0.1,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=4,
            total_instances=6,
            measured_qps=50.0,
        ),
        requests=requests,
    )


def _write_benchmark(data: BenchmarkData, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data.model_dump(), f)


class TestCDFGenerator:
    def test_basic_cdf(self):
        data = _make_benchmark()
        gen = CDFGenerator(data, source="test")
        curve = gen.generate(metric="total_latency", n_points=50)
        assert curve.source == "test"
        assert curve.metric == "total_latency"
        assert len(curve.points) == 50
        assert curve.sla_marker is None

    def test_cdf_monotonic(self):
        data = _make_benchmark()
        gen = CDFGenerator(data)
        curve = gen.generate(n_points=100)
        latencies = [p.latency_ms for p in curve.points]
        for i in range(1, len(latencies)):
            assert latencies[i] >= latencies[i - 1]

    def test_cdf_probability_range(self):
        data = _make_benchmark()
        gen = CDFGenerator(data)
        curve = gen.generate(n_points=100)
        probs = [p.cumulative_probability for p in curve.points]
        assert probs[0] == pytest.approx(0.0)
        assert probs[-1] == pytest.approx(1.0)
        for p in probs:
            assert 0.0 <= p <= 1.0

    def test_cdf_ttft_metric(self):
        data = _make_benchmark()
        gen = CDFGenerator(data)
        curve = gen.generate(metric="ttft", n_points=20)
        assert curve.metric == "ttft"
        assert len(curve.points) == 20

    def test_cdf_tpot_metric(self):
        data = _make_benchmark()
        gen = CDFGenerator(data)
        curve = gen.generate(metric="tpot", n_points=30)
        assert curve.metric == "tpot"
        assert len(curve.points) == 30

    def test_sla_marker(self):
        data = _make_benchmark(num_requests=100)
        gen = CDFGenerator(data)
        # total_latency ranges from 100.0 to 298.0
        curve = gen.generate(metric="total_latency", sla_threshold_ms=200.0)
        assert curve.sla_marker is not None
        assert curve.sla_marker.threshold_ms == 200.0
        # About 50% should be below 200ms (requests 0-49 have total < 200)
        assert 40.0 < curve.sla_marker.percentile_at_threshold < 60.0

    def test_sla_marker_all_pass(self):
        data = _make_benchmark(num_requests=50)
        gen = CDFGenerator(data)
        curve = gen.generate(metric="total_latency", sla_threshold_ms=9999.0)
        assert curve.sla_marker is not None
        assert curve.sla_marker.percentile_at_threshold == 100.0

    def test_sla_marker_none_pass(self):
        data = _make_benchmark(num_requests=50)
        gen = CDFGenerator(data)
        curve = gen.generate(metric="total_latency", sla_threshold_ms=0.0)
        assert curve.sla_marker is not None
        assert curve.sla_marker.percentile_at_threshold == 0.0

    def test_custom_points(self):
        data = _make_benchmark()
        gen = CDFGenerator(data)
        curve = gen.generate(n_points=10)
        assert len(curve.points) == 10

    def test_single_request(self):
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=1.0,
            ),
            requests=[
                BenchmarkRequest(
                    request_id="r-0",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=20.0,
                    tpot_ms=10.0,
                    total_latency_ms=150.0,
                    timestamp=1000.0,
                )
            ],
        )
        gen = CDFGenerator(data)
        curve = gen.generate(n_points=5)
        assert len(curve.points) == 5


class TestGenerateCDF:
    def test_single_file(self, tmp_path):
        data = _make_benchmark(100)
        path = tmp_path / "bench.json"
        _write_benchmark(data, path)
        report = generate_cdf([str(path)])
        assert len(report.curves) == 1
        assert report.curves[0].source == "bench.json"

    def test_multi_file(self, tmp_path):
        for name in ["a.json", "b.json", "c.json"]:
            _write_benchmark(_make_benchmark(50), tmp_path / name)
        report = generate_cdf(
            [str(tmp_path / n) for n in ["a.json", "b.json", "c.json"]]
        )
        assert len(report.curves) == 3
        sources = [c.source for c in report.curves]
        assert sources == ["a.json", "b.json", "c.json"]

    def test_custom_labels(self, tmp_path):
        _write_benchmark(_make_benchmark(50), tmp_path / "bench.json")
        report = generate_cdf(
            [str(tmp_path / "bench.json")],
            labels=["my-label"],
        )
        assert report.curves[0].source == "my-label"

    def test_with_sla(self, tmp_path):
        _write_benchmark(_make_benchmark(100), tmp_path / "b.json")
        report = generate_cdf(
            [str(tmp_path / "b.json")], sla_threshold_ms=200.0
        )
        assert report.curves[0].sla_marker is not None

    def test_json_serialization(self, tmp_path):
        _write_benchmark(_make_benchmark(50), tmp_path / "b.json")
        report = generate_cdf([str(tmp_path / "b.json")], n_points=10)
        d = report.model_dump()
        assert "curves" in d
        assert len(d["curves"][0]["points"]) == 10
        # Roundtrip
        CDFReport.model_validate(d)


class TestCDFCli:
    def test_table_output(self, tmp_path):
        _write_benchmark(_make_benchmark(50), tmp_path / "b.json")
        result = subprocess.run(
            [
                "xpyd-plan", "cdf",
                "--benchmark", str(tmp_path / "b.json"),
                "--points", "10",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "CDF" in result.stdout

    def test_json_output(self, tmp_path):
        _write_benchmark(_make_benchmark(50), tmp_path / "b.json")
        result = subprocess.run(
            [
                "xpyd-plan", "cdf",
                "--benchmark", str(tmp_path / "b.json"),
                "--output-format", "json",
                "--points", "10",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "curves" in data

    def test_csv_output(self, tmp_path):
        _write_benchmark(_make_benchmark(50), tmp_path / "b.json")
        result = subprocess.run(
            [
                "xpyd-plan", "cdf",
                "--benchmark", str(tmp_path / "b.json"),
                "--output-format", "csv",
                "--points", "5",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        reader = csv.reader(io.StringIO(result.stdout))
        rows = list(reader)
        assert rows[0] == ["source", "metric", "latency_ms", "cumulative_probability"]
        assert len(rows) == 6  # header + 5 data rows

    def test_multi_benchmark_cli(self, tmp_path):
        for name in ["a.json", "b.json"]:
            _write_benchmark(_make_benchmark(30), tmp_path / name)
        result = subprocess.run(
            [
                "xpyd-plan", "cdf",
                "--benchmark", str(tmp_path / "a.json"), str(tmp_path / "b.json"),
                "--output-format", "json",
                "--points", "5",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data["curves"]) == 2

    def test_sla_threshold_cli(self, tmp_path):
        _write_benchmark(_make_benchmark(50), tmp_path / "b.json")
        result = subprocess.run(
            [
                "xpyd-plan", "cdf",
                "--benchmark", str(tmp_path / "b.json"),
                "--sla-threshold", "150.0",
                "--output-format", "json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["curves"][0]["sla_marker"] is not None

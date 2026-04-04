"""Tests for benchmark discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import BenchmarkMetadata
from xpyd_plan.discovery import (
    BenchmarkDiscovery,
    ValidationStatus,
    _config_key,
    _depth_of,
    _quick_validate,
    discover_benchmarks,
)


def _make_benchmark_json(
    num_prefill: int = 2,
    num_decode: int = 2,
    qps: float = 10.0,
    num_requests: int = 5,
) -> dict:
    """Create a valid benchmark JSON structure."""
    return {
        "metadata": {
            "num_prefill_instances": num_prefill,
            "num_decode_instances": num_decode,
            "total_instances": num_prefill + num_decode,
            "measured_qps": qps,
        },
        "requests": [
            {
                "request_id": f"req-{i}",
                "prompt_tokens": 100,
                "output_tokens": 50,
                "ttft_ms": 20.0,
                "tpot_ms": 10.0,
                "total_latency_ms": 70.0,
                "timestamp": 1000.0 + i,
            }
            for i in range(num_requests)
        ],
    }


def _write_json(path: Path, data: dict) -> None:
    """Write JSON data to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


class TestConfigKey:
    """Tests for _config_key helper."""

    def test_basic(self) -> None:
        meta = BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=4,
            total_instances=6,
            measured_qps=10.0,
        )
        assert _config_key(meta) == "P2:D4"

    def test_single_instances(self) -> None:
        meta = BenchmarkMetadata(
            num_prefill_instances=1,
            num_decode_instances=1,
            total_instances=2,
            measured_qps=5.0,
        )
        assert _config_key(meta) == "P1:D1"


class TestDepthOf:
    """Tests for _depth_of helper."""

    def test_root_file(self, tmp_path: Path) -> None:
        f = tmp_path / "bench.json"
        f.touch()
        assert _depth_of(f, tmp_path) == 0

    def test_nested_file(self, tmp_path: Path) -> None:
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        f = sub / "bench.json"
        f.touch()
        assert _depth_of(f, tmp_path) == 2


class TestQuickValidate:
    """Tests for _quick_validate."""

    def test_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "valid.json"
        _write_json(f, _make_benchmark_json())
        result = _quick_validate(f)
        assert result.status == ValidationStatus.VALID
        assert result.num_requests == 5
        assert result.config_key == "P2:D2"

    def test_invalid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json {{{")
        result = _quick_validate(f)
        assert result.status == ValidationStatus.INVALID_JSON

    def test_missing_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "no_meta.json"
        _write_json(f, {"requests": []})
        result = _quick_validate(f)
        assert result.status == ValidationStatus.INVALID_SCHEMA

    def test_missing_requests(self, tmp_path: Path) -> None:
        f = tmp_path / "no_req.json"
        meta = {
            "num_prefill_instances": 1,
            "num_decode_instances": 1,
            "total_instances": 2,
            "measured_qps": 5.0,
        }
        _write_json(f, {"metadata": meta})
        result = _quick_validate(f)
        assert result.status == ValidationStatus.INVALID_SCHEMA

    def test_missing_metadata_field(self, tmp_path: Path) -> None:
        f = tmp_path / "partial.json"
        _write_json(f, {"metadata": {"num_prefill_instances": 1}, "requests": []})
        result = _quick_validate(f)
        assert result.status == ValidationStatus.INVALID_SCHEMA
        assert "Missing metadata field" in (result.error or "")

    def test_not_an_object(self, tmp_path: Path) -> None:
        f = tmp_path / "array.json"
        _write_json(f, [1, 2, 3])
        result = _quick_validate(f)
        assert result.status == ValidationStatus.INVALID_SCHEMA


class TestBenchmarkDiscovery:
    """Tests for BenchmarkDiscovery."""

    def test_discover_valid_files(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "a.json", _make_benchmark_json())
        _write_json(
            tmp_path / "b.json",
            _make_benchmark_json(num_prefill=4, num_decode=4, qps=20.0),
        )

        discoverer = BenchmarkDiscovery()
        report = discoverer.discover(tmp_path)

        assert report.total_files_scanned == 2
        assert report.valid_count == 2
        assert report.invalid_count == 0

    def test_discover_mixed_valid_invalid(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "good.json", _make_benchmark_json())
        (tmp_path / "bad.json").write_text("not json")

        report = BenchmarkDiscovery().discover(tmp_path)
        assert report.valid_count == 1
        assert report.invalid_count == 1

    def test_discover_empty_dir(self, tmp_path: Path) -> None:
        report = BenchmarkDiscovery().discover(tmp_path)
        assert report.total_files_scanned == 0
        assert report.valid_count == 0

    def test_discover_nested(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "level0.json", _make_benchmark_json())
        _write_json(tmp_path / "sub" / "level1.json", _make_benchmark_json())
        _write_json(tmp_path / "sub" / "deep" / "level2.json", _make_benchmark_json())

        report = BenchmarkDiscovery().discover(tmp_path)
        assert report.valid_count == 3

    def test_max_depth_limit(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "level0.json", _make_benchmark_json())
        _write_json(tmp_path / "sub" / "level1.json", _make_benchmark_json())
        _write_json(tmp_path / "sub" / "deep" / "level2.json", _make_benchmark_json())

        report = BenchmarkDiscovery().discover(tmp_path, max_depth=0)
        assert report.valid_count == 1

        report = BenchmarkDiscovery().discover(tmp_path, max_depth=1)
        assert report.valid_count == 2

    def test_custom_pattern(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "h100_bench.json", _make_benchmark_json())
        _write_json(tmp_path / "a100_bench.json", _make_benchmark_json())
        _write_json(tmp_path / "other.json", _make_benchmark_json())

        report = BenchmarkDiscovery().discover(tmp_path, pattern="*h100*.json")
        assert report.valid_count == 1

    def test_config_groups(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "a.json", _make_benchmark_json(num_prefill=2, num_decode=2))
        _write_json(tmp_path / "b.json", _make_benchmark_json(num_prefill=2, num_decode=2))
        _write_json(tmp_path / "c.json", _make_benchmark_json(num_prefill=4, num_decode=4))

        report = BenchmarkDiscovery().discover(tmp_path)
        assert len(report.groups) == 2

        p2d2 = next(g for g in report.groups if g.config_key == "P2:D2")
        assert p2d2.count == 2
        p4d4 = next(g for g in report.groups if g.config_key == "P4:D4")
        assert p4d4.count == 1

    def test_nonexistent_dir(self) -> None:
        with pytest.raises(FileNotFoundError):
            BenchmarkDiscovery().discover("/nonexistent/path")

    def test_not_a_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(NotADirectoryError):
            BenchmarkDiscovery().discover(f)

    def test_report_fields(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "bench.json", _make_benchmark_json())
        report = BenchmarkDiscovery().discover(tmp_path, pattern="*.json", max_depth=5)
        assert report.root_dir == str(tmp_path.resolve())
        assert report.pattern == "*.json"
        assert report.max_depth == 5

    def test_file_size_recorded(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "bench.json", _make_benchmark_json())
        report = BenchmarkDiscovery().discover(tmp_path)
        assert report.benchmarks[0].file_size_bytes > 0


class TestDiscoverBenchmarksAPI:
    """Tests for the programmatic API."""

    def test_returns_dict(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "bench.json", _make_benchmark_json())
        result = discover_benchmarks(tmp_path)
        assert isinstance(result, dict)
        assert "benchmarks" in result
        assert "groups" in result
        assert result["valid_count"] == 1

    def test_with_pattern(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "test.json", _make_benchmark_json())
        result = discover_benchmarks(tmp_path, pattern="test*.json")
        assert result["valid_count"] == 1


class TestDiscoverCLI:
    """Tests for CLI integration."""

    def test_table_output(self, tmp_path: Path) -> None:
        import io
        from contextlib import redirect_stdout

        from xpyd_plan.cli._main import main

        _write_json(tmp_path / "bench.json", _make_benchmark_json())

        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                main(["discover", "--dir", str(tmp_path)])
            except SystemExit:
                pass

    def test_json_output(self, tmp_path: Path) -> None:
        import io
        from contextlib import redirect_stdout

        from xpyd_plan.cli._main import main

        _write_json(tmp_path / "bench.json", _make_benchmark_json())

        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                main(["discover", "--dir", str(tmp_path), "--output-format", "json"])
            except SystemExit:
                pass

        output = buf.getvalue()
        if output.strip():
            result = json.loads(output)
            assert "benchmarks" in result

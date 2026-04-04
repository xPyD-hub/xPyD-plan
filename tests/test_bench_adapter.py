"""Tests for bench_adapter — xpyd-bench format conversion and auto-detection."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.bench_adapter import (
    FormatDetectionError,
    UnsupportedSchemaVersion,
    XpydBenchAdapter,
    detect_format,
    detect_schema_version,
    load_benchmark_auto,
    load_benchmark_auto_from_dict,
)
from xpyd_plan.benchmark_models import BenchmarkData


def _make_native_data() -> dict:
    """Create a valid native-format benchmark dataset."""
    return {
        "metadata": {
            "num_prefill_instances": 2,
            "num_decode_instances": 6,
            "total_instances": 8,
            "measured_qps": 10.0,
        },
        "requests": [
            {
                "request_id": "r1",
                "prompt_tokens": 100,
                "output_tokens": 200,
                "ttft_ms": 50.0,
                "tpot_ms": 10.0,
                "total_latency_ms": 2050.0,
                "timestamp": 1700000000.0,
            },
        ],
    }


def _make_xpyd_bench_data() -> dict:
    """Create a valid xpyd-bench format dataset."""
    return {
        "schema_version": "1",
        "bench_config": {
            "prefill_instances": 2,
            "decode_instances": 6,
            "total_instances": 8,
            "target_qps": 10.0,
            "measured_qps": 9.8,
            "duration_seconds": 60,
        },
        "results": [
            {
                "id": "req-001",
                "input_tokens": 128,
                "output_tokens": 256,
                "time_to_first_token_ms": 45.2,
                "time_per_output_token_ms": 12.1,
                "end_to_end_latency_ms": 3141.8,
                "start_time": 1700000000.0,
            },
            {
                "id": "req-002",
                "input_tokens": 64,
                "output_tokens": 128,
                "time_to_first_token_ms": 30.0,
                "time_per_output_token_ms": 11.5,
                "end_to_end_latency_ms": 1502.0,
                "start_time": 1700000001.0,
            },
        ],
    }


# --- detect_schema_version ---


class TestDetectSchemaVersion:
    def test_explicit_version(self):
        data = {"schema_version": "1", "bench_config": {}, "results": []}
        assert detect_schema_version(data) == "1"

    def test_native_format_implicit(self):
        assert detect_schema_version(_make_native_data()) == "1"

    def test_xpyd_bench_format_implicit(self):
        data = _make_xpyd_bench_data()
        del data["schema_version"]
        assert detect_schema_version(data) == "1"

    def test_unknown_format_raises(self):
        with pytest.raises(FormatDetectionError):
            detect_schema_version({"foo": "bar"})


# --- detect_format ---


class TestDetectFormat:
    def test_native(self):
        assert detect_format(_make_native_data()) == "native"

    def test_xpyd_bench(self):
        assert detect_format(_make_xpyd_bench_data()) == "xpyd-bench"

    def test_unknown_raises(self):
        with pytest.raises(FormatDetectionError):
            detect_format({"random_key": 123})


# --- XpydBenchAdapter ---


class TestXpydBenchAdapter:
    def test_convert_basic(self):
        adapter = XpydBenchAdapter()
        data = _make_xpyd_bench_data()
        result = adapter.convert(data)

        assert isinstance(result, BenchmarkData)
        assert result.metadata.num_prefill_instances == 2
        assert result.metadata.num_decode_instances == 6
        assert result.metadata.measured_qps == 9.8
        assert len(result.requests) == 2
        assert result.requests[0].request_id == "req-001"
        assert result.requests[0].prompt_tokens == 128
        assert result.requests[0].ttft_ms == 45.2
        assert result.requests[0].tpot_ms == 12.1

    def test_unsupported_version_raises(self):
        adapter = XpydBenchAdapter()
        data = _make_xpyd_bench_data()
        data["schema_version"] = "99"
        with pytest.raises(UnsupportedSchemaVersion) as exc_info:
            adapter.convert(data)
        assert "99" in str(exc_info.value)

    def test_load_from_file(self, tmp_path):
        adapter = XpydBenchAdapter()
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(_make_xpyd_bench_data()))

        result = adapter.load(path)
        assert isinstance(result, BenchmarkData)
        assert len(result.requests) == 2

    def test_load_file_not_found(self):
        adapter = XpydBenchAdapter()
        with pytest.raises(FileNotFoundError):
            adapter.load("/nonexistent/path.json")


# --- load_benchmark_auto ---


class TestLoadBenchmarkAuto:
    def test_auto_native(self, tmp_path):
        path = tmp_path / "native.json"
        path.write_text(json.dumps(_make_native_data()))
        result = load_benchmark_auto(path)
        assert isinstance(result, BenchmarkData)
        assert result.metadata.measured_qps == 10.0

    def test_auto_xpyd_bench(self, tmp_path):
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(_make_xpyd_bench_data()))
        result = load_benchmark_auto(path)
        assert isinstance(result, BenchmarkData)
        assert result.metadata.measured_qps == 9.8

    def test_auto_from_dict_native(self):
        result = load_benchmark_auto_from_dict(_make_native_data())
        assert isinstance(result, BenchmarkData)

    def test_auto_from_dict_xpyd_bench(self):
        result = load_benchmark_auto_from_dict(_make_xpyd_bench_data())
        assert isinstance(result, BenchmarkData)
        assert result.requests[0].request_id == "req-001"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_benchmark_auto("/no/such/file.json")

    def test_unsupported_version(self, tmp_path):
        data = _make_xpyd_bench_data()
        data["schema_version"] = "42"
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(data))
        with pytest.raises(UnsupportedSchemaVersion):
            load_benchmark_auto(path)


# --- Schema version edge cases ---


class TestSchemaVersionEdgeCases:
    def test_version_1_0(self):
        data = _make_xpyd_bench_data()
        data["schema_version"] = "1.0"
        adapter = XpydBenchAdapter()
        result = adapter.convert(data)
        assert isinstance(result, BenchmarkData)

    def test_version_as_int(self):
        """schema_version as integer should be coerced to string."""
        data = _make_xpyd_bench_data()
        data["schema_version"] = 1
        assert detect_schema_version(data) == "1"

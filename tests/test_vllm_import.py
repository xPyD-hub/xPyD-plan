"""Tests for vLLM Benchmark Format Importer (M108)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from xpyd_plan.vllm_import import (
    ImportConfig,
    VLLMBenchmarkData,
    VLLMImporter,
    VLLMRequest,
    _detect_format,
    _parse_vllm_request,
    import_vllm,
)


def _make_vllm_requests(n: int = 5, with_timestamps: bool = True) -> list[dict]:
    """Generate sample vLLM request dicts."""
    requests = []
    for i in range(n):
        req = {
            "prompt_len": 100 + i * 10,
            "output_len": 50 + i * 5,
            "ttft": 0.05 + i * 0.01,
            "tpot": 0.02 + i * 0.001,
            "request_latency": 1.0 + i * 0.1,
        }
        if with_timestamps:
            req["timestamp"] = 1700000000.0 + i * 0.5
        requests.append(req)
    return requests


def _write_json(data, suffix=".json") -> Path:
    """Write data to a temp JSON file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    json.dump(data, f)
    f.close()
    return Path(f.name)


class TestFormatDetection:
    """Tests for format auto-detection."""

    def test_detect_vllm_list(self):
        data = _make_vllm_requests(2)
        assert _detect_format(data) == "vllm"

    def test_detect_native_dict(self):
        data = {"metadata": {}, "requests": []}
        assert _detect_format(data) == "native"

    def test_detect_vllm_wrapper(self):
        data = {"results": _make_vllm_requests(2)}
        assert _detect_format(data) == "vllm"

    def test_detect_empty_list_raises(self):
        with pytest.raises(ValueError, match="Empty list"):
            _detect_format([])

    def test_detect_unknown_dict_raises(self):
        with pytest.raises(ValueError, match="Dict format detected"):
            _detect_format({"foo": "bar"})

    def test_detect_unknown_list_raises(self):
        with pytest.raises(ValueError, match="missing expected vLLM fields"):
            _detect_format([{"foo": "bar"}])

    def test_detect_wrong_type_raises(self):
        with pytest.raises(ValueError, match="Unexpected data type"):
            _detect_format("not a dict or list")


class TestVLLMRequest:
    """Tests for VLLMRequest model."""

    def test_basic_request(self):
        req = VLLMRequest(
            prompt_len=100, output_len=50, ttft=0.05, tpot=0.02,
            request_latency=1.0, timestamp=1700000000.0,
        )
        assert req.prompt_len == 100
        assert req.tpot == 0.02

    def test_request_without_optional_fields(self):
        req = VLLMRequest(
            prompt_len=100, output_len=50, ttft=0.05,
            request_latency=1.0,
        )
        assert req.tpot is None
        assert req.timestamp is None


class TestParseVLLMRequest:
    """Tests for single request parsing with graceful fallbacks."""

    def test_full_fields(self):
        raw = {
            "prompt_len": 100, "output_len": 50, "ttft": 0.05,
            "tpot": 0.02, "request_latency": 1.0, "timestamp": 1.0,
        }
        req, warns = _parse_vllm_request(raw, 0)
        assert req.prompt_len == 100
        assert len(warns) == 0

    def test_missing_ttft_estimated(self):
        raw = {
            "prompt_len": 100, "output_len": 50,
            "request_latency": 1.0,
        }
        req, warns = _parse_vllm_request(raw, 0)
        assert req.ttft == pytest.approx(0.1)  # 10% of 1.0
        assert any("missing 'ttft'" in w for w in warns)

    def test_itl_used_as_tpot(self):
        raw = {
            "prompt_len": 100, "output_len": 50, "ttft": 0.05,
            "itl": 0.03, "request_latency": 1.0,
        }
        req, warns = _parse_vllm_request(raw, 0)
        assert req.tpot == 0.03
        assert req.itl == 0.03

    def test_missing_tpot_estimated(self):
        raw = {
            "prompt_len": 100, "output_len": 50, "ttft": 0.05,
            "request_latency": 1.0,
        }
        req, warns = _parse_vllm_request(raw, 0)
        assert req.tpot is not None
        assert any("missing 'tpot'" in w for w in warns)

    def test_missing_prompt_len_raises(self):
        with pytest.raises(ValueError, match="missing required field 'prompt_len'"):
            _parse_vllm_request({"output_len": 50, "request_latency": 1.0}, 0)

    def test_missing_request_latency_raises(self):
        with pytest.raises(ValueError, match="missing required field 'request_latency'"):
            _parse_vllm_request({"prompt_len": 100, "output_len": 50}, 0)


class TestVLLMImporter:
    """Tests for VLLMImporter class."""

    def test_basic_import(self):
        data = _make_vllm_requests(5)
        importer = VLLMImporter(num_prefill_instances=2, num_decode_instances=4)
        result = importer.import_data(data)
        assert result.num_requests == 5
        assert result.source_format == "vllm"
        assert result.benchmark_data.metadata.num_prefill_instances == 2
        assert result.benchmark_data.metadata.num_decode_instances == 4
        assert result.benchmark_data.metadata.total_instances == 6

    def test_import_converts_to_ms(self):
        data = [{"prompt_len": 100, "output_len": 50, "ttft": 0.05,
                 "tpot": 0.02, "request_latency": 1.0, "timestamp": 1.0}]
        importer = VLLMImporter(num_prefill_instances=1, num_decode_instances=1)
        result = importer.import_data(data)
        req = result.benchmark_data.requests[0]
        assert req.ttft_ms == pytest.approx(50.0)
        assert req.tpot_ms == pytest.approx(20.0)
        assert req.total_latency_ms == pytest.approx(1000.0)

    def test_import_without_timestamps(self):
        data = _make_vllm_requests(3, with_timestamps=False)
        importer = VLLMImporter(num_prefill_instances=1, num_decode_instances=1)
        result = importer.import_data(data)
        # Sequential timestamps: 0.0, 1.0, 2.0
        timestamps = [r.timestamp for r in result.benchmark_data.requests]
        assert timestamps == [0.0, 1.0, 2.0]

    def test_import_wrapper_dict(self):
        data = {"results": _make_vllm_requests(3)}
        importer = VLLMImporter(num_prefill_instances=1, num_decode_instances=1)
        result = importer.import_data(data)
        assert result.num_requests == 3

    def test_import_empty_raises(self):
        importer = VLLMImporter(num_prefill_instances=1, num_decode_instances=1)
        with pytest.raises(ValueError, match="No requests found"):
            importer.import_data([])

    def test_invalid_instances_raises(self):
        with pytest.raises(ValueError, match="num_prefill_instances must be >= 1"):
            VLLMImporter(num_prefill_instances=0, num_decode_instances=1)
        with pytest.raises(ValueError, match="num_decode_instances must be >= 1"):
            VLLMImporter(num_prefill_instances=1, num_decode_instances=0)

    def test_import_file(self, tmp_path):
        data = _make_vllm_requests(3)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data))
        importer = VLLMImporter(num_prefill_instances=2, num_decode_instances=3)
        result = importer.import_file(path)
        assert result.num_requests == 3

    def test_request_ids_sequential(self):
        data = _make_vllm_requests(3)
        importer = VLLMImporter(num_prefill_instances=1, num_decode_instances=1)
        result = importer.import_data(data)
        ids = [r.request_id for r in result.benchmark_data.requests]
        assert ids == ["vllm-000000", "vllm-000001", "vllm-000002"]

    def test_measured_qps_from_timestamps(self):
        data = _make_vllm_requests(5, with_timestamps=True)
        importer = VLLMImporter(num_prefill_instances=1, num_decode_instances=1)
        result = importer.import_data(data)
        # 5 requests over 2.0 seconds → 2.5 QPS
        assert result.benchmark_data.metadata.measured_qps == pytest.approx(2.5)


class TestImportVLLMFunction:
    """Tests for the programmatic import_vllm() API."""

    def test_import_vllm_auto_detect(self, tmp_path):
        data = _make_vllm_requests(5)
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(data))
        result = import_vllm(input_path, num_prefill_instances=2, num_decode_instances=3)
        assert result.source_format == "vllm"
        assert result.num_requests == 5

    def test_import_vllm_with_output(self, tmp_path):
        data = _make_vllm_requests(3)
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(data))
        output_path = tmp_path / "output.json"
        import_vllm(
            input_path, num_prefill_instances=1, num_decode_instances=1,
            output_path=output_path,
        )
        assert output_path.exists()
        saved = json.loads(output_path.read_text())
        assert "metadata" in saved
        assert "requests" in saved
        assert len(saved["requests"]) == 3

    def test_import_native_passthrough(self, tmp_path):
        """Native format should be passed through without conversion."""
        native_data = {
            "metadata": {
                "num_prefill_instances": 2,
                "num_decode_instances": 4,
                "total_instances": 6,
                "measured_qps": 10.0,
            },
            "requests": [
                {
                    "request_id": "r0",
                    "prompt_tokens": 100,
                    "output_tokens": 50,
                    "ttft_ms": 50.0,
                    "tpot_ms": 20.0,
                    "total_latency_ms": 1000.0,
                    "timestamp": 1.0,
                }
            ],
        }
        input_path = tmp_path / "native.json"
        input_path.write_text(json.dumps(native_data))
        result = import_vllm(
            input_path, num_prefill_instances=1, num_decode_instances=1,
            format="auto",
        )
        assert result.source_format == "native"

    def test_import_explicit_vllm_format(self, tmp_path):
        data = _make_vllm_requests(2)
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(data))
        result = import_vllm(
            input_path, num_prefill_instances=1, num_decode_instances=1,
            format="vllm",
        )
        assert result.source_format == "vllm"


class TestImportConfig:
    """Tests for ImportConfig model."""

    def test_defaults(self):
        cfg = ImportConfig(num_prefill_instances=2, num_decode_instances=4)
        assert cfg.format == "auto"

    def test_custom_format(self):
        cfg = ImportConfig(num_prefill_instances=1, num_decode_instances=1, format="vllm")
        assert cfg.format == "vllm"


class TestVLLMBenchmarkData:
    """Tests for VLLMBenchmarkData model."""

    def test_basic(self):
        reqs = [
            VLLMRequest(prompt_len=100, output_len=50, ttft=0.05, request_latency=1.0)
        ]
        data = VLLMBenchmarkData(requests=reqs)
        assert len(data.requests) == 1

    def test_empty_raises(self):
        with pytest.raises(Exception):
            VLLMBenchmarkData(requests=[])

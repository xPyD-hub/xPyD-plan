"""Tests for SGLang Benchmark Format Importer (M110)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.sglang_import import (
    SGLangImportConfig,
    SGLangImportResult,
    SGLangRequest,
    _convert_requests,
    _detect_sglang_format,
    import_sglang,
    import_sglang_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sglang_request(
    prompt_len: int = 128,
    output_len: int = 64,
    ttft: float = 0.05,
    itl: list[float] | None = None,
    latency: float = 1.0,
    start_time: float | None = None,
    success: bool = True,
) -> dict:
    """Build a raw SGLang request dict."""
    r: dict = {
        "prompt_len": prompt_len,
        "output_len": output_len,
        "ttft": ttft,
        "itl": itl if itl is not None else [0.015, 0.014, 0.016],
        "latency": latency,
        "success": success,
    }
    if start_time is not None:
        r["start_time"] = start_time
    return r


def _make_sglang_data(n: int = 10, qps: float = 5.0) -> list[dict]:
    """Generate n SGLang request dicts."""
    interval = 1.0 / qps if qps > 0 else 0.2
    return [
        _make_sglang_request(
            prompt_len=100 + i * 10,
            output_len=50 + i * 5,
            ttft=0.04 + i * 0.001,
            itl=[0.012 + i * 0.0001] * 10,
            latency=0.5 + i * 0.05,
            start_time=1000.0 + i * interval,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestFormatDetection:
    def test_detect_sglang_format(self):
        data = _make_sglang_data(3)
        assert _detect_sglang_format(data) is True

    def test_not_sglang_if_empty(self):
        assert _detect_sglang_format([]) is False

    def test_not_sglang_if_dict(self):
        assert _detect_sglang_format({"metadata": {}, "requests": []}) is False

    def test_not_sglang_if_vllm_format(self):
        """vLLM uses 'request_latency' not 'latency'."""
        data = [
            {
                "prompt_len": 128,
                "output_len": 64,
                "ttft": 0.05,
                "itl": 0.015,
                "request_latency": 1.0,
            }
        ]
        assert _detect_sglang_format(data) is False

    def test_not_sglang_if_itl_scalar(self):
        """SGLang itl is always a list."""
        data = [
            {
                "prompt_len": 128,
                "output_len": 64,
                "ttft": 0.05,
                "itl": 0.015,
                "latency": 1.0,
            }
        ]
        assert _detect_sglang_format(data) is False


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestModels:
    def test_sglang_request_valid(self):
        r = SGLangRequest(
            prompt_len=128, output_len=64, ttft=0.05, itl=[0.01, 0.02], latency=1.0
        )
        assert r.prompt_len == 128
        assert len(r.itl) == 2
        assert r.success is True

    def test_sglang_request_invalid_prompt_len(self):
        with pytest.raises(Exception):
            SGLangRequest(prompt_len=0, output_len=64, ttft=0.05, latency=1.0)

    def test_import_config_valid(self):
        c = SGLangImportConfig(num_prefill_instances=2, num_decode_instances=4)
        assert c.format == "auto"


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


class TestConversion:
    def test_basic_conversion(self):
        requests = [
            SGLangRequest(
                prompt_len=128,
                output_len=64,
                ttft=0.05,
                itl=[0.01, 0.02, 0.03],
                latency=1.0,
                start_time=1000.0,
            )
        ]
        native, warnings, num_failed = _convert_requests(requests)
        assert len(native) == 1
        assert native[0].prompt_tokens == 128
        assert native[0].output_tokens == 64
        assert native[0].ttft_ms == pytest.approx(50.0)
        # TPOT = mean([0.01, 0.02, 0.03]) * 1000 = 20.0
        assert native[0].tpot_ms == pytest.approx(20.0)
        assert native[0].total_latency_ms == pytest.approx(1000.0)
        assert num_failed == 0

    def test_failed_requests_filtered(self):
        requests = [
            SGLangRequest(
                prompt_len=128, output_len=64, ttft=0.05, latency=1.0, success=True
            ),
            SGLangRequest(
                prompt_len=128, output_len=64, ttft=0.05, latency=1.0, success=False
            ),
        ]
        native, warnings, num_failed = _convert_requests(requests)
        assert len(native) == 1
        assert num_failed == 1
        assert any("Filtered 1 failed" in w for w in warnings)

    def test_all_failed_raises(self):
        requests = [
            SGLangRequest(
                prompt_len=128, output_len=64, ttft=0.05, latency=1.0, success=False
            ),
        ]
        with pytest.raises(ValueError, match="No successful requests"):
            _convert_requests(requests)

    def test_empty_itl_fallback(self):
        requests = [
            SGLangRequest(
                prompt_len=128,
                output_len=10,
                ttft=0.05,
                itl=[],
                latency=1.0,
                start_time=0.0,
            )
        ]
        native, warnings, _ = _convert_requests(requests)
        # TPOT fallback = (1.0 - 0.05) / (10 - 1) seconds = 0.10556 s = 105.56 ms
        expected_tpot = (1.0 - 0.05) / 9.0 * 1000.0
        assert native[0].tpot_ms == pytest.approx(expected_tpot, rel=1e-3)

    def test_sequential_timestamps_when_missing(self):
        requests = [
            SGLangRequest(
                prompt_len=128, output_len=64, ttft=0.05, latency=1.0
            ),
            SGLangRequest(
                prompt_len=256, output_len=32, ttft=0.03, latency=0.8
            ),
        ]
        native, _, _ = _convert_requests(requests)
        assert native[0].timestamp == 0.0
        assert native[1].timestamp == 1.0


# ---------------------------------------------------------------------------
# End-to-end import
# ---------------------------------------------------------------------------


class TestImportSGLang:
    def test_import_from_file(self, tmp_path: Path):
        data = _make_sglang_data(5)
        p = tmp_path / "sglang_bench.json"
        p.write_text(json.dumps(data))
        config = SGLangImportConfig(
            num_prefill_instances=2, num_decode_instances=4, format="sglang"
        )
        result = import_sglang(p, config)
        assert result.source_format == "sglang"
        assert result.num_requests == 5
        assert result.benchmark_data.metadata.num_prefill_instances == 2
        assert result.benchmark_data.metadata.num_decode_instances == 4
        assert result.benchmark_data.metadata.total_instances == 6

    def test_auto_detect_sglang(self):
        data = _make_sglang_data(3)
        config = SGLangImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="auto"
        )
        result = import_sglang_data(data, config)
        assert result.source_format == "sglang"
        assert result.num_requests == 3

    def test_auto_detect_native(self):
        native = {
            "metadata": {
                "num_prefill_instances": 2,
                "num_decode_instances": 4,
                "total_instances": 6,
                "measured_qps": 10.0,
            },
            "requests": [
                {
                    "request_id": "1",
                    "prompt_tokens": 128,
                    "output_tokens": 64,
                    "ttft_ms": 50.0,
                    "tpot_ms": 20.0,
                    "total_latency_ms": 1000.0,
                    "timestamp": 0.0,
                }
            ],
        }
        config = SGLangImportConfig(
            num_prefill_instances=2, num_decode_instances=4, format="auto"
        )
        result = import_sglang_data(native, config)
        assert result.source_format == "native"

    def test_auto_detect_unknown_raises(self):
        config = SGLangImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="auto"
        )
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            import_sglang_data({"random": "data"}, config)

    def test_measured_qps_computed(self):
        data = _make_sglang_data(10, qps=5.0)
        config = SGLangImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="sglang"
        )
        result = import_sglang_data(data, config)
        assert result.benchmark_data.metadata.measured_qps > 0

    def test_with_failed_requests(self):
        data = _make_sglang_data(5)
        data.append(_make_sglang_request(success=False, start_time=2000.0))
        config = SGLangImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="sglang"
        )
        result = import_sglang_data(data, config)
        assert result.num_requests == 5
        assert result.num_failed_filtered == 1
        assert len(result.warnings) > 0

    def test_non_list_raises(self):
        config = SGLangImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="sglang"
        )
        with pytest.raises(ValueError, match="JSON array"):
            import_sglang_data({"not": "a list"}, config)

    def test_output_file(self, tmp_path: Path):
        data = _make_sglang_data(3)
        infile = tmp_path / "in.json"
        infile.write_text(json.dumps(data))
        config = SGLangImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="sglang"
        )
        result = import_sglang(infile, config)
        # Write output manually (as CLI would)
        outfile = tmp_path / "out.json"
        outfile.write_text(result.benchmark_data.model_dump_json(indent=2))
        loaded = json.loads(outfile.read_text())
        assert "metadata" in loaded
        assert "requests" in loaded
        assert len(loaded["requests"]) == 3


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class TestResultModel:
    def test_result_fields(self):
        data = _make_sglang_data(3)
        config = SGLangImportConfig(
            num_prefill_instances=2, num_decode_instances=2, format="sglang"
        )
        result = import_sglang_data(data, config)
        assert isinstance(result, SGLangImportResult)
        assert result.source_format == "sglang"
        assert result.num_failed_filtered == 0
        assert isinstance(result.warnings, list)

    def test_result_serializable(self):
        data = _make_sglang_data(3)
        config = SGLangImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="sglang"
        )
        result = import_sglang_data(data, config)
        d = result.model_dump()
        assert "benchmark_data" in d
        assert "num_requests" in d
        assert "num_failed_filtered" in d

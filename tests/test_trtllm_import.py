"""Tests for TensorRT-LLM Benchmark Format Importer (M112)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.trtllm_import import (
    TRTLLMImportConfig,
    TRTLLMImportResult,
    TRTLLMRequest,
    _convert_requests,
    _detect_trtllm_format,
    import_trtllm,
    import_trtllm_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trtllm_request(
    input_tokens: int = 128,
    output_tokens: int = 64,
    first_token_latency: float = 0.05,
    inter_token_latencies: list[float] | None = None,
    end_to_end_latency: float = 1.0,
    timestamp: float | None = None,
    status: str = "completed",
) -> dict:
    """Build a raw TensorRT-LLM request dict."""
    r: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "first_token_latency": first_token_latency,
        "inter_token_latencies": (
            inter_token_latencies
            if inter_token_latencies is not None
            else [0.015, 0.014, 0.016]
        ),
        "end_to_end_latency": end_to_end_latency,
        "status": status,
    }
    if timestamp is not None:
        r["timestamp"] = timestamp
    return r


def _make_trtllm_data(n: int = 10, qps: float = 5.0) -> list[dict]:
    """Generate n TensorRT-LLM request dicts."""
    interval = 1.0 / qps if qps > 0 else 0.2
    return [
        _make_trtllm_request(
            input_tokens=100 + i * 10,
            output_tokens=50 + i * 5,
            first_token_latency=0.04 + i * 0.001,
            inter_token_latencies=[0.012 + i * 0.0001] * 10,
            end_to_end_latency=0.5 + i * 0.05,
            timestamp=1000.0 + i * interval,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestFormatDetection:
    def test_detect_trtllm_format(self):
        data = _make_trtllm_data(3)
        assert _detect_trtllm_format(data) is True

    def test_not_trtllm_if_empty(self):
        assert _detect_trtllm_format([]) is False

    def test_not_trtllm_if_dict(self):
        assert _detect_trtllm_format({"metadata": {}, "requests": []}) is False

    def test_not_trtllm_if_vllm_format(self):
        """vLLM uses 'prompt_len' and 'request_latency'."""
        data = [
            {
                "prompt_len": 128,
                "output_len": 64,
                "ttft": 0.05,
                "itl": 0.015,
                "request_latency": 1.0,
            }
        ]
        assert _detect_trtllm_format(data) is False

    def test_not_trtllm_if_sglang_format(self):
        """SGLang uses 'prompt_len', 'latency', and 'itl' (list)."""
        data = [
            {
                "prompt_len": 128,
                "output_len": 64,
                "ttft": 0.05,
                "itl": [0.01, 0.02],
                "latency": 1.0,
            }
        ]
        assert _detect_trtllm_format(data) is False

    def test_not_trtllm_if_itl_not_list(self):
        data = [
            {
                "input_tokens": 128,
                "output_tokens": 64,
                "first_token_latency": 0.05,
                "inter_token_latencies": 0.015,
                "end_to_end_latency": 1.0,
            }
        ]
        assert _detect_trtllm_format(data) is False

    def test_not_trtllm_if_not_list(self):
        assert _detect_trtllm_format("not a list") is False


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestModels:
    def test_trtllm_request_valid(self):
        r = TRTLLMRequest(
            input_tokens=128,
            output_tokens=64,
            first_token_latency=0.05,
            inter_token_latencies=[0.01, 0.02],
            end_to_end_latency=1.0,
        )
        assert r.input_tokens == 128
        assert len(r.inter_token_latencies) == 2
        assert r.status == "completed"

    def test_trtllm_request_invalid_input_tokens(self):
        with pytest.raises(Exception):
            TRTLLMRequest(
                input_tokens=0,
                output_tokens=64,
                first_token_latency=0.05,
                end_to_end_latency=1.0,
            )

    def test_trtllm_request_invalid_output_tokens(self):
        with pytest.raises(Exception):
            TRTLLMRequest(
                input_tokens=128,
                output_tokens=0,
                first_token_latency=0.05,
                end_to_end_latency=1.0,
            )

    def test_import_config_valid(self):
        c = TRTLLMImportConfig(num_prefill_instances=2, num_decode_instances=4)
        assert c.format == "auto"

    def test_import_config_invalid_instances(self):
        with pytest.raises(Exception):
            TRTLLMImportConfig(num_prefill_instances=0, num_decode_instances=4)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


class TestConversion:
    def test_basic_conversion(self):
        requests = [
            TRTLLMRequest(
                input_tokens=128,
                output_tokens=64,
                first_token_latency=0.05,
                inter_token_latencies=[0.01, 0.02, 0.03],
                end_to_end_latency=1.0,
                timestamp=1000.0,
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
            TRTLLMRequest(
                input_tokens=128,
                output_tokens=64,
                first_token_latency=0.05,
                end_to_end_latency=1.0,
                status="completed",
            ),
            TRTLLMRequest(
                input_tokens=128,
                output_tokens=64,
                first_token_latency=0.05,
                end_to_end_latency=1.0,
                status="failed",
            ),
        ]
        native, warnings, num_failed = _convert_requests(requests)
        assert len(native) == 1
        assert num_failed == 1
        assert any("Filtered 1 non-completed" in w for w in warnings)

    def test_timeout_requests_filtered(self):
        requests = [
            TRTLLMRequest(
                input_tokens=128,
                output_tokens=64,
                first_token_latency=0.05,
                end_to_end_latency=1.0,
                status="completed",
            ),
            TRTLLMRequest(
                input_tokens=128,
                output_tokens=64,
                first_token_latency=0.05,
                end_to_end_latency=5.0,
                status="timeout",
            ),
        ]
        native, warnings, num_failed = _convert_requests(requests)
        assert len(native) == 1
        assert num_failed == 1

    def test_all_failed_raises(self):
        requests = [
            TRTLLMRequest(
                input_tokens=128,
                output_tokens=64,
                first_token_latency=0.05,
                end_to_end_latency=1.0,
                status="failed",
            ),
        ]
        with pytest.raises(ValueError, match="No completed requests"):
            _convert_requests(requests)

    def test_empty_itl_fallback(self):
        requests = [
            TRTLLMRequest(
                input_tokens=128,
                output_tokens=10,
                first_token_latency=0.05,
                inter_token_latencies=[],
                end_to_end_latency=1.0,
                timestamp=0.0,
            )
        ]
        native, warnings, _ = _convert_requests(requests)
        # TPOT fallback = (1.0 - 0.05) / (10 - 1) seconds = 0.10556 s = 105.56 ms
        expected_tpot = (1.0 - 0.05) / 9.0 * 1000.0
        assert native[0].tpot_ms == pytest.approx(expected_tpot, rel=1e-3)
        assert any("empty inter_token_latencies" in w for w in warnings)

    def test_single_output_token_tpot_zero(self):
        requests = [
            TRTLLMRequest(
                input_tokens=128,
                output_tokens=1,
                first_token_latency=0.05,
                inter_token_latencies=[],
                end_to_end_latency=0.06,
                timestamp=0.0,
            )
        ]
        native, _, _ = _convert_requests(requests)
        assert native[0].tpot_ms == pytest.approx(0.0)

    def test_sequential_timestamps_when_missing(self):
        requests = [
            TRTLLMRequest(
                input_tokens=128,
                output_tokens=64,
                first_token_latency=0.05,
                end_to_end_latency=1.0,
            ),
            TRTLLMRequest(
                input_tokens=256,
                output_tokens=32,
                first_token_latency=0.03,
                end_to_end_latency=0.8,
            ),
        ]
        native, _, _ = _convert_requests(requests)
        assert native[0].timestamp == 0.0
        assert native[1].timestamp == 1.0


# ---------------------------------------------------------------------------
# End-to-end import
# ---------------------------------------------------------------------------


class TestImportTRTLLM:
    def test_import_from_file(self, tmp_path: Path):
        data = _make_trtllm_data(5)
        p = tmp_path / "trtllm_bench.json"
        p.write_text(json.dumps(data))
        config = TRTLLMImportConfig(
            num_prefill_instances=2, num_decode_instances=4, format="trtllm"
        )
        result = import_trtllm(p, config)
        assert result.source_format == "trtllm"
        assert result.num_requests == 5
        assert result.benchmark_data.metadata.num_prefill_instances == 2
        assert result.benchmark_data.metadata.num_decode_instances == 4
        assert result.benchmark_data.metadata.total_instances == 6

    def test_auto_detect_trtllm(self):
        data = _make_trtllm_data(3)
        config = TRTLLMImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="auto"
        )
        result = import_trtllm_data(data, config)
        assert result.source_format == "trtllm"
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
        config = TRTLLMImportConfig(
            num_prefill_instances=2, num_decode_instances=4, format="auto"
        )
        result = import_trtllm_data(native, config)
        assert result.source_format == "native"

    def test_auto_detect_unknown_raises(self):
        config = TRTLLMImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="auto"
        )
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            import_trtllm_data({"random": "data"}, config)

    def test_measured_qps_computed(self):
        data = _make_trtllm_data(10, qps=5.0)
        config = TRTLLMImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="trtllm"
        )
        result = import_trtllm_data(data, config)
        assert result.benchmark_data.metadata.measured_qps > 0

    def test_with_failed_requests(self):
        data = _make_trtllm_data(5)
        data.append(_make_trtllm_request(status="failed", timestamp=2000.0))
        config = TRTLLMImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="trtllm"
        )
        result = import_trtllm_data(data, config)
        assert result.num_requests == 5
        assert result.num_failed_filtered == 1
        assert len(result.warnings) > 0

    def test_non_list_raises(self):
        config = TRTLLMImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="trtllm"
        )
        with pytest.raises(ValueError, match="JSON array"):
            import_trtllm_data({"not": "a list"}, config)

    def test_output_file(self, tmp_path: Path):
        data = _make_trtllm_data(3)
        infile = tmp_path / "in.json"
        infile.write_text(json.dumps(data))
        config = TRTLLMImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="trtllm"
        )
        result = import_trtllm(infile, config)
        outfile = tmp_path / "out.json"
        outfile.write_text(result.benchmark_data.model_dump_json(indent=2))
        loaded = json.loads(outfile.read_text())
        assert "metadata" in loaded
        assert "requests" in loaded
        assert len(loaded["requests"]) == 3

    def test_single_request(self):
        data = [_make_trtllm_request(timestamp=100.0)]
        config = TRTLLMImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="trtllm"
        )
        result = import_trtllm_data(data, config)
        assert result.num_requests == 1
        # Single request → QPS defaults to 1.0 (cannot compute from duration)
        assert result.benchmark_data.metadata.measured_qps > 0

    def test_file_not_found(self, tmp_path: Path):
        config = TRTLLMImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="trtllm"
        )
        with pytest.raises(FileNotFoundError):
            import_trtllm(tmp_path / "nonexistent.json", config)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class TestResultModel:
    def test_result_fields(self):
        data = _make_trtllm_data(3)
        config = TRTLLMImportConfig(
            num_prefill_instances=2, num_decode_instances=2, format="trtllm"
        )
        result = import_trtllm_data(data, config)
        assert isinstance(result, TRTLLMImportResult)
        assert result.source_format == "trtllm"
        assert result.num_failed_filtered == 0
        assert isinstance(result.warnings, list)

    def test_result_serializable(self):
        data = _make_trtllm_data(3)
        config = TRTLLMImportConfig(
            num_prefill_instances=1, num_decode_instances=1, format="trtllm"
        )
        result = import_trtllm_data(data, config)
        d = result.model_dump()
        assert "benchmark_data" in d
        assert "num_requests" in d
        assert "num_failed_filtered" in d


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIImport:
    def test_format_choices_include_trtllm(self):
        """Verify trtllm is in the CLI format choices."""
        import argparse

        from xpyd_plan.cli._import import add_import_parser

        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        add_import_parser(subs)
        # Parse with --format trtllm to verify it's accepted
        args = parser.parse_args([
            "import",
            "--input", "dummy.json",
            "--prefill-instances", "1",
            "--decode-instances", "1",
            "--format", "trtllm",
        ])
        assert args.format == "trtllm"

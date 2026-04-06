"""TensorRT-LLM Benchmark Format Importer — import TensorRT-LLM benchmark output.

Converts TensorRT-LLM benchmark JSON output (list of request dicts) into
the native xpyd-plan BenchmarkData format for downstream analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest


class TRTLLMRequest(BaseModel):
    """A single request record from TensorRT-LLM benchmark output."""

    input_tokens: int = Field(..., ge=1, description="Number of input/prompt tokens")
    output_tokens: int = Field(
        ..., ge=1, description="Number of output/generated tokens"
    )
    first_token_latency: float = Field(
        ..., ge=0, description="Time to first token (seconds)"
    )
    inter_token_latencies: list[float] = Field(
        default_factory=list,
        description="List of inter-token latencies (seconds)",
    )
    end_to_end_latency: float = Field(
        ..., ge=0, description="Total request latency (seconds)"
    )
    timestamp: float | None = Field(
        None, description="Request start timestamp (epoch seconds)"
    )
    status: str = Field(
        "completed", description="Request status (completed/failed/timeout)"
    )


class TRTLLMBenchmarkData(BaseModel):
    """Parsed TensorRT-LLM benchmark data (list of request records)."""

    requests: list[TRTLLMRequest] = Field(..., min_length=1)


class TRTLLMImportConfig(BaseModel):
    """Configuration for TensorRT-LLM import."""

    num_prefill_instances: int = Field(
        ..., ge=1, description="Number of prefill instances"
    )
    num_decode_instances: int = Field(
        ..., ge=1, description="Number of decode instances"
    )
    format: str = Field(
        "auto", description="Input format: 'trtllm', 'native', or 'auto'"
    )


class TRTLLMImportResult(BaseModel):
    """Result of a TensorRT-LLM import operation."""

    benchmark_data: BenchmarkData
    source_format: str = Field(..., description="Detected source format")
    num_requests: int = Field(..., ge=1, description="Number of requests imported")
    num_failed_filtered: int = Field(
        0, description="Number of failed requests filtered out"
    )
    warnings: list[str] = Field(default_factory=list, description="Import warnings")


def _detect_trtllm_format(data: Any) -> bool:
    """Detect whether data is TensorRT-LLM format.

    TensorRT-LLM format: list of dicts with 'input_tokens',
    'end_to_end_latency', and 'inter_token_latencies' (list) keys.
    """
    if not isinstance(data, list) or len(data) == 0:
        return False
    first = data[0]
    if not isinstance(first, dict):
        return False
    has_input_tokens = "input_tokens" in first
    has_e2e = "end_to_end_latency" in first
    has_itl = "inter_token_latencies" in first and isinstance(
        first.get("inter_token_latencies"), list
    )
    return has_input_tokens and has_e2e and has_itl


def _convert_requests(
    trtllm_requests: list[TRTLLMRequest],
) -> tuple[list[BenchmarkRequest], list[str], int]:
    """Convert TensorRT-LLM requests to native format.

    Returns (native_requests, warnings, num_failed_filtered).
    """
    warnings: list[str] = []
    native_requests: list[BenchmarkRequest] = []

    # Filter non-completed requests
    successful = [r for r in trtllm_requests if r.status == "completed"]
    num_failed = len(trtllm_requests) - len(successful)
    if num_failed > 0:
        warnings.append(
            f"Filtered {num_failed} non-completed request(s) "
            f"(status != 'completed')"
        )

    if not successful:
        raise ValueError(
            "No completed requests to import — all requests have non-completed status"
        )

    has_timestamps = any(r.timestamp is not None for r in successful)
    warned_empty_itl = False

    for i, req in enumerate(successful):
        # Compute TPOT from inter_token_latencies (mean)
        if req.inter_token_latencies and len(req.inter_token_latencies) > 0:
            tpot_s = sum(req.inter_token_latencies) / len(
                req.inter_token_latencies
            )
        else:
            # Fallback: estimate from (e2e - ftl) / (output_tokens - 1)
            if req.output_tokens > 1:
                tpot_s = (req.end_to_end_latency - req.first_token_latency) / (
                    req.output_tokens - 1
                )
            else:
                tpot_s = 0.0
            if not warned_empty_itl:
                warnings.append(
                    "Some requests have empty inter_token_latencies, "
                    "estimated TPOT from latency"
                )
                warned_empty_itl = True

        # Determine timestamp
        if req.timestamp is not None:
            timestamp = req.timestamp
        elif has_timestamps:
            timestamp = 0.0
        else:
            timestamp = float(i)

        native_requests.append(
            BenchmarkRequest(
                request_id=str(i + 1),
                prompt_tokens=req.input_tokens,
                output_tokens=req.output_tokens,
                ttft_ms=req.first_token_latency * 1000.0,
                tpot_ms=tpot_s * 1000.0,
                total_latency_ms=req.end_to_end_latency * 1000.0,
                timestamp=timestamp,
            )
        )

    return native_requests, warnings, num_failed


def import_trtllm(
    path: str | Path,
    config: TRTLLMImportConfig,
) -> TRTLLMImportResult:
    """Import a TensorRT-LLM benchmark file and convert to native format.

    Parameters
    ----------
    path
        Path to the TensorRT-LLM benchmark JSON file.
    config
        Import configuration (instance counts, format hint).

    Returns
    -------
    TRTLLMImportResult
        Converted benchmark data with metadata.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return import_trtllm_data(raw, config)


def import_trtllm_data(
    data: Any,
    config: TRTLLMImportConfig,
) -> TRTLLMImportResult:
    """Import TensorRT-LLM benchmark data from a parsed JSON object.

    Parameters
    ----------
    data
        Parsed JSON data (expected: list of request dicts).
    config
        Import configuration.

    Returns
    -------
    TRTLLMImportResult
        Converted benchmark data with metadata.
    """
    # Format detection
    if config.format == "auto":
        if _detect_trtllm_format(data):
            source_format = "trtllm"
        elif isinstance(data, dict) and "metadata" in data and "requests" in data:
            source_format = "native"
        else:
            raise ValueError(
                "Cannot auto-detect format. Use --format trtllm or --format native."
            )
    else:
        source_format = config.format

    if source_format == "native":
        benchmark_data = BenchmarkData.model_validate(data)
        return TRTLLMImportResult(
            benchmark_data=benchmark_data,
            source_format="native",
            num_requests=len(benchmark_data.requests),
            num_failed_filtered=0,
            warnings=[],
        )

    # Parse TensorRT-LLM format
    if not isinstance(data, list):
        raise ValueError(
            "TensorRT-LLM format expects a JSON array of request objects"
        )

    trtllm_data = TRTLLMBenchmarkData(
        requests=[TRTLLMRequest(**r) for r in data]
    )
    native_requests, warnings, num_failed = _convert_requests(trtllm_data.requests)

    # Compute measured QPS
    if len(native_requests) >= 2:
        timestamps = [r.timestamp for r in native_requests]
        duration = max(timestamps) - min(timestamps)
        measured_qps = len(native_requests) / duration if duration > 0 else 1.0
    else:
        measured_qps = 1.0

    total_instances = config.num_prefill_instances + config.num_decode_instances

    benchmark_data = BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=config.num_prefill_instances,
            num_decode_instances=config.num_decode_instances,
            total_instances=total_instances,
            measured_qps=measured_qps,
        ),
        requests=native_requests,
    )

    return TRTLLMImportResult(
        benchmark_data=benchmark_data,
        source_format="trtllm",
        num_requests=len(native_requests),
        num_failed_filtered=num_failed,
        warnings=warnings,
    )

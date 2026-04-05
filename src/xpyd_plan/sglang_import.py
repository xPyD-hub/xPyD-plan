"""SGLang Benchmark Format Importer — import sglang.bench_serving output.

Converts SGLang benchmark JSON output (list of RequestFuncOutput dicts) into
the native xpyd-plan BenchmarkData format for downstream analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest


class SGLangRequest(BaseModel):
    """A single request record from SGLang bench_serving output."""

    prompt_len: int = Field(..., ge=1, description="Number of prompt tokens")
    output_len: int = Field(..., ge=1, description="Number of output/generated tokens")
    ttft: float = Field(..., ge=0, description="Time to first token (seconds)")
    itl: list[float] = Field(
        default_factory=list,
        description="List of inter-token latencies (seconds)",
    )
    latency: float = Field(..., ge=0, description="Total request latency (seconds)")
    start_time: float | None = Field(
        None, description="Request start timestamp (epoch seconds)"
    )
    success: bool = Field(True, description="Whether the request succeeded")


class SGLangBenchmarkData(BaseModel):
    """Parsed SGLang benchmark data (list of request records)."""

    requests: list[SGLangRequest] = Field(..., min_length=1)


class SGLangImportConfig(BaseModel):
    """Configuration for SGLang import."""

    num_prefill_instances: int = Field(
        ..., ge=1, description="Number of prefill instances"
    )
    num_decode_instances: int = Field(
        ..., ge=1, description="Number of decode instances"
    )
    format: str = Field(
        "auto", description="Input format: 'sglang', 'native', or 'auto'"
    )


class SGLangImportResult(BaseModel):
    """Result of an SGLang import operation."""

    benchmark_data: BenchmarkData
    source_format: str = Field(..., description="Detected source format")
    num_requests: int = Field(..., ge=1, description="Number of requests imported")
    num_failed_filtered: int = Field(
        0, description="Number of failed requests filtered out"
    )
    warnings: list[str] = Field(default_factory=list, description="Import warnings")


def _detect_sglang_format(data: Any) -> bool:
    """Detect whether data is SGLang format.

    SGLang format: list of dicts with 'prompt_len', 'latency', and 'itl' (list) keys.
    """
    if not isinstance(data, list) or len(data) == 0:
        return False
    first = data[0]
    if not isinstance(first, dict):
        return False
    # SGLang has 'latency' (not 'request_latency') and 'itl' as a list
    has_latency = "latency" in first
    has_prompt_len = "prompt_len" in first
    has_itl = "itl" in first and isinstance(first.get("itl"), list)
    # Distinguish from vLLM: vLLM uses 'request_latency', SGLang uses 'latency'
    # and SGLang's itl is a list while vLLM's is a scalar or absent
    not_vllm = "request_latency" not in first
    return has_latency and has_prompt_len and has_itl and not_vllm


def _convert_requests(
    sglang_requests: list[SGLangRequest],
) -> tuple[list[BenchmarkRequest], list[str], int]:
    """Convert SGLang requests to native format.

    Returns (native_requests, warnings, num_failed_filtered).
    """
    warnings: list[str] = []
    native_requests: list[BenchmarkRequest] = []
    num_failed = 0

    # Filter failed requests
    successful = [r for r in sglang_requests if r.success]
    num_failed = len(sglang_requests) - len(successful)
    if num_failed > 0:
        warnings.append(
            f"Filtered {num_failed} failed request(s) (success=false)"
        )

    if not successful:
        raise ValueError(
            "No successful requests to import — all requests have success=false"
        )

    # Generate sequential timestamps if missing
    has_timestamps = any(r.start_time is not None for r in successful)

    for i, req in enumerate(successful):
        # Compute TPOT from itl list (mean of inter-token latencies)
        if req.itl and len(req.itl) > 0:
            tpot_s = sum(req.itl) / len(req.itl)
        else:
            # Fallback: estimate from (latency - ttft) / output_len
            if req.output_len > 1:
                tpot_s = (req.latency - req.ttft) / (req.output_len - 1)
            else:
                tpot_s = 0.0
            warnings.append(
                f"Request {i}: empty itl list, estimated TPOT from latency"
            ) if i == 0 else None  # Only warn once

        # Determine timestamp
        if req.start_time is not None:
            timestamp = req.start_time
        elif has_timestamps:
            timestamp = 0.0  # Placeholder
        else:
            timestamp = float(i)

        native_requests.append(
            BenchmarkRequest(
                request_id=str(i + 1),
                prompt_tokens=req.prompt_len,
                output_tokens=req.output_len,
                ttft_ms=req.ttft * 1000.0,
                tpot_ms=tpot_s * 1000.0,
                total_latency_ms=req.latency * 1000.0,
                timestamp=timestamp,
            )
        )

    # Deduplicate warnings
    seen: set[str] = set()
    unique_warnings: list[str] = []
    for w in warnings:
        if w and w not in seen:
            seen.add(w)
            unique_warnings.append(w)

    return native_requests, unique_warnings, num_failed


def import_sglang(
    path: str | Path,
    config: SGLangImportConfig,
) -> SGLangImportResult:
    """Import an SGLang benchmark file and convert to native format.

    Parameters
    ----------
    path
        Path to the SGLang benchmark JSON file.
    config
        Import configuration (instance counts, format hint).

    Returns
    -------
    SGLangImportResult
        Converted benchmark data with metadata.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return import_sglang_data(raw, config)


def import_sglang_data(
    data: Any,
    config: SGLangImportConfig,
) -> SGLangImportResult:
    """Import SGLang benchmark data from a parsed JSON object.

    Parameters
    ----------
    data
        Parsed JSON data (expected: list of request dicts).
    config
        Import configuration.

    Returns
    -------
    SGLangImportResult
        Converted benchmark data with metadata.
    """
    # Format detection
    if config.format == "auto":
        if _detect_sglang_format(data):
            source_format = "sglang"
        elif isinstance(data, dict) and "metadata" in data and "requests" in data:
            source_format = "native"
        else:
            raise ValueError(
                "Cannot auto-detect format. Use --format sglang or --format native."
            )
    else:
        source_format = config.format

    if source_format == "native":
        benchmark_data = BenchmarkData.model_validate(data)
        return SGLangImportResult(
            benchmark_data=benchmark_data,
            source_format="native",
            num_requests=len(benchmark_data.requests),
            num_failed_filtered=0,
            warnings=[],
        )

    # Parse SGLang format
    if not isinstance(data, list):
        raise ValueError("SGLang format expects a JSON array of request objects")

    sglang_data = SGLangBenchmarkData(requests=[SGLangRequest(**r) for r in data])
    native_requests, warnings, num_failed = _convert_requests(sglang_data.requests)

    # Compute measured QPS
    if len(native_requests) >= 2:
        timestamps = [r.timestamp for r in native_requests]
        duration = max(timestamps) - min(timestamps)
        measured_qps = len(native_requests) / duration if duration > 0 else 0.0
    else:
        measured_qps = 0.0

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

    return SGLangImportResult(
        benchmark_data=benchmark_data,
        source_format="sglang",
        num_requests=len(native_requests),
        num_failed_filtered=num_failed,
        warnings=warnings,
    )

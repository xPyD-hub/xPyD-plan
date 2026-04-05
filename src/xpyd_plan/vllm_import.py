"""vLLM Benchmark Format Importer — import vLLM benchmark_serving.py output.

Converts vLLM benchmark JSON output (list of request dicts) into the native
xpyd-plan BenchmarkData format for downstream analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest


class VLLMRequest(BaseModel):
    """A single request record from vLLM benchmark_serving.py output."""

    prompt_len: int = Field(..., ge=1, description="Number of prompt tokens")
    output_len: int = Field(..., ge=1, description="Number of output/generated tokens")
    ttft: float = Field(..., ge=0, description="Time to first token (seconds)")
    tpot: float | None = Field(None, ge=0, description="Time per output token (seconds)")
    itl: float | None = Field(None, ge=0, description="Inter-token latency (seconds)")
    request_latency: float = Field(..., ge=0, description="Total request latency (seconds)")
    timestamp: float | None = Field(None, description="Request start timestamp (epoch seconds)")


class VLLMBenchmarkData(BaseModel):
    """Parsed vLLM benchmark data (list of request records)."""

    requests: list[VLLMRequest] = Field(..., min_length=1)


class ImportConfig(BaseModel):
    """Configuration for vLLM import."""

    num_prefill_instances: int = Field(..., ge=1, description="Number of prefill instances")
    num_decode_instances: int = Field(..., ge=1, description="Number of decode instances")
    format: str = Field("auto", description="Input format: 'vllm', 'native', or 'auto'")


class ImportResult(BaseModel):
    """Result of a vLLM import operation."""

    benchmark_data: BenchmarkData
    source_format: str = Field(..., description="Detected source format")
    num_requests: int = Field(..., ge=1, description="Number of requests imported")
    warnings: list[str] = Field(default_factory=list, description="Import warnings")


def _detect_format(data: Any) -> str:
    """Detect whether data is vLLM format or native format.

    vLLM format: list of dicts with 'prompt_len' and 'request_latency' keys.
    Native format: dict with 'metadata' and 'requests' keys.

    Returns
    -------
    str
        'vllm' or 'native'.

    Raises
    ------
    ValueError
        If the format cannot be determined.
    """
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("Empty list: cannot detect format")
        first = data[0]
        if isinstance(first, dict) and "prompt_len" in first and "request_latency" in first:
            return "vllm"
        raise ValueError(
            "List format detected but missing expected vLLM fields "
            "(prompt_len, request_latency)"
        )
    if isinstance(data, dict):
        if "metadata" in data and "requests" in data:
            return "native"
        # Could be a vLLM wrapper with a results key
        if "results" in data and isinstance(data["results"], list):
            return "vllm"
        raise ValueError(
            "Dict format detected but missing expected fields "
            "(metadata+requests for native, or results for vLLM wrapper)"
        )
    raise ValueError(f"Unexpected data type: {type(data).__name__}")


def _extract_vllm_requests(data: Any) -> list[dict[str, Any]]:
    """Extract the request list from vLLM data (handles both list and wrapper dict)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    raise ValueError("Cannot extract vLLM request list from data")


def _parse_vllm_request(raw: dict[str, Any], index: int) -> tuple[VLLMRequest, list[str]]:
    """Parse a single vLLM request dict, handling missing/alternative fields.

    Returns (VLLMRequest, warnings).
    """
    warnings: list[str] = []

    prompt_len = raw.get("prompt_len")
    if prompt_len is None:
        raise ValueError(f"Request {index}: missing required field 'prompt_len'")

    output_len = raw.get("output_len")
    if output_len is None:
        raise ValueError(f"Request {index}: missing required field 'output_len'")

    request_latency = raw.get("request_latency")
    if request_latency is None:
        raise ValueError(f"Request {index}: missing required field 'request_latency'")

    ttft = raw.get("ttft")
    if ttft is None:
        # Fall back: estimate TTFT as 10% of request_latency
        ttft = request_latency * 0.1
        warnings.append(f"Request {index}: missing 'ttft', estimated from request_latency")

    # tpot: try tpot first, then itl
    tpot = raw.get("tpot")
    itl = raw.get("itl")
    if tpot is None and itl is not None:
        tpot = itl
    if tpot is None and output_len > 1:
        # Estimate from (total - ttft) / (output_tokens - 1)
        remaining = request_latency - ttft
        tpot = remaining / (output_len - 1) if output_len > 1 else 0.0
        warnings.append(f"Request {index}: missing 'tpot'/'itl', estimated from latency")

    timestamp = raw.get("timestamp")

    return VLLMRequest(
        prompt_len=int(prompt_len),
        output_len=int(output_len),
        ttft=float(ttft),
        tpot=float(tpot) if tpot is not None else None,
        itl=float(itl) if itl is not None else None,
        request_latency=float(request_latency),
        timestamp=float(timestamp) if timestamp is not None else None,
    ), warnings


class VLLMImporter:
    """Import vLLM benchmark_serving.py output into native BenchmarkData format.

    Parameters
    ----------
    num_prefill_instances : int
        Number of prefill instances in the cluster.
    num_decode_instances : int
        Number of decode instances in the cluster.
    """

    def __init__(
        self,
        num_prefill_instances: int,
        num_decode_instances: int,
    ) -> None:
        if num_prefill_instances < 1:
            raise ValueError("num_prefill_instances must be >= 1")
        if num_decode_instances < 1:
            raise ValueError("num_decode_instances must be >= 1")
        self._prefill = num_prefill_instances
        self._decode = num_decode_instances

    def import_data(self, data: Any) -> ImportResult:
        """Import vLLM benchmark data from parsed JSON.

        Parameters
        ----------
        data : Any
            Parsed JSON data — either a list of request dicts or a wrapper dict.

        Returns
        -------
        ImportResult
            The converted benchmark data and metadata.
        """
        raw_requests = _extract_vllm_requests(data)
        if not raw_requests:
            raise ValueError("No requests found in vLLM data")

        all_warnings: list[str] = []
        vllm_requests: list[VLLMRequest] = []

        for i, raw in enumerate(raw_requests):
            req, warns = _parse_vllm_request(raw, i)
            vllm_requests.append(req)
            all_warnings.extend(warns)

        # Convert to native BenchmarkRequest format
        benchmark_requests: list[BenchmarkRequest] = []
        has_timestamps = any(r.timestamp is not None for r in vllm_requests)

        for i, vr in enumerate(vllm_requests):
            # Generate sequential timestamps if missing
            if vr.timestamp is not None:
                ts = vr.timestamp
            elif has_timestamps:
                # Some have timestamps, some don't — use 0.0 for missing
                ts = 0.0
                all_warnings.append(
                    f"Request {i}: missing timestamp while others have it, using 0.0"
                )
            else:
                # No timestamps at all — generate sequential
                ts = float(i)

            # vLLM times are in seconds, convert to ms
            ttft_ms = vr.ttft * 1000.0
            total_latency_ms = vr.request_latency * 1000.0

            # TPOT: use tpot if available, otherwise estimate
            if vr.tpot is not None:
                tpot_ms = vr.tpot * 1000.0
            elif vr.output_len > 1:
                tpot_ms = (total_latency_ms - ttft_ms) / (vr.output_len - 1)
            else:
                tpot_ms = 0.0

            benchmark_requests.append(
                BenchmarkRequest(
                    request_id=f"vllm-{i:06d}",
                    prompt_tokens=vr.prompt_len,
                    output_tokens=vr.output_len,
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    total_latency_ms=total_latency_ms,
                    timestamp=ts,
                )
            )

        # Compute measured QPS
        if len(benchmark_requests) >= 2 and has_timestamps:
            timestamps = [r.timestamp for r in benchmark_requests]
            duration = max(timestamps) - min(timestamps)
            measured_qps = (
                len(benchmark_requests) / duration
                if duration > 0
                else float(len(benchmark_requests))
            )
        else:
            # Estimate from total latencies
            total_time = sum(vr.request_latency for vr in vllm_requests)
            measured_qps = (
                len(vllm_requests) / (total_time / len(vllm_requests))
                if total_time > 0
                else 1.0
            )

        metadata = BenchmarkMetadata(
            num_prefill_instances=self._prefill,
            num_decode_instances=self._decode,
            total_instances=self._prefill + self._decode,
            measured_qps=measured_qps,
        )

        benchmark_data = BenchmarkData(
            metadata=metadata,
            requests=benchmark_requests,
        )

        return ImportResult(
            benchmark_data=benchmark_data,
            source_format="vllm",
            num_requests=len(benchmark_requests),
            warnings=all_warnings,
        )

    def import_file(self, path: str | Path) -> ImportResult:
        """Import from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to the vLLM benchmark JSON file.

        Returns
        -------
        ImportResult
        """
        path = Path(path)
        with path.open() as f:
            data = json.load(f)
        return self.import_data(data)


def import_vllm(
    input_path: str | Path,
    num_prefill_instances: int,
    num_decode_instances: int,
    output_path: str | Path | None = None,
    format: str = "auto",
) -> ImportResult:
    """Import vLLM benchmark data and optionally save as native format.

    This is the programmatic API for vLLM benchmark import.

    Parameters
    ----------
    input_path : str or Path
        Path to the input benchmark JSON file.
    num_prefill_instances : int
        Number of prefill instances in the cluster.
    num_decode_instances : int
        Number of decode instances in the cluster.
    output_path : str or Path or None
        If provided, save the converted data to this path.
    format : str
        Input format: 'vllm', 'native', or 'auto' (default).

    Returns
    -------
    ImportResult
    """
    input_path = Path(input_path)
    with input_path.open() as f:
        data = json.load(f)

    # Detect or use specified format
    if format == "auto":
        detected = _detect_format(data)
    else:
        detected = format

    if detected == "native":
        # Already native — just load it
        from .bench_adapter import load_benchmark_auto

        benchmark_data = load_benchmark_auto(str(input_path))
        return ImportResult(
            benchmark_data=benchmark_data,
            source_format="native",
            num_requests=len(benchmark_data.requests),
            warnings=[],
        )

    importer = VLLMImporter(
        num_prefill_instances=num_prefill_instances,
        num_decode_instances=num_decode_instances,
    )
    result = importer.import_data(data)

    if output_path is not None:
        output_path = Path(output_path)
        with output_path.open("w") as f:
            json.dump(result.benchmark_data.model_dump(), f, indent=2)
            f.write("\n")

    return result

"""Adapter for xpyd-bench output format — converts to internal BenchmarkData.

Supports schema version auto-detection and format conversion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest

# Supported schema versions
SUPPORTED_VERSIONS = {"1", "1.0"}


class UnsupportedSchemaVersion(ValueError):
    """Raised when the benchmark data schema version is not supported."""

    def __init__(self, version: str) -> None:
        self.version = version
        super().__init__(
            f"Unsupported schema version: {version!r}. "
            f"Supported versions: {', '.join(sorted(SUPPORTED_VERSIONS))}"
        )


class FormatDetectionError(ValueError):
    """Raised when the format of benchmark data cannot be determined."""


def detect_schema_version(data: dict[str, Any]) -> str:
    """Detect the schema version of benchmark data.

    Auto-detection logic:
    - If "schema_version" key exists, use it directly.
    - If data has "metadata" and "requests" keys (native format), return "1".
    - If data has "bench_config" and "results" keys (xpyd-bench format), return "1".

    Args:
        data: Parsed JSON data.

    Returns:
        Detected schema version string.

    Raises:
        FormatDetectionError: If the format cannot be determined.
    """
    if "schema_version" in data:
        return str(data["schema_version"])

    # Native format (already matches BenchmarkData)
    if "metadata" in data and "requests" in data:
        return "1"

    # xpyd-bench format
    if "bench_config" in data and "results" in data:
        return "1"

    raise FormatDetectionError(
        "Cannot detect schema version. Expected 'schema_version' field, "
        "or recognized format with 'metadata'+'requests' or 'bench_config'+'results'."
    )


def detect_format(data: dict[str, Any]) -> str:
    """Detect whether data is 'native' or 'xpyd-bench' format.

    Args:
        data: Parsed JSON data.

    Returns:
        'native' or 'xpyd-bench'.

    Raises:
        FormatDetectionError: If format cannot be determined.
    """
    if "metadata" in data and "requests" in data:
        return "native"
    if "bench_config" in data and "results" in data:
        return "xpyd-bench"
    raise FormatDetectionError(
        "Cannot detect format. Expected 'metadata'+'requests' (native) "
        "or 'bench_config'+'results' (xpyd-bench)."
    )


class XpydBenchAdapter:
    """Converts xpyd-bench output format to internal BenchmarkData.

    xpyd-bench format:
    ```json
    {
        "schema_version": "1",
        "bench_config": {
            "prefill_instances": 2,
            "decode_instances": 6,
            "total_instances": 8,
            "target_qps": 10.0,
            "measured_qps": 9.8,
            "duration_seconds": 60
        },
        "results": [
            {
                "id": "req-001",
                "input_tokens": 128,
                "output_tokens": 256,
                "time_to_first_token_ms": 45.2,
                "time_per_output_token_ms": 12.1,
                "end_to_end_latency_ms": 3141.8,
                "start_time": 1700000000.0
            }
        ]
    }
    ```
    """

    def convert(self, data: dict[str, Any]) -> BenchmarkData:
        """Convert xpyd-bench format data to BenchmarkData.

        Args:
            data: Parsed xpyd-bench JSON output.

        Returns:
            BenchmarkData in native format.

        Raises:
            UnsupportedSchemaVersion: If schema version is not supported.
            KeyError: If required fields are missing.
            ValueError: If data is invalid.
        """
        version = detect_schema_version(data)
        if version not in SUPPORTED_VERSIONS:
            raise UnsupportedSchemaVersion(version)

        config = data["bench_config"]
        results = data["results"]

        metadata = BenchmarkMetadata(
            num_prefill_instances=config["prefill_instances"],
            num_decode_instances=config["decode_instances"],
            total_instances=config["total_instances"],
            measured_qps=config["measured_qps"],
        )

        requests = [
            BenchmarkRequest(
                request_id=r["id"],
                prompt_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                ttft_ms=r["time_to_first_token_ms"],
                tpot_ms=r["time_per_output_token_ms"],
                total_latency_ms=r["end_to_end_latency_ms"],
                timestamp=r["start_time"],
            )
            for r in results
        ]

        return BenchmarkData(metadata=metadata, requests=requests)

    def load(self, path: str | Path) -> BenchmarkData:
        """Load and convert xpyd-bench output file.

        Args:
            path: Path to xpyd-bench JSON output file.

        Returns:
            BenchmarkData.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        raw = json.loads(path.read_text())
        return self.convert(raw)


def load_benchmark_auto(path: str | Path) -> BenchmarkData:
    """Load benchmark data with automatic format detection.

    Detects whether the file is native format or xpyd-bench format
    and loads accordingly.

    Args:
        path: Path to benchmark JSON file.

    Returns:
        BenchmarkData.

    Raises:
        FormatDetectionError: If format cannot be determined.
        UnsupportedSchemaVersion: If schema version is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    raw = json.loads(path.read_text())
    return load_benchmark_auto_from_dict(raw)


def load_benchmark_auto_from_dict(data: dict[str, Any]) -> BenchmarkData:
    """Load benchmark data from dict with automatic format detection.

    Args:
        data: Parsed JSON data.

    Returns:
        BenchmarkData.
    """
    version = detect_schema_version(data)
    if version not in SUPPORTED_VERSIONS:
        raise UnsupportedSchemaVersion(version)

    fmt = detect_format(data)
    if fmt == "native":
        return BenchmarkData(**data)
    else:
        adapter = XpydBenchAdapter()
        return adapter.convert(data)

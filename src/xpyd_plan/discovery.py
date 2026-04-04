"""Batch benchmark discovery and auto-loading from directory trees."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkMetadata


class ValidationStatus(str, Enum):
    """Quick validation result for a candidate file."""

    VALID = "valid"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"


class DiscoveredBenchmark(BaseModel):
    """Summary of a discovered benchmark file without full loading."""

    path: str = Field(..., description="Absolute path to benchmark file")
    status: ValidationStatus = Field(..., description="Quick validation result")
    num_requests: int | None = Field(None, description="Number of requests (if valid)")
    measured_qps: float | None = Field(None, description="Measured QPS (if valid)")
    num_prefill_instances: int | None = Field(None, description="Prefill instances")
    num_decode_instances: int | None = Field(None, description="Decode instances")
    total_instances: int | None = Field(None, description="Total instances")
    config_key: str | None = Field(
        None, description="Cluster config key for grouping (e.g. 'P2:D4')"
    )
    file_size_bytes: int = Field(..., description="File size in bytes")
    error: str | None = Field(None, description="Error message if invalid")


class ConfigGroup(BaseModel):
    """Group of benchmarks sharing the same cluster configuration."""

    config_key: str = Field(..., description="Cluster config key (e.g. 'P2:D4')")
    num_prefill_instances: int
    num_decode_instances: int
    total_instances: int
    benchmarks: list[DiscoveredBenchmark] = Field(default_factory=list)
    count: int = Field(0, description="Number of benchmarks in this group")


class DiscoveryReport(BaseModel):
    """Complete discovery report."""

    root_dir: str = Field(..., description="Root directory that was scanned")
    pattern: str = Field(..., description="Glob pattern used")
    max_depth: int | None = Field(None, description="Max depth limit (None = unlimited)")
    total_files_scanned: int = Field(0, description="Total JSON files found")
    valid_count: int = Field(0, description="Number of valid benchmark files")
    invalid_count: int = Field(0, description="Number of invalid files")
    benchmarks: list[DiscoveredBenchmark] = Field(default_factory=list)
    groups: list[ConfigGroup] = Field(default_factory=list)


def _config_key(meta: BenchmarkMetadata) -> str:
    """Create a grouping key from cluster config."""
    return f"P{meta.num_prefill_instances}:D{meta.num_decode_instances}"


def _quick_validate(path: Path) -> DiscoveredBenchmark:
    """Quickly validate a file as a benchmark without full Pydantic parsing."""
    file_size = path.stat().st_size

    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return DiscoveredBenchmark(
            path=str(path.resolve()),
            status=ValidationStatus.INVALID_JSON,
            file_size_bytes=file_size,
            error=str(e),
        )

    # Quick schema check: must have metadata and requests
    if not isinstance(raw, dict):
        return DiscoveredBenchmark(
            path=str(path.resolve()),
            status=ValidationStatus.INVALID_SCHEMA,
            file_size_bytes=file_size,
            error="Root is not a JSON object",
        )

    metadata_raw = raw.get("metadata")
    requests_raw = raw.get("requests")

    if not isinstance(metadata_raw, dict) or not isinstance(requests_raw, list):
        return DiscoveredBenchmark(
            path=str(path.resolve()),
            status=ValidationStatus.INVALID_SCHEMA,
            file_size_bytes=file_size,
            error="Missing or invalid 'metadata' or 'requests' field",
        )

    # Validate metadata fields exist
    required_meta = [
        "num_prefill_instances",
        "num_decode_instances",
        "total_instances",
        "measured_qps",
    ]
    for field in required_meta:
        if field not in metadata_raw:
            return DiscoveredBenchmark(
                path=str(path.resolve()),
                status=ValidationStatus.INVALID_SCHEMA,
                file_size_bytes=file_size,
                error=f"Missing metadata field: {field}",
            )

    try:
        meta = BenchmarkMetadata(**metadata_raw)
    except Exception as e:
        return DiscoveredBenchmark(
            path=str(path.resolve()),
            status=ValidationStatus.INVALID_SCHEMA,
            file_size_bytes=file_size,
            error=f"Invalid metadata: {e}",
        )

    key = _config_key(meta)

    return DiscoveredBenchmark(
        path=str(path.resolve()),
        status=ValidationStatus.VALID,
        num_requests=len(requests_raw),
        measured_qps=meta.measured_qps,
        num_prefill_instances=meta.num_prefill_instances,
        num_decode_instances=meta.num_decode_instances,
        total_instances=meta.total_instances,
        config_key=key,
        file_size_bytes=file_size,
    )


def _depth_of(path: Path, root: Path) -> int:
    """Compute the directory depth of path relative to root."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return 0
    # Number of parent directories (file itself doesn't count)
    return len(rel.parts) - 1


class BenchmarkDiscovery:
    """Discover and catalog benchmark files in a directory tree."""

    def discover(
        self,
        root_dir: str | Path,
        pattern: str = "**/*.json",
        max_depth: int | None = None,
    ) -> DiscoveryReport:
        """Scan a directory tree for benchmark files.

        Args:
            root_dir: Root directory to scan.
            pattern: Glob pattern for matching files (default: all JSON).
            max_depth: Maximum directory depth to scan (None = unlimited).

        Returns:
            DiscoveryReport with all discovered benchmarks and config groups.

        Raises:
            FileNotFoundError: If root_dir does not exist.
            NotADirectoryError: If root_dir is not a directory.
        """
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root_dir}")

        benchmarks: list[DiscoveredBenchmark] = []
        groups_map: dict[str, ConfigGroup] = {}

        candidates = sorted(root.glob(pattern))
        for candidate in candidates:
            if not candidate.is_file():
                continue

            if max_depth is not None and _depth_of(candidate, root) > max_depth:
                continue

            result = _quick_validate(candidate)
            benchmarks.append(result)

            if result.status == ValidationStatus.VALID and result.config_key:
                if result.config_key not in groups_map:
                    groups_map[result.config_key] = ConfigGroup(
                        config_key=result.config_key,
                        num_prefill_instances=result.num_prefill_instances or 0,
                        num_decode_instances=result.num_decode_instances or 0,
                        total_instances=result.total_instances or 0,
                    )
                group = groups_map[result.config_key]
                group.benchmarks.append(result)
                group.count = len(group.benchmarks)

        valid_count = sum(1 for b in benchmarks if b.status == ValidationStatus.VALID)

        return DiscoveryReport(
            root_dir=str(root.resolve()),
            pattern=pattern,
            max_depth=max_depth,
            total_files_scanned=len(benchmarks),
            valid_count=valid_count,
            invalid_count=len(benchmarks) - valid_count,
            benchmarks=benchmarks,
            groups=sorted(groups_map.values(), key=lambda g: g.config_key),
        )


def discover_benchmarks(
    root_dir: str | Path,
    pattern: str = "**/*.json",
    max_depth: int | None = None,
) -> dict:
    """Programmatic API for benchmark discovery.

    Args:
        root_dir: Root directory to scan.
        pattern: Glob pattern for matching files.
        max_depth: Maximum directory depth.

    Returns:
        Dictionary representation of the DiscoveryReport.
    """
    discoverer = BenchmarkDiscovery()
    report = discoverer.discover(root_dir, pattern=pattern, max_depth=max_depth)
    return report.model_dump()

"""Benchmark environment fingerprint for reproducibility verification."""

from __future__ import annotations

import hashlib
import json
from enum import Enum

from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class Compatibility(str, Enum):
    """Compatibility classification between two environments."""

    IDENTICAL = "IDENTICAL"
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"


class EnvironmentFingerprint(BaseModel):
    """Captured environment metadata from a benchmark run."""

    num_prefill_instances: int = Field(..., description="Number of prefill instances")
    num_decode_instances: int = Field(..., description="Number of decode instances")
    total_instances: int = Field(..., description="Total instances")
    measured_qps: float = Field(..., description="Measured QPS")
    num_requests: int = Field(..., description="Number of requests in benchmark")
    prompt_tokens_min: int = Field(..., description="Min prompt tokens across requests")
    prompt_tokens_max: int = Field(..., description="Max prompt tokens across requests")
    output_tokens_min: int = Field(..., description="Min output tokens across requests")
    output_tokens_max: int = Field(..., description="Max output tokens across requests")
    hash: str = Field(..., description="SHA-256 fingerprint hash")


class FingerprintDiff(BaseModel):
    """A single difference between two fingerprints."""

    field: str = Field(..., description="Field name that differs")
    baseline_value: str = Field(..., description="Value in baseline")
    current_value: str = Field(..., description="Value in current")


class FingerprintComparison(BaseModel):
    """Result of comparing two environment fingerprints."""

    baseline_hash: str = Field(..., description="Baseline fingerprint hash")
    current_hash: str = Field(..., description="Current fingerprint hash")
    compatibility: Compatibility = Field(..., description="Compatibility classification")
    differences: list[FingerprintDiff] = Field(
        default_factory=list, description="List of differences"
    )
    identical: bool = Field(..., description="Whether fingerprints are identical")


# Fields that make environments incompatible if different
_MAJOR_FIELDS = {"num_prefill_instances", "num_decode_instances", "total_instances"}
# Fields that are minor differences
_MINOR_FIELDS = {
    "measured_qps",
    "num_requests",
    "prompt_tokens_min",
    "prompt_tokens_max",
    "output_tokens_min",
    "output_tokens_max",
}


class EnvironmentFingerprinter:
    """Extract and compare environment fingerprints from benchmark data."""

    def extract(self, data: BenchmarkData) -> EnvironmentFingerprint:
        """Extract environment fingerprint from benchmark data."""
        prompt_tokens = [r.prompt_tokens for r in data.requests]
        output_tokens = [r.output_tokens for r in data.requests]

        fields = {
            "num_prefill_instances": data.metadata.num_prefill_instances,
            "num_decode_instances": data.metadata.num_decode_instances,
            "total_instances": data.metadata.total_instances,
            "measured_qps": round(data.metadata.measured_qps, 2),
            "num_requests": len(data.requests),
            "prompt_tokens_min": min(prompt_tokens),
            "prompt_tokens_max": max(prompt_tokens),
            "output_tokens_min": min(output_tokens),
            "output_tokens_max": max(output_tokens),
        }

        # Hash based on cluster config (major fields) for stability
        hash_input = json.dumps(
            {k: fields[k] for k in sorted(_MAJOR_FIELDS)},
            sort_keys=True,
        )
        fingerprint_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        return EnvironmentFingerprint(hash=fingerprint_hash, **fields)

    def compare(
        self,
        baseline: EnvironmentFingerprint,
        current: EnvironmentFingerprint,
    ) -> FingerprintComparison:
        """Compare two environment fingerprints."""
        differences: list[FingerprintDiff] = []
        has_major_diff = False

        all_fields = _MAJOR_FIELDS | _MINOR_FIELDS
        for field_name in sorted(all_fields):
            base_val = getattr(baseline, field_name)
            curr_val = getattr(current, field_name)
            if base_val != curr_val:
                differences.append(
                    FingerprintDiff(
                        field=field_name,
                        baseline_value=str(base_val),
                        current_value=str(curr_val),
                    )
                )
                if field_name in _MAJOR_FIELDS:
                    has_major_diff = True

        identical = len(differences) == 0
        if identical:
            compatibility = Compatibility.IDENTICAL
        elif has_major_diff:
            compatibility = Compatibility.INCOMPATIBLE
        else:
            compatibility = Compatibility.COMPATIBLE

        return FingerprintComparison(
            baseline_hash=baseline.hash,
            current_hash=current.hash,
            compatibility=compatibility,
            differences=differences,
            identical=identical,
        )


def fingerprint_benchmark(
    benchmark_path: str,
    compare_path: str | None = None,
) -> EnvironmentFingerprint | FingerprintComparison:
    """Programmatic API: fingerprint a benchmark, optionally compare two.

    Args:
        benchmark_path: Path to benchmark JSON file.
        compare_path: Optional second benchmark file to compare against.

    Returns:
        EnvironmentFingerprint if single file, FingerprintComparison if comparing two.
    """
    from xpyd_plan.bench_adapter import load_benchmark_auto

    fp = EnvironmentFingerprinter()

    data = load_benchmark_auto(benchmark_path)
    baseline_fp = fp.extract(data)

    if compare_path is None:
        return baseline_fp

    data2 = load_benchmark_auto(compare_path)
    current_fp = fp.extract(data2)

    return fp.compare(baseline_fp, current_fp)

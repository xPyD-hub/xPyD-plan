"""Benchmark filtering and slicing — select subsets of benchmark data by criteria."""

from __future__ import annotations

import random
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from .benchmark_models import BenchmarkData, BenchmarkRequest


class FilterConfig(BaseModel):
    """Configuration for filtering benchmark requests."""

    min_prompt_tokens: Optional[int] = Field(None, ge=1, description="Minimum prompt tokens")
    max_prompt_tokens: Optional[int] = Field(None, ge=1, description="Maximum prompt tokens")
    min_output_tokens: Optional[int] = Field(None, ge=1, description="Minimum output tokens")
    max_output_tokens: Optional[int] = Field(None, ge=1, description="Maximum output tokens")
    min_ttft_ms: Optional[float] = Field(None, ge=0, description="Minimum TTFT (ms)")
    max_ttft_ms: Optional[float] = Field(None, ge=0, description="Maximum TTFT (ms)")
    min_tpot_ms: Optional[float] = Field(None, ge=0, description="Minimum TPOT (ms)")
    max_tpot_ms: Optional[float] = Field(None, ge=0, description="Maximum TPOT (ms)")
    min_total_latency_ms: Optional[float] = Field(
        None, ge=0, description="Minimum total latency (ms)"
    )
    max_total_latency_ms: Optional[float] = Field(
        None, ge=0, description="Maximum total latency (ms)"
    )
    time_start: Optional[float] = Field(None, description="Start timestamp (epoch seconds)")
    time_end: Optional[float] = Field(None, description="End timestamp (epoch seconds)")
    sample_count: Optional[int] = Field(None, ge=1, description="Random sample N requests")
    sample_fraction: Optional[float] = Field(
        None, gt=0.0, le=1.0, description="Random sample fraction (0, 1]"
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducible sampling")

    @model_validator(mode="after")
    def validate_ranges(self) -> "FilterConfig":
        """Ensure min <= max for all range pairs."""
        pairs = [
            ("min_prompt_tokens", "max_prompt_tokens"),
            ("min_output_tokens", "max_output_tokens"),
            ("min_ttft_ms", "max_ttft_ms"),
            ("min_tpot_ms", "max_tpot_ms"),
            ("min_total_latency_ms", "max_total_latency_ms"),
            ("time_start", "time_end"),
        ]
        for min_field, max_field in pairs:
            min_val = getattr(self, min_field)
            max_val = getattr(self, max_field)
            if min_val is not None and max_val is not None and min_val > max_val:
                msg = f"{min_field} ({min_val}) must be <= {max_field} ({max_val})"
                raise ValueError(msg)
        if self.sample_count is not None and self.sample_fraction is not None:
            msg = "Cannot specify both sample_count and sample_fraction"
            raise ValueError(msg)
        return self


class BenchmarkFilterResult(BaseModel):
    """Result of applying a filter to benchmark data."""

    original_count: int = Field(..., description="Number of requests before filtering")
    filtered_count: int = Field(..., description="Number of requests after filtering")
    removed_count: int = Field(..., description="Number of requests removed")
    retention_rate: float = Field(..., description="Fraction of requests retained")
    filters_applied: list[str] = Field(
        default_factory=list, description="List of filter descriptions applied"
    )
    data: BenchmarkData = Field(..., description="Filtered benchmark data")


class BenchmarkFilter:
    """Filter benchmark data by configurable criteria."""

    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def apply(self, data: BenchmarkData) -> BenchmarkFilterResult:
        """Apply all configured filters to benchmark data."""
        original_count = len(data.requests)
        requests = list(data.requests)
        filters_applied: list[str] = []

        # Token count filters
        requests, desc = self._filter_range(
            requests, "prompt_tokens", self.config.min_prompt_tokens,
            self.config.max_prompt_tokens,
        )
        if desc:
            filters_applied.append(desc)

        requests, desc = self._filter_range(
            requests, "output_tokens", self.config.min_output_tokens,
            self.config.max_output_tokens,
        )
        if desc:
            filters_applied.append(desc)

        # Latency filters
        requests, desc = self._filter_range(
            requests, "ttft_ms", self.config.min_ttft_ms, self.config.max_ttft_ms,
        )
        if desc:
            filters_applied.append(desc)

        requests, desc = self._filter_range(
            requests, "tpot_ms", self.config.min_tpot_ms, self.config.max_tpot_ms,
        )
        if desc:
            filters_applied.append(desc)

        requests, desc = self._filter_range(
            requests, "total_latency_ms", self.config.min_total_latency_ms,
            self.config.max_total_latency_ms,
        )
        if desc:
            filters_applied.append(desc)

        # Time window filter
        requests, desc = self._filter_range(
            requests, "timestamp", self.config.time_start, self.config.time_end,
        )
        if desc:
            filters_applied.append(desc)

        # Sampling (applied last, after all deterministic filters)
        if self.config.sample_count is not None or self.config.sample_fraction is not None:
            rng = random.Random(self.config.seed)
            if self.config.sample_count is not None:
                n = min(self.config.sample_count, len(requests))
                requests = rng.sample(requests, n)
                filters_applied.append(f"sample_count={self.config.sample_count}")
            else:
                assert self.config.sample_fraction is not None
                n = max(1, int(len(requests) * self.config.sample_fraction))
                requests = rng.sample(requests, n)
                filters_applied.append(f"sample_fraction={self.config.sample_fraction}")

        if not requests:
            msg = "Filter removed all requests — no data remaining"
            raise ValueError(msg)

        # Adjust QPS proportionally
        retention = len(requests) / original_count if original_count > 0 else 0.0
        adjusted_qps = data.metadata.measured_qps * retention

        filtered_data = BenchmarkData(
            metadata=data.metadata.model_copy(update={"measured_qps": adjusted_qps}),
            requests=requests,
        )

        return BenchmarkFilterResult(
            original_count=original_count,
            filtered_count=len(requests),
            removed_count=original_count - len(requests),
            retention_rate=retention,
            filters_applied=filters_applied,
            data=filtered_data,
        )

    @staticmethod
    def _filter_range(
        requests: list[BenchmarkRequest],
        attr: str,
        min_val: float | int | None,
        max_val: float | int | None,
    ) -> tuple[list[BenchmarkRequest], str]:
        """Filter requests where attr is within [min_val, max_val]."""
        if min_val is None and max_val is None:
            return requests, ""

        parts: list[str] = []
        filtered = requests
        if min_val is not None:
            filtered = [r for r in filtered if getattr(r, attr) >= min_val]
            parts.append(f"{attr}>={min_val}")
        if max_val is not None:
            filtered = [r for r in filtered if getattr(r, attr) <= max_val]
            parts.append(f"{attr}<={max_val}")

        return filtered, " & ".join(parts)


def filter_benchmark(
    data: BenchmarkData,
    config: FilterConfig,
) -> BenchmarkFilterResult:
    """Programmatic API: filter benchmark data by criteria.

    Args:
        data: Benchmark data to filter.
        config: Filter configuration.

    Returns:
        BenchmarkFilterResult with filtered data and statistics.
    """
    return BenchmarkFilter(config).apply(data)

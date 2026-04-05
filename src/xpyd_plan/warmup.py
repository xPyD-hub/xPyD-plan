"""Warm-up exclusion filter — detect and remove warm-up requests from benchmark data.

Uses the timeline module's warm-up detection logic to identify the initial
period where latency is elevated, then strips those requests to produce
cleaner data for analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata


class WarmupWindow(BaseModel):
    """Detected warm-up window details."""

    start_timestamp: float = Field(..., description="Start of warm-up (epoch seconds)")
    end_timestamp: float = Field(..., description="End of warm-up (epoch seconds)")
    duration_s: float = Field(..., ge=0, description="Warm-up duration in seconds")
    requests_excluded: int = Field(..., ge=0, description="Number of requests in warm-up window")


class LatencyComparison(BaseModel):
    """Before/after latency comparison."""

    before_p50_ms: float = Field(..., ge=0)
    before_p95_ms: float = Field(..., ge=0)
    before_p99_ms: float = Field(..., ge=0)
    after_p50_ms: float = Field(..., ge=0)
    after_p95_ms: float = Field(..., ge=0)
    after_p99_ms: float = Field(..., ge=0)


class WarmupReport(BaseModel):
    """Report of warm-up filtering operation."""

    warmup_detected: bool = Field(..., description="Whether warm-up was detected")
    warmup_window: Optional[WarmupWindow] = Field(
        None, description="Warm-up window details"
    )
    original_request_count: int = Field(..., ge=0)
    filtered_request_count: int = Field(..., ge=0)
    original_qps: float = Field(..., ge=0)
    adjusted_qps: float = Field(..., ge=0)
    latency_comparison: Optional[LatencyComparison] = Field(
        None, description="Total latency comparison before/after filtering"
    )


def _percentile(values: list[float], pct: float) -> float:
    """Compute percentile from sorted values."""
    if not values:
        return 0.0
    arr = np.array(values)
    return float(np.percentile(arr, pct))


class WarmupFilter:
    """Detect and remove warm-up requests from benchmark data."""

    def __init__(
        self,
        warmup_seconds: Optional[float] = None,
        warmup_factor: float = 2.0,
        window_size_s: float = 10.0,
    ) -> None:
        """Initialize filter.

        Args:
            warmup_seconds: Fixed warm-up duration to exclude. If None, auto-detect.
            warmup_factor: P95 multiplier threshold for warm-up detection.
            window_size_s: Window size for warm-up detection.
        """
        if warmup_seconds is not None and warmup_seconds < 0:
            raise ValueError("warmup_seconds must be non-negative")
        if warmup_factor <= 0:
            raise ValueError("warmup_factor must be positive")
        if window_size_s <= 0:
            raise ValueError("window_size_s must be positive")

        self._warmup_seconds = warmup_seconds
        self._warmup_factor = warmup_factor
        self._window_size_s = window_size_s

    def filter(self, data: BenchmarkData) -> tuple[BenchmarkData, WarmupReport]:
        """Filter warm-up requests from benchmark data.

        Returns:
            Tuple of (filtered BenchmarkData, WarmupReport).
        """
        if not data.requests:
            report = WarmupReport(
                warmup_detected=False,
                original_request_count=0,
                filtered_request_count=0,
                original_qps=data.metadata.measured_qps,
                adjusted_qps=data.metadata.measured_qps,
            )
            return data, report

        sorted_reqs = sorted(data.requests, key=lambda r: r.timestamp)
        min_ts = sorted_reqs[0].timestamp
        max_ts = sorted_reqs[-1].timestamp

        if self._warmup_seconds is not None:
            warmup_duration = self._warmup_seconds
        else:
            warmup_duration = self._auto_detect_warmup(sorted_reqs, min_ts)

        if warmup_duration <= 0:
            # No warm-up detected
            report = WarmupReport(
                warmup_detected=False,
                original_request_count=len(data.requests),
                filtered_request_count=len(data.requests),
                original_qps=data.metadata.measured_qps,
                adjusted_qps=data.metadata.measured_qps,
            )
            return data, report

        cutoff_ts = min_ts + warmup_duration
        warmup_reqs = [r for r in sorted_reqs if r.timestamp < cutoff_ts]
        kept_reqs = [r for r in sorted_reqs if r.timestamp >= cutoff_ts]

        if not kept_reqs:
            # All requests are in warm-up — return original data with report
            report = WarmupReport(
                warmup_detected=True,
                warmup_window=WarmupWindow(
                    start_timestamp=min_ts,
                    end_timestamp=cutoff_ts,
                    duration_s=round(warmup_duration, 3),
                    requests_excluded=len(warmup_reqs),
                ),
                original_request_count=len(data.requests),
                filtered_request_count=0,
                original_qps=data.metadata.measured_qps,
                adjusted_qps=0.0,
            )
            # Cannot create empty BenchmarkData (min 1 request), return original
            return data, report

        # Compute adjusted QPS
        total_duration = max_ts - min_ts
        kept_duration = max_ts - cutoff_ts
        if total_duration > 0 and kept_duration > 0:
            adjusted_qps = data.metadata.measured_qps * (kept_duration / total_duration)
        else:
            adjusted_qps = data.metadata.measured_qps

        # Latency comparison
        before_totals = [r.total_latency_ms for r in sorted_reqs]
        after_totals = [r.total_latency_ms for r in kept_reqs]
        comparison = LatencyComparison(
            before_p50_ms=round(_percentile(before_totals, 50), 3),
            before_p95_ms=round(_percentile(before_totals, 95), 3),
            before_p99_ms=round(_percentile(before_totals, 99), 3),
            after_p50_ms=round(_percentile(after_totals, 50), 3),
            after_p95_ms=round(_percentile(after_totals, 95), 3),
            after_p99_ms=round(_percentile(after_totals, 99), 3),
        )

        report = WarmupReport(
            warmup_detected=True,
            warmup_window=WarmupWindow(
                start_timestamp=min_ts,
                end_timestamp=cutoff_ts,
                duration_s=round(warmup_duration, 3),
                requests_excluded=len(warmup_reqs),
            ),
            original_request_count=len(data.requests),
            filtered_request_count=len(kept_reqs),
            original_qps=data.metadata.measured_qps,
            adjusted_qps=round(adjusted_qps, 3),
            latency_comparison=comparison,
        )

        filtered_data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=data.metadata.num_prefill_instances,
                num_decode_instances=data.metadata.num_decode_instances,
                total_instances=data.metadata.total_instances,
                measured_qps=round(adjusted_qps, 3),
            ),
            requests=kept_reqs,
        )

        return filtered_data, report

    def _auto_detect_warmup(
        self, sorted_reqs: list, min_ts: float
    ) -> float:
        """Auto-detect warm-up duration using windowed P95 analysis.

        Similar to timeline module logic: compute P95 per window,
        identify contiguous initial windows exceeding threshold.
        """
        if len(sorted_reqs) < 2:
            return 0.0

        max_ts = sorted_reqs[-1].timestamp
        total_duration = max_ts - min_ts
        if total_duration <= 0:
            return 0.0

        # Build windows
        n_windows = max(2, int(total_duration / self._window_size_s) + 1)
        windows: list[list[float]] = [[] for _ in range(n_windows)]

        for r in sorted_reqs:
            idx = min(int((r.timestamp - min_ts) / self._window_size_s), n_windows - 1)
            windows[idx].append(r.total_latency_ms)

        # Get P95 per window (skip empty)
        window_p95s: list[tuple[int, float]] = []
        for i, w in enumerate(windows):
            if w:
                window_p95s.append((i, _percentile(w, 95)))

        if len(window_p95s) < 2:
            return 0.0

        # Steady-state = median P95 from second half
        half = len(window_p95s) // 2
        second_half_vals = sorted([p for _, p in window_p95s[half:]])
        steady_p95 = _percentile(second_half_vals, 50)

        if steady_p95 == 0:
            return 0.0

        threshold = steady_p95 * self._warmup_factor
        warmup_window_count = 0

        for idx, p95 in window_p95s:
            if p95 > threshold:
                warmup_window_count += 1
            else:
                break

        if warmup_window_count == 0:
            return 0.0

        return warmup_window_count * self._window_size_s


def filter_warmup(
    benchmark_path: str,
    *,
    warmup_seconds: Optional[float] = None,
    warmup_factor: float = 2.0,
    window_size_s: float = 10.0,
) -> dict:
    """Programmatic API for warm-up filtering.

    Args:
        benchmark_path: Path to benchmark JSON file.
        warmup_seconds: Fixed warm-up duration. None for auto-detect.
        warmup_factor: P95 multiplier for auto-detection.
        window_size_s: Window size for auto-detection.

    Returns:
        Dict with 'report' and 'filtered_data' keys.
    """
    data = load_benchmark_auto(benchmark_path)
    filt = WarmupFilter(
        warmup_seconds=warmup_seconds,
        warmup_factor=warmup_factor,
        window_size_s=window_size_s,
    )
    filtered_data, report = filt.filter(data)
    return {
        "report": report.model_dump(),
        "filtered_data": filtered_data.model_dump(),
    }

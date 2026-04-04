"""Request queuing time analysis — estimate queuing delay and concurrency overlap."""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class CongestionLevel(str, Enum):
    """Congestion classification based on queuing ratio."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ConcurrencyPoint(BaseModel):
    """A point in the concurrency profile over time."""

    timestamp: float = Field(..., description="Epoch seconds")
    concurrent_requests: int = Field(..., ge=0, description="In-flight requests at this time")


class ConcurrencyProfile(BaseModel):
    """Concurrency statistics over the benchmark duration."""

    peak_concurrency: int = Field(..., ge=0, description="Maximum concurrent requests")
    mean_concurrency: float = Field(..., ge=0, description="Mean concurrent requests")
    p50_concurrency: float = Field(..., ge=0, description="Median concurrent requests")
    p95_concurrency: float = Field(..., ge=0, description="P95 concurrent requests")
    p99_concurrency: float = Field(..., ge=0, description="P99 concurrent requests")
    points: list[ConcurrencyPoint] = Field(
        default_factory=list,
        description="Sampled concurrency profile points (may be downsampled)",
    )


class QueueStats(BaseModel):
    """Queuing delay statistics across all requests."""

    min_queue_ms: float = Field(..., ge=0, description="Minimum estimated queue delay (ms)")
    mean_queue_ms: float = Field(..., ge=0, description="Mean estimated queue delay (ms)")
    p50_queue_ms: float = Field(..., ge=0, description="Median estimated queue delay (ms)")
    p95_queue_ms: float = Field(..., ge=0, description="P95 estimated queue delay (ms)")
    p99_queue_ms: float = Field(..., ge=0, description="P99 estimated queue delay (ms)")
    max_queue_ms: float = Field(..., ge=0, description="Maximum estimated queue delay (ms)")
    queue_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        description="Mean fraction of total latency spent in queue (0-1)",
    )


class QueueReport(BaseModel):
    """Complete request queuing time analysis report."""

    total_requests: int = Field(..., ge=1, description="Total analyzed requests")
    duration_seconds: float = Field(..., ge=0, description="Benchmark duration in seconds")
    queue_stats: QueueStats = Field(..., description="Queuing delay statistics")
    concurrency: ConcurrencyProfile = Field(..., description="Concurrency profile")
    congestion_level: CongestionLevel = Field(..., description="Overall congestion classification")
    recommendation: str = Field(..., description="Actionable recommendation")


class QueueAnalyzer:
    """Analyze request queuing times and concurrency profiles from benchmark data.

    Queuing delay is estimated as the gap between when a request arrives and
    when it could begin processing. We approximate this by computing the
    concurrency at each request's arrival time: if concurrency exceeds the
    total instances, the request is likely queued.

    The estimated queue delay per request is:
        queue_ms = max(0, total_latency_ms - estimated_service_ms)
    where estimated_service_ms is the median total_latency of requests that
    arrived with concurrency <= total_instances (i.e., unqueued baseline).
    """

    def __init__(self, max_profile_points: int = 200) -> None:
        self._max_profile_points = max_profile_points

    def analyze(self, data: BenchmarkData) -> QueueReport:
        """Run queuing analysis on benchmark data."""
        requests = sorted(data.requests, key=lambda r: r.timestamp)
        total_instances = data.metadata.total_instances

        # Build events for concurrency calculation
        events: list[tuple[float, int]] = []
        for r in requests:
            start = r.timestamp
            end = start + r.total_latency_ms / 1000.0
            events.append((start, 1))
            events.append((end, -1))
        events.sort(key=lambda e: (e[0], e[1]))

        # Compute concurrency at each event point
        timestamps: list[float] = []
        concurrency_values: list[int] = []
        current = 0
        for ts, delta in events:
            current += delta
            timestamps.append(ts)
            concurrency_values.append(current)

        conc_arr = np.array(concurrency_values, dtype=np.float64)

        # Concurrency profile stats
        if len(conc_arr) == 0:
            conc_arr = np.array([0.0])

        peak = int(np.max(conc_arr))
        mean_conc = float(np.mean(conc_arr))
        p50_conc = float(np.percentile(conc_arr, 50))
        p95_conc = float(np.percentile(conc_arr, 95))
        p99_conc = float(np.percentile(conc_arr, 99))

        # Build sampled profile points
        step = max(1, len(timestamps) // self._max_profile_points)
        points = [
            ConcurrencyPoint(timestamp=timestamps[i], concurrent_requests=concurrency_values[i])
            for i in range(0, len(timestamps), step)
        ]

        concurrency_profile = ConcurrencyProfile(
            peak_concurrency=peak,
            mean_concurrency=round(mean_conc, 2),
            p50_concurrency=round(p50_conc, 2),
            p95_concurrency=round(p95_conc, 2),
            p99_concurrency=round(p99_conc, 2),
            points=points,
        )

        # Compute concurrency at each request's start time
        # For efficiency, sweep through sorted starts against sorted events
        sorted_starts = [(r.timestamp, i) for i, r in enumerate(requests)]
        sorted_starts.sort()

        # Reset and sweep
        all_events_sorted = sorted(events, key=lambda e: (e[0], e[1]))
        running = 0
        conc_at_start: dict[int, int] = {}
        ei = 0
        for ts, req_idx in sorted_starts:
            while ei < len(all_events_sorted) and all_events_sorted[ei][0] <= ts:
                running += all_events_sorted[ei][1]
                ei += 1
            conc_at_start[req_idx] = running

        # Estimate baseline service time from low-concurrency requests
        low_conc_latencies = []
        for i, r in enumerate(requests):
            if conc_at_start.get(i, 0) <= total_instances:
                low_conc_latencies.append(r.total_latency_ms)

        if low_conc_latencies:
            baseline_ms = float(np.median(low_conc_latencies))
        else:
            # All requests are queued; use minimum latency as baseline
            baseline_ms = min(r.total_latency_ms for r in requests)

        # Estimate queue delay per request
        queue_delays: list[float] = []
        for r in requests:
            delay = max(0.0, r.total_latency_ms - baseline_ms)
            queue_delays.append(delay)

        q_arr = np.array(queue_delays, dtype=np.float64)
        total_latencies = np.array([r.total_latency_ms for r in requests], dtype=np.float64)
        mean_total = float(np.mean(total_latencies))
        mean_queue = float(np.mean(q_arr))
        queue_ratio = mean_queue / mean_total if mean_total > 0 else 0.0

        queue_stats = QueueStats(
            min_queue_ms=round(float(np.min(q_arr)), 3),
            mean_queue_ms=round(mean_queue, 3),
            p50_queue_ms=round(float(np.percentile(q_arr, 50)), 3),
            p95_queue_ms=round(float(np.percentile(q_arr, 95)), 3),
            p99_queue_ms=round(float(np.percentile(q_arr, 99)), 3),
            max_queue_ms=round(float(np.max(q_arr)), 3),
            queue_ratio=round(min(queue_ratio, 1.0), 4),
        )

        # Duration
        if len(requests) > 1:
            duration = requests[-1].timestamp - requests[0].timestamp
        else:
            duration = 0.0

        # Congestion classification
        if queue_ratio >= 0.5:
            congestion = CongestionLevel.CRITICAL
        elif queue_ratio >= 0.3:
            congestion = CongestionLevel.HIGH
        elif queue_ratio >= 0.1:
            congestion = CongestionLevel.MODERATE
        else:
            congestion = CongestionLevel.LOW

        # Recommendation
        if congestion == CongestionLevel.CRITICAL:
            recommendation = (
                f"Critical queuing detected — {queue_ratio:.0%} of latency is queue wait. "
                f"Peak concurrency ({peak}) far exceeds capacity ({total_instances}). "
                "Scale up instances urgently."
            )
        elif congestion == CongestionLevel.HIGH:
            recommendation = (
                f"High queuing — {queue_ratio:.0%} of latency is queue wait. "
                f"Peak concurrency {peak} vs {total_instances} instances. "
                "Consider adding instances to reduce tail latency."
            )
        elif congestion == CongestionLevel.MODERATE:
            recommendation = (
                f"Moderate queuing — {queue_ratio:.0%} of latency is queue wait. "
                "Some requests are delayed during traffic bursts. "
                "Monitor during peak hours."
            )
        else:
            recommendation = (
                "Minimal queuing detected. "
                "Instances are handling the load without significant queue buildup."
            )

        return QueueReport(
            total_requests=len(requests),
            duration_seconds=round(duration, 3),
            queue_stats=queue_stats,
            concurrency=concurrency_profile,
            congestion_level=congestion,
            recommendation=recommendation,
        )


def analyze_queue(benchmark_path: str) -> dict:
    """Programmatic API: analyze queuing from a benchmark file.

    Args:
        benchmark_path: Path to benchmark JSON file.

    Returns:
        Dict with full queue analysis results.
    """
    from .bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    analyzer = QueueAnalyzer()
    report = analyzer.analyze(data)
    return report.model_dump()

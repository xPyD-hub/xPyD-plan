"""Latency spike detection — identify sudden latency bursts in benchmark data."""

from __future__ import annotations

import statistics
from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class SpikeSeverity(str, Enum):
    """Severity of a detected spike."""

    MINOR = "minor"        # magnitude < 5x
    MODERATE = "moderate"  # magnitude 5-10x
    SEVERE = "severe"      # magnitude > 10x


class SpikeEvent(BaseModel):
    """A contiguous group of spike requests."""

    start_index: int = Field(..., description="Index of first spike request")
    end_index: int = Field(..., description="Index of last spike request (inclusive)")
    request_count: int = Field(..., ge=1, description="Number of requests in spike")
    metric: str = Field(..., description="Latency metric that spiked")
    peak_value: float = Field(..., description="Maximum latency value in spike")
    baseline_value: float = Field(..., description="Rolling baseline at spike start")
    magnitude: float = Field(..., description="peak / baseline ratio")
    severity: SpikeSeverity = Field(..., description="Spike severity classification")
    start_timestamp: float | None = Field(None, description="Timestamp of first spike request")
    end_timestamp: float | None = Field(None, description="Timestamp of last spike request")


class SpikeSummary(BaseModel):
    """Summary statistics across all detected spikes for one metric."""

    metric: str = Field(..., description="Metric name")
    spike_count: int = Field(..., ge=0, description="Number of spike events")
    total_affected_requests: int = Field(..., ge=0, description="Total requests in spikes")
    affected_fraction: float = Field(
        ..., ge=0, le=1, description="Fraction of all requests affected"
    )
    worst_magnitude: float = Field(..., ge=0, description="Highest spike magnitude")
    worst_severity: SpikeSeverity = Field(..., description="Severity of worst spike")


class SpikeReport(BaseModel):
    """Complete spike detection report."""

    total_requests: int = Field(..., description="Total requests analyzed")
    window_size: int = Field(..., description="Rolling window size used")
    threshold: float = Field(..., description="Spike threshold multiplier used")
    events: list[SpikeEvent] = Field(..., description="All detected spike events")
    summaries: list[SpikeSummary] = Field(..., description="Per-metric summary")
    has_spikes: bool = Field(..., description="Whether any spikes were detected")
    recommendation: str = Field(..., description="Human-readable recommendation")


def _classify_severity(magnitude: float) -> SpikeSeverity:
    """Classify spike severity based on magnitude."""
    if magnitude >= 10.0:
        return SpikeSeverity.SEVERE
    if magnitude >= 5.0:
        return SpikeSeverity.MODERATE
    return SpikeSeverity.MINOR


def _rolling_median(values: list[float], idx: int, window_size: int) -> float:
    """Compute median of the window preceding idx."""
    start = max(0, idx - window_size)
    window = values[start:idx]
    if not window:
        return values[idx]
    return statistics.median(window)


class SpikeDetector:
    """Detect latency spikes in benchmark data."""

    METRICS = ["ttft_ms", "tpot_ms", "total_latency_ms"]

    def detect(
        self,
        data: BenchmarkData,
        window_size: int = 50,
        threshold: float = 3.0,
    ) -> SpikeReport:
        """Detect spikes across all latency metrics.

        Args:
            data: Benchmark data to analyze.
            window_size: Rolling window size for baseline computation.
            threshold: Multiplier above baseline median to classify as spike.

        Returns:
            SpikeReport with all detected spike events.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if threshold <= 1.0:
            raise ValueError("threshold must be > 1.0")

        requests = sorted(data.requests, key=lambda r: r.timestamp)
        n = len(requests)
        all_events: list[SpikeEvent] = []

        for metric in self.METRICS:
            values = [getattr(r, metric) for r in requests]
            timestamps = [r.timestamp for r in requests]

            # Identify spike indices
            spike_flags = []
            for i in range(n):
                baseline = _rolling_median(values, i, window_size)
                if baseline > 0 and values[i] >= baseline * threshold:
                    spike_flags.append((i, baseline))
                else:
                    spike_flags.append(None)

            # Group consecutive spikes into events
            events = self._group_spikes(spike_flags, values, timestamps, metric)
            all_events.extend(events)

        # Build summaries
        summaries = self._build_summaries(all_events, n)

        has_spikes = len(all_events) > 0
        if has_spikes:
            worst = max(all_events, key=lambda e: e.magnitude)
            recommendation = (
                f"Detected {len(all_events)} spike event(s). "
                f"Worst: {worst.metric} at {worst.magnitude:.1f}× baseline "
                f"({worst.severity.value}). "
                f"Investigate transient system issues during spike windows."
            )
        else:
            recommendation = "No latency spikes detected. Latency appears stable."

        return SpikeReport(
            total_requests=n,
            window_size=window_size,
            threshold=threshold,
            events=all_events,
            summaries=summaries,
            has_spikes=has_spikes,
            recommendation=recommendation,
        )

    def _group_spikes(
        self,
        spike_flags: list[tuple[int, float] | None],
        values: list[float],
        timestamps: list[float],
        metric: str,
    ) -> list[SpikeEvent]:
        """Group consecutive spike indices into SpikeEvent objects."""
        events: list[SpikeEvent] = []
        i = 0
        n = len(spike_flags)
        while i < n:
            if spike_flags[i] is not None:
                start = i
                baseline = spike_flags[i][1]
                peak = values[i]
                while i < n and spike_flags[i] is not None:
                    peak = max(peak, values[i])
                    i += 1
                end = i - 1
                magnitude = peak / baseline if baseline > 0 else float("inf")
                events.append(
                    SpikeEvent(
                        start_index=start,
                        end_index=end,
                        request_count=end - start + 1,
                        metric=metric,
                        peak_value=peak,
                        baseline_value=baseline,
                        magnitude=round(magnitude, 2),
                        severity=_classify_severity(magnitude),
                        start_timestamp=timestamps[start],
                        end_timestamp=timestamps[end],
                    )
                )
            else:
                i += 1
        return events

    def _build_summaries(
        self, events: list[SpikeEvent], total_requests: int
    ) -> list[SpikeSummary]:
        """Build per-metric summaries."""
        summaries = []
        for metric in self.METRICS:
            metric_events = [e for e in events if e.metric == metric]
            count = len(metric_events)
            affected = sum(e.request_count for e in metric_events)
            if count > 0:
                worst = max(metric_events, key=lambda e: e.magnitude)
                summaries.append(
                    SpikeSummary(
                        metric=metric,
                        spike_count=count,
                        total_affected_requests=affected,
                        affected_fraction=round(affected / total_requests, 4),
                        worst_magnitude=worst.magnitude,
                        worst_severity=worst.severity,
                    )
                )
            else:
                summaries.append(
                    SpikeSummary(
                        metric=metric,
                        spike_count=0,
                        total_affected_requests=0,
                        affected_fraction=0.0,
                        worst_magnitude=0.0,
                        worst_severity=SpikeSeverity.MINOR,
                    )
                )
        return summaries


def detect_spikes(
    data: BenchmarkData,
    window_size: int = 50,
    threshold: float = 3.0,
) -> dict:
    """Programmatic API for spike detection.

    Args:
        data: Benchmark data to analyze.
        window_size: Rolling window size for baseline.
        threshold: Spike threshold multiplier.

    Returns:
        Dictionary with spike detection results.
    """
    detector = SpikeDetector()
    report = detector.detect(data, window_size=window_size, threshold=threshold)
    return report.model_dump()

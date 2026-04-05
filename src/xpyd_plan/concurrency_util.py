"""Concurrency utilization analysis — time-windowed instance utilization estimation."""

from __future__ import annotations

import enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class UtilizationLevel(str, enum.Enum):
    """Utilization classification for a time window."""

    IDLE = "idle"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class UtilizationWindow(BaseModel):
    """Utilization metrics for a single time window."""

    window_start: float = Field(..., description="Window start time (epoch seconds)")
    window_end: float = Field(..., description="Window end time (epoch seconds)")
    concurrent_requests: int = Field(..., ge=0, description="Peak concurrent requests in window")
    avg_concurrent: float = Field(..., ge=0, description="Average concurrent requests in window")
    utilization_pct: float = Field(
        ..., ge=0, le=100, description="Utilization percentage (avg_concurrent / total_instances)"
    )
    level: UtilizationLevel = Field(..., description="Utilization classification")
    requests_started: int = Field(..., ge=0, description="Requests that started in this window")
    requests_completed: int = Field(
        ..., ge=0, description="Requests that completed in this window"
    )


class RightSizingRecommendation(BaseModel):
    """Instance right-sizing recommendation."""

    current_instances: int = Field(..., description="Current total instances")
    peak_concurrent: int = Field(..., description="Peak concurrent requests observed")
    avg_concurrent: float = Field(..., description="Average concurrent requests")
    p95_concurrent: int = Field(..., description="P95 concurrent requests")
    recommended_min: int = Field(
        ..., description="Minimum instances to handle average load"
    )
    recommended_target: int = Field(
        ..., description="Target instances for P95 load with headroom"
    )
    over_provisioned: bool = Field(
        ..., description="True if current instances significantly exceed need"
    )
    under_provisioned: bool = Field(
        ..., description="True if current instances insufficient for P95 load"
    )


class UtilizationReport(BaseModel):
    """Complete concurrency utilization analysis report."""

    total_requests: int = Field(..., description="Total requests analyzed")
    total_instances: int = Field(..., description="Total instances in cluster")
    duration_seconds: float = Field(..., description="Benchmark duration in seconds")
    window_size_seconds: float = Field(..., description="Analysis window size")
    windows: list[UtilizationWindow] = Field(
        default_factory=list, description="Per-window utilization data"
    )
    idle_windows: int = Field(..., ge=0, description="Number of idle windows")
    high_windows: int = Field(..., ge=0, description="Number of high-utilization windows")
    avg_utilization_pct: float = Field(
        ..., ge=0, le=100, description="Average utilization across all windows"
    )
    peak_concurrent: int = Field(..., ge=0, description="Peak concurrent requests overall")
    recommendation: Optional[RightSizingRecommendation] = Field(
        None, description="Right-sizing recommendation"
    )


def _classify_utilization(pct: float) -> UtilizationLevel:
    """Classify utilization percentage into a level."""
    if pct < 20:
        return UtilizationLevel.IDLE
    elif pct < 50:
        return UtilizationLevel.LOW
    elif pct < 80:
        return UtilizationLevel.MODERATE
    else:
        return UtilizationLevel.HIGH


class ConcurrencyUtilizationAnalyzer:
    """Analyze time-windowed concurrency and instance utilization."""

    def __init__(self, window_size: float = 1.0) -> None:
        """Initialize analyzer.

        Args:
            window_size: Window size in seconds (default 1.0).
        """
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.window_size = window_size

    def analyze(self, data: BenchmarkData) -> UtilizationReport:
        """Run concurrency utilization analysis on benchmark data."""
        requests = data.requests
        if not requests:
            raise ValueError("No requests in benchmark data")

        total_instances = data.metadata.total_instances

        # Compute start and end times for each request
        starts = np.array([r.timestamp for r in requests])
        durations_s = np.array([r.total_latency_ms / 1000.0 for r in requests])
        ends = starts + durations_s

        global_start = float(np.min(starts))
        global_end = float(np.max(ends))
        duration = global_end - global_start

        if duration <= 0:
            duration = self.window_size

        # Create time windows
        n_windows = max(1, int(np.ceil(duration / self.window_size)))
        windows: list[UtilizationWindow] = []

        all_concurrent_peaks: list[int] = []
        all_concurrent_avgs: list[float] = []

        for i in range(n_windows):
            w_start = global_start + i * self.window_size
            w_end = w_start + self.window_size

            # Requests active during this window
            started_mask = (starts >= w_start) & (starts < w_end)
            completed_mask = (ends >= w_start) & (ends < w_end)

            started_count = int(np.sum(started_mask))
            completed_count = int(np.sum(completed_mask))

            # Estimate average concurrency by sampling within the window
            n_samples = 10
            sample_times = np.linspace(w_start, w_end, n_samples, endpoint=False)
            concurrency_samples = []
            for t in sample_times:
                concurrent = int(np.sum((starts <= t) & (ends > t)))
                concurrency_samples.append(concurrent)

            peak_concurrent = max(concurrency_samples) if concurrency_samples else 0
            avg_concurrent = float(np.mean(concurrency_samples)) if concurrency_samples else 0.0

            util_pct = (
                min(100.0, (avg_concurrent / total_instances) * 100.0)
                if total_instances > 0
                else 0.0
            )
            level = _classify_utilization(util_pct)

            windows.append(
                UtilizationWindow(
                    window_start=w_start,
                    window_end=w_end,
                    concurrent_requests=peak_concurrent,
                    avg_concurrent=round(avg_concurrent, 2),
                    utilization_pct=round(util_pct, 1),
                    level=level,
                    requests_started=started_count,
                    requests_completed=completed_count,
                )
            )
            all_concurrent_peaks.append(peak_concurrent)
            all_concurrent_avgs.append(avg_concurrent)

        idle_windows = sum(1 for w in windows if w.level == UtilizationLevel.IDLE)
        high_windows = sum(1 for w in windows if w.level == UtilizationLevel.HIGH)
        avg_util = float(np.mean([w.utilization_pct for w in windows])) if windows else 0.0
        overall_peak = max(all_concurrent_peaks) if all_concurrent_peaks else 0

        # Compute P95 concurrency from all sampled values
        all_samples = []
        n_samples = 10
        for i in range(n_windows):
            w_start = global_start + i * self.window_size
            w_end = w_start + self.window_size
            sample_times = np.linspace(w_start, w_end, n_samples, endpoint=False)
            for t in sample_times:
                concurrent = int(np.sum((starts <= t) & (ends > t)))
                all_samples.append(concurrent)

        p95_concurrent = int(np.percentile(all_samples, 95)) if all_samples else 0
        avg_concurrent_overall = float(np.mean(all_samples)) if all_samples else 0.0

        # Right-sizing recommendation
        headroom = 1.2  # 20% headroom
        recommended_min = max(2, int(np.ceil(avg_concurrent_overall)))
        recommended_target = max(2, int(np.ceil(p95_concurrent * headroom)))
        over_provisioned = total_instances > recommended_target * 1.5
        under_provisioned = total_instances < p95_concurrent

        recommendation = RightSizingRecommendation(
            current_instances=total_instances,
            peak_concurrent=overall_peak,
            avg_concurrent=round(avg_concurrent_overall, 2),
            p95_concurrent=p95_concurrent,
            recommended_min=recommended_min,
            recommended_target=recommended_target,
            over_provisioned=over_provisioned,
            under_provisioned=under_provisioned,
        )

        return UtilizationReport(
            total_requests=len(requests),
            total_instances=total_instances,
            duration_seconds=round(duration, 3),
            window_size_seconds=self.window_size,
            windows=windows,
            idle_windows=idle_windows,
            high_windows=high_windows,
            avg_utilization_pct=round(avg_util, 1),
            peak_concurrent=overall_peak,
            recommendation=recommendation,
        )


def analyze_concurrency_util(
    data: BenchmarkData,
    window_size: float = 1.0,
) -> UtilizationReport:
    """Programmatic API: analyze concurrency utilization.

    Args:
        data: Benchmark data to analyze.
        window_size: Window size in seconds (default 1.0).

    Returns:
        UtilizationReport with per-window utilization and right-sizing recommendation.
    """
    analyzer = ConcurrencyUtilizationAnalyzer(window_size=window_size)
    return analyzer.analyze(data)

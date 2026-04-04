"""Request timeline analysis — temporal latency patterns from benchmark data."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class LatencyTrendDirection(str, Enum):
    """Direction of latency trend over the benchmark duration."""

    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"


class TimeWindow(BaseModel):
    """Latency statistics for a single time window."""

    window_index: int = Field(..., ge=0, description="0-based window index")
    start_time: float = Field(..., description="Window start timestamp (epoch seconds)")
    end_time: float = Field(..., description="Window end timestamp (epoch seconds)")
    request_count: int = Field(..., ge=0, description="Requests in this window")
    ttft_p50_ms: float = Field(..., ge=0, description="TTFT P50 in ms")
    ttft_p95_ms: float = Field(..., ge=0, description="TTFT P95 in ms")
    tpot_p50_ms: float = Field(..., ge=0, description="TPOT P50 in ms")
    tpot_p95_ms: float = Field(..., ge=0, description="TPOT P95 in ms")
    total_p50_ms: float = Field(..., ge=0, description="Total latency P50 in ms")
    total_p95_ms: float = Field(..., ge=0, description="Total latency P95 in ms")


class WarmupAnalysis(BaseModel):
    """Warmup period detection result."""

    detected: bool = Field(..., description="Whether a warmup period was detected")
    warmup_windows: int = Field(0, ge=0, description="Number of warmup windows")
    warmup_duration_s: float = Field(0.0, ge=0, description="Warmup duration in seconds")
    steady_state_p95_ms: float = Field(0.0, ge=0, description="Steady-state P95 total latency")
    warmup_peak_p95_ms: float = Field(0.0, ge=0, description="Peak P95 during warmup")


class LatencyTrend(BaseModel):
    """Linear regression trend over the benchmark duration."""

    direction: LatencyTrendDirection = Field(..., description="Trend direction")
    slope_ms_per_s: float = Field(..., description="Slope: ms latency change per second")
    r_squared: float = Field(..., ge=0, le=1, description="R² goodness of fit")


class TimelineReport(BaseModel):
    """Complete timeline analysis report."""

    windows: list[TimeWindow] = Field(..., description="Per-window statistics")
    warmup: WarmupAnalysis = Field(..., description="Warmup detection result")
    trend: LatencyTrend = Field(..., description="Latency trend over time")
    recommendation: str = Field(..., description="Human-readable summary")


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of sorted values."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (p / 100.0) * (len(sorted_v) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Simple linear regression. Returns (slope, intercept, r_squared)."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_yy = sum((y - mean_y) ** 2 for y in ys)

    if ss_xx == 0:
        return 0.0, mean_y, 0.0

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    if ss_yy == 0:
        r_squared = 1.0 if ss_xy == 0 else 0.0
    else:
        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

    return slope, intercept, r_squared


class TimelineAnalyzer:
    """Analyze latency patterns over time from benchmark data."""

    def __init__(self, window_size_s: float = 10.0, warmup_factor: float = 2.0) -> None:
        """Initialize analyzer.

        Args:
            window_size_s: Time window size in seconds.
            warmup_factor: P95 threshold multiplier for warmup detection.
        """
        if window_size_s <= 0:
            raise ValueError("window_size_s must be positive")
        if warmup_factor <= 0:
            raise ValueError("warmup_factor must be positive")
        self.window_size_s = window_size_s
        self.warmup_factor = warmup_factor

    def analyze(self, data: BenchmarkData) -> TimelineReport:
        """Run timeline analysis on benchmark data.

        Args:
            data: Benchmark data with timestamped requests.

        Returns:
            TimelineReport with windows, warmup, and trend analysis.

        Raises:
            ValueError: If fewer than 2 requests.
        """
        if len(data.requests) < 2:
            raise ValueError("Need at least 2 requests for timeline analysis")

        # Sort by timestamp
        sorted_reqs = sorted(data.requests, key=lambda r: r.timestamp)
        t_min = sorted_reqs[0].timestamp
        t_max = sorted_reqs[-1].timestamp
        duration = t_max - t_min

        if duration == 0:
            # All same timestamp — single window
            num_windows = 1
        else:
            num_windows = max(1, math.ceil(duration / self.window_size_s))

        # Build windows
        windows: list[TimeWindow] = []
        for i in range(num_windows):
            w_start = t_min + i * self.window_size_s
            w_end = w_start + self.window_size_s

            reqs_in_window = [
                r for r in sorted_reqs
                if (r.timestamp >= w_start and r.timestamp < w_end)
                or (i == num_windows - 1 and r.timestamp == w_end)
            ]

            if not reqs_in_window:
                windows.append(TimeWindow(
                    window_index=i,
                    start_time=w_start,
                    end_time=w_end,
                    request_count=0,
                    ttft_p50_ms=0.0, ttft_p95_ms=0.0,
                    tpot_p50_ms=0.0, tpot_p95_ms=0.0,
                    total_p50_ms=0.0, total_p95_ms=0.0,
                ))
                continue

            ttft = [r.ttft_ms for r in reqs_in_window]
            tpot = [r.tpot_ms for r in reqs_in_window]
            total = [r.total_latency_ms for r in reqs_in_window]

            windows.append(TimeWindow(
                window_index=i,
                start_time=round(w_start, 3),
                end_time=round(w_end, 3),
                request_count=len(reqs_in_window),
                ttft_p50_ms=round(_percentile(ttft, 50), 3),
                ttft_p95_ms=round(_percentile(ttft, 95), 3),
                tpot_p50_ms=round(_percentile(tpot, 50), 3),
                tpot_p95_ms=round(_percentile(tpot, 95), 3),
                total_p50_ms=round(_percentile(total, 50), 3),
                total_p95_ms=round(_percentile(total, 95), 3),
            ))

        # Warmup detection
        warmup = self._detect_warmup(windows)

        # Trend detection (use total_p95 per window)
        trend = self._detect_trend(windows)

        # Recommendation
        recommendation = self._build_recommendation(warmup, trend)

        return TimelineReport(
            windows=windows,
            warmup=warmup,
            trend=trend,
            recommendation=recommendation,
        )

    def _detect_warmup(self, windows: list[TimeWindow]) -> WarmupAnalysis:
        """Detect warmup period where P95 latency greatly exceeds steady state."""
        populated = [w for w in windows if w.request_count > 0]
        if len(populated) < 2:
            return WarmupAnalysis(detected=False)

        # Steady state = median of P95 values from the second half
        half = len(populated) // 2
        second_half_p95 = sorted([w.total_p95_ms for w in populated[half:]])
        steady_p95 = _percentile(second_half_p95, 50)

        if steady_p95 == 0:
            return WarmupAnalysis(detected=False)

        threshold = steady_p95 * self.warmup_factor
        warmup_count = 0
        peak = 0.0

        for w in populated:
            if w.total_p95_ms > threshold:
                warmup_count += 1
                peak = max(peak, w.total_p95_ms)
            else:
                break  # warmup is contiguous from the start

        if warmup_count == 0:
            return WarmupAnalysis(detected=False)

        warmup_duration = warmup_count * self.window_size_s

        return WarmupAnalysis(
            detected=True,
            warmup_windows=warmup_count,
            warmup_duration_s=round(warmup_duration, 3),
            steady_state_p95_ms=round(steady_p95, 3),
            warmup_peak_p95_ms=round(peak, 3),
        )

    def _detect_trend(self, windows: list[TimeWindow]) -> LatencyTrend:
        """Detect latency trend via linear regression on per-window P95."""
        populated = [w for w in windows if w.request_count > 0]
        if len(populated) < 2:
            return LatencyTrend(
                direction=LatencyTrendDirection.STABLE,
                slope_ms_per_s=0.0,
                r_squared=0.0,
            )

        # Use window midpoint as x, total P95 as y
        xs = [(w.start_time + w.end_time) / 2 for w in populated]
        ys = [w.total_p95_ms for w in populated]

        slope, _intercept, r_sq = _linear_regression(xs, ys)

        # Classify: significant if R² >= 0.3 and slope magnitude is non-trivial
        if r_sq < 0.3 or abs(slope) < 0.001:
            direction = LatencyTrendDirection.STABLE
        elif slope > 0:
            direction = LatencyTrendDirection.DEGRADING
        else:
            direction = LatencyTrendDirection.IMPROVING

        return LatencyTrend(
            direction=direction,
            slope_ms_per_s=round(slope, 6),
            r_squared=round(r_sq, 6),
        )

    def _build_recommendation(
        self, warmup: WarmupAnalysis, trend: LatencyTrend
    ) -> str:
        """Build human-readable recommendation."""
        parts: list[str] = []

        if warmup.detected:
            parts.append(
                f"Warmup detected: first {warmup.warmup_windows} window(s) "
                f"({warmup.warmup_duration_s}s) show P95 latency up to "
                f"{warmup.warmup_peak_p95_ms:.1f}ms vs {warmup.steady_state_p95_ms:.1f}ms "
                f"steady state. Consider excluding warmup data from SLA analysis."
            )

        if trend.direction == LatencyTrendDirection.DEGRADING:
            parts.append(
                f"Latency is degrading over time (slope: {trend.slope_ms_per_s:+.3f} ms/s, "
                f"R²={trend.r_squared:.3f}). Investigate resource contention or queue buildup."
            )
        elif trend.direction == LatencyTrendDirection.IMPROVING:
            parts.append(
                f"Latency is improving over time (slope: {trend.slope_ms_per_s:+.3f} ms/s, "
                f"R²={trend.r_squared:.3f}). System may be warming caches or JIT-compiling."
            )

        if not parts:
            parts.append("No significant temporal patterns detected. Latency is stable over time.")

        return " ".join(parts)


def analyze_timeline(
    data: BenchmarkData,
    window_size_s: float = 10.0,
    warmup_factor: float = 2.0,
) -> dict:
    """Programmatic API for timeline analysis.

    Args:
        data: Benchmark data to analyze.
        window_size_s: Time window size in seconds.
        warmup_factor: Warmup detection threshold multiplier.

    Returns:
        Dictionary representation of the TimelineReport.
    """
    analyzer = TimelineAnalyzer(
        window_size_s=window_size_s,
        warmup_factor=warmup_factor,
    )
    report = analyzer.analyze(data)
    return report.model_dump()

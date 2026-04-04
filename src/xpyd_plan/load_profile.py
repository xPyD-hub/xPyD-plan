"""Load profile classification for benchmark data."""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class ProfileType(str, Enum):
    """Load profile classification."""

    STEADY_STATE = "steady_state"
    RAMP_UP = "ramp_up"
    RAMP_DOWN = "ramp_down"
    BURST = "burst"
    CYCLIC = "cyclic"
    UNKNOWN = "unknown"


class RateWindow(BaseModel):
    """Request rate in a time window."""

    start_time: float = Field(description="Window start (epoch seconds)")
    end_time: float = Field(description="Window end (epoch seconds)")
    request_count: int = Field(description="Requests in this window")
    rate_rps: float = Field(description="Requests per second in this window")


class LoadProfile(BaseModel):
    """Detected load profile with supporting evidence."""

    profile_type: ProfileType = Field(description="Classified load profile type")
    confidence: float = Field(description="Classification confidence (0.0–1.0)")
    rate_cv: float = Field(description="Coefficient of variation of per-window rates")
    rate_trend_slope: float = Field(description="Linear regression slope of rate over time")
    peak_to_trough_ratio: float = Field(
        description="Ratio of max rate to min rate (min clamped to avoid div-by-zero)",
    )
    description: str = Field(description="Human-readable description of the profile")


class LoadProfileReport(BaseModel):
    """Complete load profile analysis report."""

    profile: LoadProfile = Field(description="Detected load profile")
    windows: list[RateWindow] = Field(description="Per-window rate data")
    total_requests: int = Field(description="Total requests in benchmark")
    duration_seconds: float = Field(description="Total benchmark duration")
    overall_rate_rps: float = Field(description="Average requests per second")


class LoadProfileClassifier:
    """Classify the load pattern of a benchmark."""

    def __init__(self, data: BenchmarkData) -> None:
        self._data = data

    def classify(
        self,
        window_size: float = 5.0,
    ) -> LoadProfileReport:
        """Classify the load profile.

        Args:
            window_size: Time window size in seconds for rate computation.
        """
        requests = self._data.requests
        n = len(requests)

        if n == 0:
            return LoadProfileReport(
                profile=LoadProfile(
                    profile_type=ProfileType.UNKNOWN,
                    confidence=0.0,
                    rate_cv=0.0,
                    rate_trend_slope=0.0,
                    peak_to_trough_ratio=1.0,
                    description="No requests in benchmark data.",
                ),
                windows=[],
                total_requests=0,
                duration_seconds=0.0,
                overall_rate_rps=0.0,
            )

        timestamps = sorted(r.timestamp for r in requests)
        t_min = timestamps[0]
        t_max = timestamps[-1]
        duration = t_max - t_min

        if duration <= 0:
            return LoadProfileReport(
                profile=LoadProfile(
                    profile_type=ProfileType.STEADY_STATE,
                    confidence=0.5,
                    rate_cv=0.0,
                    rate_trend_slope=0.0,
                    peak_to_trough_ratio=1.0,
                    description="All requests have the same timestamp; treating as steady state.",
                ),
                windows=[
                    RateWindow(
                        start_time=t_min,
                        end_time=t_min + window_size,
                        request_count=n,
                        rate_rps=n / window_size,
                    )
                ],
                total_requests=n,
                duration_seconds=0.0,
                overall_rate_rps=float(n),
            )

        # Build time windows
        windows: list[RateWindow] = []
        t = t_min
        ts_array = np.array(timestamps)
        while t < t_max:
            t_end = min(t + window_size, t_max + 0.001)
            count = int(np.sum((ts_array >= t) & (ts_array < t_end)))
            actual_width = t_end - t
            rate = count / actual_width if actual_width > 0 else 0.0
            windows.append(
                RateWindow(
                    start_time=round(t, 3),
                    end_time=round(t_end, 3),
                    request_count=count,
                    rate_rps=round(rate, 4),
                )
            )
            t += window_size

        if len(windows) < 2:
            return LoadProfileReport(
                profile=LoadProfile(
                    profile_type=ProfileType.STEADY_STATE,
                    confidence=0.5,
                    rate_cv=0.0,
                    rate_trend_slope=0.0,
                    peak_to_trough_ratio=1.0,
                    description="Too few windows for pattern detection; treating as steady state.",
                ),
                windows=windows,
                total_requests=n,
                duration_seconds=round(duration, 3),
                overall_rate_rps=round(n / duration, 4),
            )

        rates = np.array([w.rate_rps for w in windows])
        mean_rate = float(np.mean(rates))
        std_rate = float(np.std(rates))
        cv = std_rate / mean_rate if mean_rate > 0 else 0.0

        # Linear regression for trend
        x = np.arange(len(rates), dtype=float)
        if len(x) > 1:
            slope = float(np.polyfit(x, rates, 1)[0])
        else:
            slope = 0.0

        # Normalize slope relative to mean rate
        norm_slope = slope / mean_rate if mean_rate > 0 else 0.0

        min_rate = float(np.min(rates))
        max_rate = float(np.max(rates))
        peak_trough = max_rate / max(min_rate, 0.001)

        # Classification logic
        profile_type, confidence, description = self._classify_pattern(
            cv, norm_slope, peak_trough, rates
        )

        return LoadProfileReport(
            profile=LoadProfile(
                profile_type=profile_type,
                confidence=round(confidence, 3),
                rate_cv=round(cv, 4),
                rate_trend_slope=round(slope, 4),
                peak_to_trough_ratio=round(peak_trough, 4),
                description=description,
            ),
            windows=windows,
            total_requests=n,
            duration_seconds=round(duration, 3),
            overall_rate_rps=round(n / duration, 4),
        )

    @staticmethod
    def _classify_pattern(
        cv: float,
        norm_slope: float,
        peak_trough: float,
        rates: np.ndarray,
    ) -> tuple[ProfileType, float, str]:
        """Classify the load pattern based on rate statistics."""
        # Steady state: low CV
        if cv < 0.2:
            return (
                ProfileType.STEADY_STATE,
                min(1.0, 1.0 - cv / 0.2),
                f"Request rate is stable (CV={cv:.3f}).",
            )

        # Ramp up: strong positive trend
        if norm_slope > 0.1 and cv >= 0.2:
            conf = min(1.0, abs(norm_slope) / 0.3)
            return (
                ProfileType.RAMP_UP,
                conf,
                f"Request rate increases over time (normalized slope={norm_slope:.3f}).",
            )

        # Ramp down: strong negative trend
        if norm_slope < -0.1 and cv >= 0.2:
            conf = min(1.0, abs(norm_slope) / 0.3)
            return (
                ProfileType.RAMP_DOWN,
                conf,
                f"Request rate decreases over time (normalized slope={norm_slope:.3f}).",
            )

        # Burst: very high peak-to-trough ratio
        if peak_trough > 5.0:
            conf = min(1.0, peak_trough / 10.0)
            return (
                ProfileType.BURST,
                conf,
                f"Burst pattern detected (peak/trough={peak_trough:.1f}x).",
            )

        # Cyclic: check for sign changes in rate derivative
        if len(rates) >= 4:
            diffs = np.diff(rates)
            sign_changes = int(np.sum(np.abs(np.diff(np.sign(diffs))) > 0))
            if sign_changes >= len(rates) // 2:
                conf = min(1.0, sign_changes / len(rates))
                return (
                    ProfileType.CYCLIC,
                    conf,
                    f"Cyclic pattern detected ({sign_changes} direction changes).",
                )

        # Fallback: burst if high CV
        if cv >= 0.5:
            return (
                ProfileType.BURST,
                0.5,
                f"High rate variance (CV={cv:.3f}), classified as burst.",
            )

        return (
            ProfileType.UNKNOWN,
            0.3,
            f"No clear pattern detected (CV={cv:.3f}).",
        )


def classify_load_profile(
    data: BenchmarkData,
    window_size: float = 5.0,
) -> dict:
    """Programmatic API for load profile classification.

    Returns:
        dict representation of LoadProfileReport.
    """
    classifier = LoadProfileClassifier(data)
    report = classifier.classify(window_size=window_size)
    return report.model_dump()

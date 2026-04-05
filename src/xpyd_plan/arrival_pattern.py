"""Request arrival pattern analysis.

Analyze inter-arrival times to characterize request distribution
patterns: Poisson (exponential), bursty, periodic, or uniform.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.benchmark_models import BenchmarkData


class ArrivalPattern(str, Enum):
    """Classified arrival pattern type."""

    POISSON = "poisson"
    BURSTY = "bursty"
    PERIODIC = "periodic"
    UNIFORM = "uniform"
    UNKNOWN = "unknown"


class InterArrivalStats(BaseModel):
    """Statistics on inter-arrival times."""

    count: int = Field(..., ge=0, description="Number of inter-arrival intervals")
    mean_ms: float = Field(..., ge=0, description="Mean inter-arrival time (ms)")
    std_ms: float = Field(..., ge=0, description="Std deviation of inter-arrival time (ms)")
    min_ms: float = Field(..., ge=0, description="Minimum inter-arrival time (ms)")
    p50_ms: float = Field(..., ge=0, description="Median inter-arrival time (ms)")
    p95_ms: float = Field(..., ge=0, description="P95 inter-arrival time (ms)")
    p99_ms: float = Field(..., ge=0, description="P99 inter-arrival time (ms)")
    max_ms: float = Field(..., ge=0, description="Maximum inter-arrival time (ms)")
    cv: float = Field(..., ge=0, description="Coefficient of variation (std/mean)")


class BurstInfo(BaseModel):
    """Information about detected bursts."""

    burst_count: int = Field(..., ge=0, description="Number of detected bursts")
    avg_burst_size: float = Field(..., ge=0, description="Average requests per burst")
    burst_fraction: float = Field(
        ..., ge=0, le=1, description="Fraction of requests in bursts"
    )


class ArrivalPatternReport(BaseModel):
    """Full arrival pattern analysis report."""

    pattern: ArrivalPattern = Field(..., description="Classified arrival pattern")
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in pattern classification"
    )
    inter_arrival: InterArrivalStats = Field(
        ..., description="Inter-arrival time statistics"
    )
    burst_info: Optional[BurstInfo] = Field(
        None, description="Burst details (if bursty pattern)"
    )
    poisson_fit_p_value: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="P-value from exponentiality test (high = good Poisson fit)",
    )
    request_count: int = Field(..., ge=0)
    duration_s: float = Field(..., ge=0)
    measured_qps: float = Field(..., ge=0)


class ArrivalPatternAnalyzer:
    """Analyze request arrival patterns from benchmark data."""

    def __init__(
        self,
        burst_threshold: float = 0.1,
    ) -> None:
        """Initialize analyzer.

        Args:
            burst_threshold: Inter-arrival times below this fraction of the mean
                are considered part of a burst. Default 0.1 (10% of mean).
        """
        if burst_threshold <= 0 or burst_threshold >= 1:
            raise ValueError("burst_threshold must be in (0, 1)")
        self._burst_threshold = burst_threshold

    def analyze(self, data: BenchmarkData) -> ArrivalPatternReport:
        """Analyze arrival patterns in benchmark data.

        Args:
            data: Benchmark data to analyze.

        Returns:
            ArrivalPatternReport with classification and statistics.
        """
        if len(data.requests) < 2:
            return ArrivalPatternReport(
                pattern=ArrivalPattern.UNKNOWN,
                confidence=0.0,
                inter_arrival=InterArrivalStats(
                    count=0,
                    mean_ms=0.0,
                    std_ms=0.0,
                    min_ms=0.0,
                    p50_ms=0.0,
                    p95_ms=0.0,
                    p99_ms=0.0,
                    max_ms=0.0,
                    cv=0.0,
                ),
                request_count=len(data.requests),
                duration_s=0.0,
                measured_qps=data.metadata.measured_qps,
            )

        sorted_reqs = sorted(data.requests, key=lambda r: r.timestamp)
        timestamps = np.array([r.timestamp for r in sorted_reqs])

        # Inter-arrival times in milliseconds
        iat_s = np.diff(timestamps)
        iat_ms = iat_s * 1000.0

        # Remove exact zeros (simultaneous arrivals) for statistics
        iat_nonzero = iat_ms[iat_ms > 0]
        if len(iat_nonzero) == 0:
            iat_nonzero = iat_ms

        mean_ms = float(np.mean(iat_nonzero))
        std_ms = float(np.std(iat_nonzero, ddof=1)) if len(iat_nonzero) > 1 else 0.0
        cv = std_ms / mean_ms if mean_ms > 0 else 0.0

        stats = InterArrivalStats(
            count=len(iat_ms),
            mean_ms=round(mean_ms, 3),
            std_ms=round(std_ms, 3),
            min_ms=round(float(np.min(iat_ms)), 3),
            p50_ms=round(float(np.percentile(iat_ms, 50)), 3),
            p95_ms=round(float(np.percentile(iat_ms, 95)), 3),
            p99_ms=round(float(np.percentile(iat_ms, 99)), 3),
            max_ms=round(float(np.max(iat_ms)), 3),
            cv=round(cv, 4),
        )

        duration_s = float(timestamps[-1] - timestamps[0])

        # Burst detection
        burst_info = self._detect_bursts(iat_ms, mean_ms)

        # Poisson test (exponentiality)
        poisson_p = self._test_exponentiality(iat_nonzero)

        # Classify
        pattern, confidence = self._classify(cv, poisson_p, burst_info, iat_ms)

        return ArrivalPatternReport(
            pattern=pattern,
            confidence=round(confidence, 3),
            inter_arrival=stats,
            burst_info=burst_info if burst_info.burst_count > 0 else None,
            poisson_fit_p_value=round(poisson_p, 4) if poisson_p is not None else None,
            request_count=len(data.requests),
            duration_s=round(duration_s, 3),
            measured_qps=data.metadata.measured_qps,
        )

    def _detect_bursts(
        self, iat_ms: np.ndarray, mean_ms: float
    ) -> BurstInfo:
        """Detect bursts based on inter-arrival time threshold."""
        if mean_ms <= 0 or len(iat_ms) == 0:
            return BurstInfo(burst_count=0, avg_burst_size=0.0, burst_fraction=0.0)

        threshold = mean_ms * self._burst_threshold
        in_burst = iat_ms < threshold

        burst_count = 0
        burst_total_reqs = 0
        current_burst_size = 0
        prev_in_burst = False

        for is_burst in in_burst:
            if is_burst:
                if not prev_in_burst:
                    burst_count += 1
                    current_burst_size = 2  # At least 2 requests form a burst
                else:
                    current_burst_size += 1
                prev_in_burst = True
            else:
                if prev_in_burst:
                    burst_total_reqs += current_burst_size
                    current_burst_size = 0
                prev_in_burst = False

        if prev_in_burst:
            burst_total_reqs += current_burst_size

        avg_burst = burst_total_reqs / burst_count if burst_count > 0 else 0.0
        total_reqs = len(iat_ms) + 1
        fraction = burst_total_reqs / total_reqs if total_reqs > 0 else 0.0

        return BurstInfo(
            burst_count=burst_count,
            avg_burst_size=round(avg_burst, 1),
            burst_fraction=round(min(fraction, 1.0), 4),
        )

    def _test_exponentiality(self, iat_ms: np.ndarray) -> Optional[float]:
        """Test if inter-arrival times follow exponential distribution.

        Uses the coefficient of variation test: for exponential distribution,
        CV should be approximately 1.0.
        """
        if len(iat_ms) < 10:
            return None

        mean = float(np.mean(iat_ms))
        if mean <= 0:
            return None

        std = float(np.std(iat_ms, ddof=1))
        cv = std / mean

        # For exponential, CV ≈ 1. Use a simple heuristic p-value based on
        # distance from CV=1. This is a simplified approach.
        distance = abs(cv - 1.0)
        # Map distance to a pseudo p-value: distance 0 → p=1, distance 1 → p≈0.05
        p_value = max(0.0, min(1.0, np.exp(-3.0 * distance)))
        return float(p_value)

    def _classify(
        self,
        cv: float,
        poisson_p: Optional[float],
        burst_info: BurstInfo,
        iat_ms: np.ndarray,
    ) -> tuple[ArrivalPattern, float]:
        """Classify the arrival pattern."""
        # Check for uniform (very low CV)
        if cv < 0.3:
            confidence = max(0.5, 1.0 - cv / 0.3)
            return ArrivalPattern.UNIFORM, confidence

        # Check for bursty (high burst fraction)
        if burst_info.burst_count > 0 and burst_info.burst_fraction > 0.3:
            confidence = min(1.0, burst_info.burst_fraction)
            return ArrivalPattern.BURSTY, confidence

        # Check for periodic (detect regularity)
        if self._check_periodicity(iat_ms):
            return ArrivalPattern.PERIODIC, 0.7

        # Check Poisson fit
        if poisson_p is not None and poisson_p > 0.3:
            confidence = min(1.0, poisson_p)
            return ArrivalPattern.POISSON, confidence

        # High CV but not matching other patterns → bursty
        if cv > 1.5:
            return ArrivalPattern.BURSTY, 0.6

        # Moderate CV → likely Poisson
        if 0.7 <= cv <= 1.3:
            return ArrivalPattern.POISSON, 0.5

        return ArrivalPattern.UNKNOWN, 0.3

    def _check_periodicity(self, iat_ms: np.ndarray) -> bool:
        """Check if inter-arrival times show periodic pattern.

        Uses autocorrelation at lag 1 — high positive autocorrelation
        suggests periodicity.
        """
        if len(iat_ms) < 20:
            return False

        # Compute lag-1 autocorrelation
        mean = np.mean(iat_ms)
        diff = iat_ms - mean
        var = np.sum(diff**2)
        if var == 0:
            return True  # All identical = perfectly periodic

        autocorr = np.sum(diff[:-1] * diff[1:]) / var
        return float(autocorr) > 0.5


def analyze_arrival_pattern(
    benchmark_path: str,
    *,
    burst_threshold: float = 0.1,
) -> dict:
    """Programmatic API for arrival pattern analysis.

    Args:
        benchmark_path: Path to benchmark JSON file.
        burst_threshold: Threshold for burst detection (fraction of mean IAT).

    Returns:
        Dict with analysis results.
    """
    data = load_benchmark_auto(benchmark_path)
    analyzer = ArrivalPatternAnalyzer(burst_threshold=burst_threshold)
    report = analyzer.analyze(data)
    return report.model_dump()

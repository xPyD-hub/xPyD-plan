"""Request Rate Limiter Recommender — derive safe rate limits from benchmark data.

Given multiple benchmark files at different QPS levels and SLA constraints,
recommend a sustainable rate limit with safety margin and burst allowance.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class BurstConfig(BaseModel):
    """Burst allowance configuration derived from benchmark data."""

    sustained_qps: float = Field(..., ge=0, description="Recommended sustained rate limit")
    burst_qps: float = Field(..., ge=0, description="Maximum burst rate (short-term peak)")
    burst_window_seconds: float = Field(
        ..., gt=0, description="Time window for burst allowance"
    )
    burst_ratio: float = Field(
        ..., ge=1.0, description="Burst QPS / sustained QPS ratio"
    )


class HeadroomInfo(BaseModel):
    """Headroom analysis at the recommended rate limit."""

    metric: str = Field(..., description="Metric name")
    sla_threshold_ms: float = Field(..., gt=0)
    estimated_latency_ms: float = Field(
        ..., ge=0, description="Estimated latency at recommended QPS"
    )
    headroom_ms: float = Field(..., description="SLA threshold - estimated latency")
    headroom_pct: float = Field(..., description="Headroom as percentage of SLA threshold")


class RateLimitReport(BaseModel):
    """Complete rate limit recommendation report."""

    max_sla_compliant_qps: float | None = Field(
        None, description="Maximum QPS that satisfies all SLA constraints"
    )
    safety_margin: float = Field(..., ge=0, le=1, description="Applied safety margin fraction")
    recommended_qps: float = Field(
        ..., ge=0, description="Recommended sustained rate limit"
    )
    burst: BurstConfig | None = Field(
        None, description="Burst allowance details"
    )
    headroom: list[HeadroomInfo] = Field(
        default_factory=list, description="Per-metric headroom at recommended QPS"
    )
    data_points: int = Field(..., ge=0, description="Number of benchmark files analyzed")
    qps_range: tuple[float, float] | None = Field(
        None, description="(min, max) measured QPS across benchmarks"
    )
    recommendation: str = Field(..., description="Human-readable recommendation")


class RateLimitRecommender:
    """Recommend rate limits from multi-QPS benchmark data and SLA constraints.

    Analyzes benchmarks at different QPS levels, fits QPS→latency relationships,
    and finds the maximum safe QPS with a configurable safety margin.

    Parameters
    ----------
    safety_margin : float
        Fraction to subtract from max SLA-compliant QPS (default 0.10 = 10%).
    burst_window_seconds : float
        Time window (seconds) for short-term burst allowance computation (default 10).
    """

    def __init__(
        self,
        safety_margin: float = 0.10,
        burst_window_seconds: float = 10.0,
    ) -> None:
        if not 0 <= safety_margin < 1:
            msg = f"safety_margin must be in [0, 1), got {safety_margin}"
            raise ValueError(msg)
        if burst_window_seconds <= 0:
            msg = f"burst_window_seconds must be > 0, got {burst_window_seconds}"
            raise ValueError(msg)
        self.safety_margin = safety_margin
        self.burst_window_seconds = burst_window_seconds

    def recommend(
        self,
        benchmarks: list[BenchmarkData],
        sla_ttft_ms: float | None = None,
        sla_tpot_ms: float | None = None,
        sla_total_ms: float | None = None,
    ) -> RateLimitReport:
        """Analyze benchmarks and produce a rate limit recommendation.

        Parameters
        ----------
        benchmarks : list[BenchmarkData]
            Benchmark files at different QPS levels (at least 2).
        sla_ttft_ms : float | None
            SLA threshold for TTFT P95 (ms).
        sla_tpot_ms : float | None
            SLA threshold for TPOT P95 (ms).
        sla_total_ms : float | None
            SLA threshold for total latency P95 (ms).

        Returns
        -------
        RateLimitReport
        """
        if len(benchmarks) < 2:
            return RateLimitReport(
                max_sla_compliant_qps=None,
                safety_margin=self.safety_margin,
                recommended_qps=0.0,
                data_points=len(benchmarks),
                qps_range=None,
                recommendation="Need at least 2 benchmark files at different QPS levels.",
            )

        if sla_ttft_ms is None and sla_tpot_ms is None and sla_total_ms is None:
            return RateLimitReport(
                max_sla_compliant_qps=None,
                safety_margin=self.safety_margin,
                recommended_qps=0.0,
                data_points=len(benchmarks),
                qps_range=None,
                recommendation="No SLA thresholds specified. Cannot determine rate limit.",
            )

        # Extract QPS and P95 latencies per benchmark
        points: list[dict] = []
        for bm in benchmarks:
            qps = bm.metadata.measured_qps
            reqs = bm.requests
            if not reqs:
                continue
            ttfts = np.array([r.ttft_ms for r in reqs])
            tpots = np.array([r.tpot_ms for r in reqs])
            totals = np.array([r.total_latency_ms for r in reqs])
            points.append({
                "qps": qps,
                "ttft_p95": float(np.percentile(ttfts, 95)),
                "tpot_p95": float(np.percentile(tpots, 95)),
                "total_p95": float(np.percentile(totals, 95)),
                "requests": reqs,
            })

        points.sort(key=lambda p: p["qps"])
        qps_values = np.array([p["qps"] for p in points])
        qps_range = (float(qps_values[0]), float(qps_values[-1]))

        # For each SLA metric, find max compliant QPS via interpolation
        sla_checks: list[tuple[str, float, np.ndarray]] = []
        if sla_ttft_ms is not None:
            sla_checks.append(("ttft_p95", sla_ttft_ms, np.array([p["ttft_p95"] for p in points])))
        if sla_tpot_ms is not None:
            sla_checks.append(("tpot_p95", sla_tpot_ms, np.array([p["tpot_p95"] for p in points])))
        if sla_total_ms is not None:
            sla_checks.append((
                "total_p95",
                sla_total_ms,
                np.array([p["total_p95"] for p in points]),
            ))

        max_compliant_per_metric: list[tuple[str, float, float]] = []  # (metric, max_qps, sla)

        for metric_name, sla_threshold, latencies in sla_checks:
            max_qps = self._find_max_compliant_qps(qps_values, latencies, sla_threshold)
            if max_qps is not None:
                max_compliant_per_metric.append((metric_name, max_qps, sla_threshold))

        if not max_compliant_per_metric:
            # All metrics violate SLA even at lowest QPS
            return RateLimitReport(
                max_sla_compliant_qps=None,
                safety_margin=self.safety_margin,
                recommended_qps=0.0,
                data_points=len(points),
                qps_range=qps_range,
                recommendation=(
                    "SLA is violated at all tested QPS levels. "
                    "Cannot recommend a rate limit."
                ),
            )

        # Overall max compliant = minimum across metrics (most constrained)
        overall_max_qps = min(m[1] for m in max_compliant_per_metric)
        recommended = overall_max_qps * (1.0 - self.safety_margin)

        # Compute headroom at recommended QPS
        headroom_list: list[HeadroomInfo] = []
        for metric_name, _, sla_threshold in max_compliant_per_metric:
            latencies = np.array([p[metric_name] for p in points])
            estimated = float(np.interp(recommended, qps_values, latencies))
            headroom_ms = sla_threshold - estimated
            headroom_pct = (headroom_ms / sla_threshold) * 100 if sla_threshold > 0 else 0
            headroom_list.append(HeadroomInfo(
                metric=metric_name,
                sla_threshold_ms=sla_threshold,
                estimated_latency_ms=round(estimated, 2),
                headroom_ms=round(headroom_ms, 2),
                headroom_pct=round(headroom_pct, 1),
            ))

        # Compute burst allowance from short-window peak QPS analysis
        burst = self._compute_burst(points, recommended)

        bottleneck = min(max_compliant_per_metric, key=lambda x: x[1])
        rec_text = (
            f"Recommended sustained rate limit: {recommended:.1f} QPS "
            f"(max SLA-compliant: {overall_max_qps:.1f} QPS, "
            f"{self.safety_margin * 100:.0f}% safety margin). "
            f"Bottleneck metric: {bottleneck[0]}."
        )

        return RateLimitReport(
            max_sla_compliant_qps=round(overall_max_qps, 2),
            safety_margin=self.safety_margin,
            recommended_qps=round(recommended, 2),
            burst=burst,
            headroom=headroom_list,
            data_points=len(points),
            qps_range=qps_range,
            recommendation=rec_text,
        )

    def _find_max_compliant_qps(
        self,
        qps_values: np.ndarray,
        latencies: np.ndarray,
        sla_threshold: float,
    ) -> float | None:
        """Find max QPS where latency ≤ SLA threshold via linear interpolation."""
        # If even the lowest QPS violates SLA, no compliant point
        if latencies[0] > sla_threshold:
            return None

        # If all points are compliant, max is the highest tested QPS
        if latencies[-1] <= sla_threshold:
            return float(qps_values[-1])

        # Find crossing point via interpolation
        for i in range(len(latencies) - 1):
            if latencies[i] <= sla_threshold < latencies[i + 1]:
                # Linear interpolation between points i and i+1
                frac = (sla_threshold - latencies[i]) / (latencies[i + 1] - latencies[i])
                return float(qps_values[i] + frac * (qps_values[i + 1] - qps_values[i]))

        # Fallback: last compliant point
        compliant_mask = latencies <= sla_threshold
        if compliant_mask.any():
            return float(qps_values[compliant_mask][-1])
        return None

    def _compute_burst(
        self,
        points: list[dict],
        sustained_qps: float,
    ) -> BurstConfig | None:
        """Compute burst allowance from short-window peak analysis."""
        if sustained_qps <= 0:
            return None

        # Estimate burst capacity: look at the ratio between peak per-second rate
        # and average rate within each benchmark, use the max observed burst ratio
        burst_ratios: list[float] = []
        for p in points:
            reqs = p["requests"]
            if len(reqs) < 2:
                continue
            timestamps = sorted(r.timestamp for r in reqs)
            if timestamps[-1] - timestamps[0] < 1.0:
                continue
            # Count requests per 1-second buckets
            t0 = timestamps[0]
            duration = timestamps[-1] - t0
            n_buckets = max(1, int(duration))
            counts = [0] * n_buckets
            for t in timestamps:
                idx = min(int(t - t0), n_buckets - 1)
                counts[idx] += 1
            avg_rate = len(timestamps) / duration
            peak_rate = max(counts)
            if avg_rate > 0:
                burst_ratios.append(peak_rate / avg_rate)

        if not burst_ratios:
            burst_ratio = 1.5  # Default burst ratio
        else:
            burst_ratio = max(1.0, float(np.median(burst_ratios)))

        burst_qps = sustained_qps * burst_ratio

        return BurstConfig(
            sustained_qps=round(sustained_qps, 2),
            burst_qps=round(burst_qps, 2),
            burst_window_seconds=self.burst_window_seconds,
            burst_ratio=round(burst_ratio, 2),
        )


def recommend_rate_limit(
    benchmarks: list[BenchmarkData],
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_ms: float | None = None,
    safety_margin: float = 0.10,
    burst_window_seconds: float = 10.0,
) -> RateLimitReport:
    """Convenience function for rate limit recommendation.

    Parameters
    ----------
    benchmarks : list[BenchmarkData]
        Benchmark files at different QPS levels (at least 2).
    sla_ttft_ms : float | None
        SLA threshold for TTFT P95 (ms).
    sla_tpot_ms : float | None
        SLA threshold for TPOT P95 (ms).
    sla_total_ms : float | None
        SLA threshold for total latency P95 (ms).
    safety_margin : float
        Safety margin fraction (default 0.10).
    burst_window_seconds : float
        Burst analysis window (default 10.0).

    Returns
    -------
    RateLimitReport
    """
    recommender = RateLimitRecommender(
        safety_margin=safety_margin,
        burst_window_seconds=burst_window_seconds,
    )
    return recommender.recommend(
        benchmarks,
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_ms=sla_total_ms,
    )

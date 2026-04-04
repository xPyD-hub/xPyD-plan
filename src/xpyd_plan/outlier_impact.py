"""Outlier impact analysis — quantify how outliers affect SLA compliance and latency percentiles."""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class ImpactRecommendation(str, Enum):
    """Recommendation on whether to filter outliers."""

    FILTER = "filter"  # Outliers materially affect SLA compliance
    KEEP = "keep"      # Outliers have negligible impact


class MetricImpact(BaseModel):
    """Impact of outlier removal on a single latency metric."""

    metric: str = Field(..., description="Metric name (ttft, tpot, total_latency)")
    p50_before_ms: float = Field(..., ge=0)
    p50_after_ms: float = Field(..., ge=0)
    p50_delta_ms: float = Field(..., description="after - before")
    p95_before_ms: float = Field(..., ge=0)
    p95_after_ms: float = Field(..., ge=0)
    p95_delta_ms: float = Field(..., description="after - before")
    p99_before_ms: float = Field(..., ge=0)
    p99_after_ms: float = Field(..., ge=0)
    p99_delta_ms: float = Field(..., description="after - before")


class SLAComplianceComparison(BaseModel):
    """SLA compliance before and after outlier removal."""

    sla_ttft_ms: float | None = Field(None, description="TTFT SLA threshold")
    sla_tpot_ms: float | None = Field(None, description="TPOT SLA threshold")
    sla_total_ms: float | None = Field(None, description="Total latency SLA threshold")
    pass_before: bool = Field(..., description="SLA passed with outliers")
    pass_after: bool = Field(..., description="SLA passed without outliers")
    flipped: bool = Field(..., description="True if pass/fail changed")


class ImpactReport(BaseModel):
    """Complete outlier impact analysis report."""

    total_requests: int = Field(..., ge=0)
    outlier_count: int = Field(..., ge=0)
    outlier_fraction: float = Field(..., ge=0, le=1)
    iqr_multiplier: float = Field(..., gt=0)
    metric_impacts: list[MetricImpact] = Field(
        ..., description="Per-metric impact details"
    )
    sla_comparison: SLAComplianceComparison | None = Field(
        None, description="SLA comparison (only when SLA thresholds provided)"
    )
    recommendation: ImpactRecommendation
    summary: str = Field(..., description="Human-readable summary")


def _iqr_outlier_mask(values: np.ndarray, multiplier: float) -> np.ndarray:
    """Return boolean mask where True = outlier via IQR method."""
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (values < lower) | (values > upper)


def _compute_percentiles(
    values: np.ndarray,
) -> tuple[float, float, float]:
    """Return (P50, P95, P99)."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.percentile(values, 50)),
        float(np.percentile(values, 95)),
        float(np.percentile(values, 99)),
    )


def _check_sla_p95(
    values: np.ndarray,
    threshold: float | None,
) -> bool:
    """Check if P95 of values is within SLA threshold."""
    if threshold is None:
        return True
    if len(values) == 0:
        return True
    return float(np.percentile(values, 95)) <= threshold


class OutlierImpactAnalyzer:
    """Analyze the impact of outliers on latency percentiles and SLA compliance."""

    def analyze(
        self,
        data: BenchmarkData,
        *,
        iqr_multiplier: float = 1.5,
        sla_ttft_ms: float | None = None,
        sla_tpot_ms: float | None = None,
        sla_total_ms: float | None = None,
    ) -> ImpactReport:
        """Run outlier impact analysis.

        Args:
            data: Benchmark data to analyze.
            iqr_multiplier: IQR multiplier for outlier detection (default 1.5).
            sla_ttft_ms: Optional TTFT SLA threshold in ms.
            sla_tpot_ms: Optional TPOT SLA threshold in ms.
            sla_total_ms: Optional total latency SLA threshold in ms.

        Returns:
            ImpactReport with per-metric deltas and recommendation.
        """
        requests = data.requests
        if not requests:
            return ImpactReport(
                total_requests=0,
                outlier_count=0,
                outlier_fraction=0.0,
                iqr_multiplier=iqr_multiplier,
                metric_impacts=[],
                sla_comparison=None,
                recommendation=ImpactRecommendation.KEEP,
                summary="No requests to analyze.",
            )

        ttft = np.array([r.ttft_ms for r in requests])
        tpot = np.array([r.tpot_ms for r in requests])
        total = np.array([r.total_latency_ms for r in requests])

        # Union of outliers across all metrics
        mask = (
            _iqr_outlier_mask(ttft, iqr_multiplier)
            | _iqr_outlier_mask(tpot, iqr_multiplier)
            | _iqr_outlier_mask(total, iqr_multiplier)
        )
        clean_mask = ~mask
        outlier_count = int(mask.sum())
        total_requests = len(requests)

        ttft_clean = ttft[clean_mask]
        tpot_clean = tpot[clean_mask]
        total_clean = total[clean_mask]

        # Compute per-metric impacts
        metric_impacts: list[MetricImpact] = []
        for name, before_arr, after_arr in [
            ("ttft", ttft, ttft_clean),
            ("tpot", tpot, tpot_clean),
            ("total_latency", total, total_clean),
        ]:
            b50, b95, b99 = _compute_percentiles(before_arr)
            a50, a95, a99 = _compute_percentiles(after_arr)
            metric_impacts.append(
                MetricImpact(
                    metric=name,
                    p50_before_ms=b50,
                    p50_after_ms=a50,
                    p50_delta_ms=round(a50 - b50, 4),
                    p95_before_ms=b95,
                    p95_after_ms=a95,
                    p95_delta_ms=round(a95 - b95, 4),
                    p99_before_ms=b99,
                    p99_after_ms=a99,
                    p99_delta_ms=round(a99 - b99, 4),
                )
            )

        # SLA comparison
        has_sla = any(
            t is not None for t in [sla_ttft_ms, sla_tpot_ms, sla_total_ms]
        )
        sla_comparison: SLAComplianceComparison | None = None
        flipped = False
        if has_sla:
            pass_before = (
                _check_sla_p95(ttft, sla_ttft_ms)
                and _check_sla_p95(tpot, sla_tpot_ms)
                and _check_sla_p95(total, sla_total_ms)
            )
            pass_after = (
                _check_sla_p95(ttft_clean, sla_ttft_ms)
                and _check_sla_p95(tpot_clean, sla_tpot_ms)
                and _check_sla_p95(total_clean, sla_total_ms)
            )
            flipped = pass_before != pass_after
            sla_comparison = SLAComplianceComparison(
                sla_ttft_ms=sla_ttft_ms,
                sla_tpot_ms=sla_tpot_ms,
                sla_total_ms=sla_total_ms,
                pass_before=pass_before,
                pass_after=pass_after,
                flipped=flipped,
            )

        # Recommendation
        if flipped:
            recommendation = ImpactRecommendation.FILTER
        else:
            # Check if any P95 delta is > 5% of original value
            material = False
            for mi in metric_impacts:
                if mi.p95_before_ms > 0 and abs(mi.p95_delta_ms) / mi.p95_before_ms > 0.05:
                    material = True
                    break
            recommendation = (
                ImpactRecommendation.FILTER if material
                else ImpactRecommendation.KEEP
            )

        frac = outlier_count / total_requests if total_requests > 0 else 0.0
        summary_parts = [
            f"{outlier_count}/{total_requests} requests ({frac:.1%}) identified as outliers.",
        ]
        if flipped:
            summary_parts.append(
                "SLA compliance flips when outliers are removed — filtering recommended."
            )
        elif recommendation == ImpactRecommendation.FILTER:
            summary_parts.append(
                "Outliers materially shift P95 latency (>5%) — filtering recommended."
            )
        else:
            summary_parts.append(
                "Outliers have negligible impact on results — no filtering needed."
            )

        return ImpactReport(
            total_requests=total_requests,
            outlier_count=outlier_count,
            outlier_fraction=round(frac, 6),
            iqr_multiplier=iqr_multiplier,
            metric_impacts=metric_impacts,
            sla_comparison=sla_comparison,
            recommendation=recommendation,
            summary=" ".join(summary_parts),
        )


def analyze_outlier_impact(
    benchmark_path: str,
    *,
    iqr_multiplier: float = 1.5,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_ms: float | None = None,
) -> dict:
    """Programmatic API for outlier impact analysis.

    Args:
        benchmark_path: Path to benchmark JSON file.
        iqr_multiplier: IQR multiplier for outlier detection.
        sla_ttft_ms: Optional TTFT SLA threshold.
        sla_tpot_ms: Optional TPOT SLA threshold.
        sla_total_ms: Optional total latency SLA threshold.

    Returns:
        Dict representation of ImpactReport.
    """
    from .bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    analyzer = OutlierImpactAnalyzer()
    report = analyzer.analyze(
        data,
        iqr_multiplier=iqr_multiplier,
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_ms=sla_total_ms,
    )
    return report.model_dump()

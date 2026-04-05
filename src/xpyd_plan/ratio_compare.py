"""Latency percentile comparison across P:D ratios."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class PercentileStats(BaseModel):
    """Latency percentiles for a single metric."""

    p50: float = Field(..., description="50th percentile latency (ms)")
    p95: float = Field(..., description="95th percentile latency (ms)")
    p99: float = Field(..., description="99th percentile latency (ms)")


class RatioMetrics(BaseModel):
    """All latency metrics for a single P:D ratio configuration."""

    ratio_label: str = Field(..., description="P:D ratio label (e.g. '4:2')")
    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    total_instances: int = Field(..., ge=2)
    request_count: int = Field(..., ge=1)
    measured_qps: float = Field(..., gt=0)
    ttft: PercentileStats = Field(..., description="TTFT percentiles")
    tpot: PercentileStats = Field(..., description="TPOT percentiles")
    total_latency: PercentileStats = Field(..., description="Total latency percentiles")


class BestRatio(BaseModel):
    """Best ratio for a specific metric+percentile combination."""

    metric: str = Field(..., description="Metric name (ttft, tpot, total_latency)")
    percentile: str = Field(..., description="Percentile level (p50, p95, p99)")
    ratio_label: str = Field(..., description="Best ratio label")
    value_ms: float = Field(..., description="Best value in ms")


class RatioComparison(BaseModel):
    """Complete ratio comparison report."""

    ratios: list[RatioMetrics] = Field(..., description="Per-ratio metrics")
    best_ratios: list[BestRatio] = Field(..., description="Best ratio per metric+percentile")
    overall_best: str = Field(..., description="Overall best ratio (most wins)")


class RatioComparer:
    """Compare latency percentiles across P:D ratio configurations."""

    def compare(self, datasets: list[BenchmarkData]) -> RatioComparison:
        """Compare multiple benchmark datasets with different P:D ratios.

        Args:
            datasets: List of BenchmarkData, each from a different P:D ratio.

        Returns:
            RatioComparison with per-ratio metrics and best-ratio identification.

        Raises:
            ValueError: If fewer than 2 datasets provided.
        """
        if len(datasets) < 2:
            raise ValueError("At least 2 benchmark datasets required for comparison")

        ratios: list[RatioMetrics] = []
        for data in datasets:
            meta = data.metadata
            label = f"{meta.num_prefill_instances}:{meta.num_decode_instances}"
            requests = data.requests

            ttft_vals = [r.ttft_ms for r in requests]
            tpot_vals = [r.tpot_ms for r in requests]
            total_vals = [r.total_latency_ms for r in requests]

            ratios.append(
                RatioMetrics(
                    ratio_label=label,
                    num_prefill=meta.num_prefill_instances,
                    num_decode=meta.num_decode_instances,
                    total_instances=meta.total_instances,
                    request_count=len(requests),
                    measured_qps=meta.measured_qps,
                    ttft=self._compute_percentiles(ttft_vals),
                    tpot=self._compute_percentiles(tpot_vals),
                    total_latency=self._compute_percentiles(total_vals),
                )
            )

        best_ratios = self._find_best_ratios(ratios)
        overall_best = self._find_overall_best(best_ratios)

        return RatioComparison(
            ratios=ratios,
            best_ratios=best_ratios,
            overall_best=overall_best,
        )

    @staticmethod
    def _compute_percentiles(values: list[float]) -> PercentileStats:
        """Compute P50/P95/P99 for a list of values."""
        arr = np.array(values)
        return PercentileStats(
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
        )

    @staticmethod
    def _find_best_ratios(ratios: list[RatioMetrics]) -> list[BestRatio]:
        """Find the best ratio for each metric+percentile combination."""
        results: list[BestRatio] = []
        for metric_name in ("ttft", "tpot", "total_latency"):
            for pct in ("p50", "p95", "p99"):
                best_label = ""
                best_val = float("inf")
                for r in ratios:
                    stats: PercentileStats = getattr(r, metric_name)
                    val = getattr(stats, pct)
                    if val < best_val:
                        best_val = val
                        best_label = r.ratio_label
                results.append(
                    BestRatio(
                        metric=metric_name,
                        percentile=pct,
                        ratio_label=best_label,
                        value_ms=best_val,
                    )
                )
        return results

    @staticmethod
    def _find_overall_best(best_ratios: list[BestRatio]) -> str:
        """Find the ratio that wins in the most metric+percentile combinations."""
        wins: dict[str, int] = {}
        for br in best_ratios:
            wins[br.ratio_label] = wins.get(br.ratio_label, 0) + 1
        return max(wins, key=lambda k: wins[k])


def compare_ratios(datasets: list[BenchmarkData]) -> RatioComparison:
    """Programmatic API for ratio comparison.

    Args:
        datasets: List of BenchmarkData from different P:D ratios.

    Returns:
        RatioComparison report.
    """
    return RatioComparer().compare(datasets)

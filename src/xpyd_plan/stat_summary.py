"""Multi-benchmark statistical summary across repeated runs."""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class RunStability(str, Enum):
    """Stability classification for a single run."""

    MOST_STABLE = "MOST_STABLE"
    LEAST_STABLE = "LEAST_STABLE"
    NORMAL = "NORMAL"


class LatencyAggStats(BaseModel):
    """Aggregated latency statistics across runs for one metric."""

    mean_of_means_ms: float = Field(..., description="Mean of per-run mean latencies")
    std_of_means_ms: float = Field(..., description="Std dev of per-run mean latencies")
    mean_of_p50s_ms: float = Field(..., description="Mean of per-run P50 latencies")
    std_of_p50s_ms: float = Field(..., description="Std dev of per-run P50 latencies")
    mean_of_p95s_ms: float = Field(..., description="Mean of per-run P95 latencies")
    std_of_p95s_ms: float = Field(..., description="Std dev of per-run P95 latencies")
    mean_of_p99s_ms: float = Field(..., description="Mean of per-run P99 latencies")
    std_of_p99s_ms: float = Field(..., description="Std dev of per-run P99 latencies")
    cv_of_p95s: float = Field(
        ..., description="Coefficient of variation of P95s across runs"
    )


class AggregatedStats(BaseModel):
    """Cross-run aggregated statistics."""

    num_runs: int = Field(..., description="Number of benchmark runs")
    total_requests: int = Field(..., description="Total requests across all runs")
    mean_qps: float = Field(..., description="Mean QPS across runs")
    std_qps: float = Field(..., description="Std dev of QPS across runs")
    cv_qps: float = Field(..., description="Coefficient of variation of QPS")
    ttft: LatencyAggStats = Field(..., description="TTFT aggregated stats")
    tpot: LatencyAggStats = Field(..., description="TPOT aggregated stats")
    total_latency: LatencyAggStats = Field(
        ..., description="Total latency aggregated stats"
    )
    repeatability_cv: float = Field(
        ...,
        description="Overall repeatability CV (mean of per-metric P95 CVs)",
    )


class RunSummary(BaseModel):
    """Summary of a single benchmark run within the collection."""

    index: int = Field(..., description="Run index (0-based)")
    request_count: int = Field(..., description="Number of requests in this run")
    measured_qps: float = Field(..., description="Measured QPS")
    ttft_p95_ms: float = Field(..., description="TTFT P95")
    tpot_p95_ms: float = Field(..., description="TPOT P95")
    total_latency_p95_ms: float = Field(..., description="Total latency P95")
    stability: RunStability = Field(
        RunStability.NORMAL, description="Stability classification"
    )


class StatSummaryReport(BaseModel):
    """Multi-benchmark statistical summary report."""

    runs: list[RunSummary] = Field(..., description="Per-run summaries")
    aggregated: AggregatedStats = Field(..., description="Cross-run aggregated stats")
    most_stable_run: int = Field(..., description="Index of most stable run")
    least_stable_run: int = Field(..., description="Index of least stable run")


class StatSummaryAnalyzer:
    """Compute cross-run statistics from multiple benchmark files."""

    def __init__(self, datasets: list[BenchmarkData]) -> None:
        if len(datasets) < 2:
            raise ValueError("At least 2 benchmark datasets are required")
        self._datasets = datasets

    def summarize(self) -> StatSummaryReport:
        """Produce a statistical summary across all runs."""
        run_summaries: list[RunSummary] = []
        qps_list: list[float] = []

        per_run_means: dict[str, list[float]] = {
            "ttft": [], "tpot": [], "total_latency": []
        }
        per_run_p50s: dict[str, list[float]] = {
            "ttft": [], "tpot": [], "total_latency": []
        }
        per_run_p95s: dict[str, list[float]] = {
            "ttft": [], "tpot": [], "total_latency": []
        }
        per_run_p99s: dict[str, list[float]] = {
            "ttft": [], "tpot": [], "total_latency": []
        }

        for i, data in enumerate(self._datasets):
            reqs = data.requests
            ttft_arr = np.array([r.ttft_ms for r in reqs])
            tpot_arr = np.array([r.tpot_ms for r in reqs])
            total_arr = np.array([r.total_latency_ms for r in reqs])

            for key, arr in [("ttft", ttft_arr), ("tpot", tpot_arr), ("total_latency", total_arr)]:
                per_run_means[key].append(float(np.mean(arr)))
                per_run_p50s[key].append(float(np.percentile(arr, 50)))
                per_run_p95s[key].append(float(np.percentile(arr, 95)))
                per_run_p99s[key].append(float(np.percentile(arr, 99)))

            qps_list.append(data.metadata.measured_qps)

            run_summaries.append(RunSummary(
                index=i,
                request_count=len(reqs),
                measured_qps=data.metadata.measured_qps,
                ttft_p95_ms=per_run_p95s["ttft"][-1],
                tpot_p95_ms=per_run_p95s["tpot"][-1],
                total_latency_p95_ms=per_run_p95s["total_latency"][-1],
            ))

        # Build aggregated latency stats
        def _build_latency_agg(key: str) -> LatencyAggStats:
            means = np.array(per_run_means[key])
            p50s = np.array(per_run_p50s[key])
            p95s = np.array(per_run_p95s[key])
            p99s = np.array(per_run_p99s[key])
            mean_p95 = float(np.mean(p95s))
            cv_p95 = float(np.std(p95s) / mean_p95) if mean_p95 > 0 else 0.0
            return LatencyAggStats(
                mean_of_means_ms=float(np.mean(means)),
                std_of_means_ms=float(np.std(means)),
                mean_of_p50s_ms=float(np.mean(p50s)),
                std_of_p50s_ms=float(np.std(p50s)),
                mean_of_p95s_ms=mean_p95,
                std_of_p95s_ms=float(np.std(p95s)),
                mean_of_p99s_ms=float(np.mean(p99s)),
                std_of_p99s_ms=float(np.std(p99s)),
                cv_of_p95s=cv_p95,
            )

        ttft_agg = _build_latency_agg("ttft")
        tpot_agg = _build_latency_agg("tpot")
        total_agg = _build_latency_agg("total_latency")

        qps_arr = np.array(qps_list)
        mean_qps = float(np.mean(qps_arr))
        std_qps = float(np.std(qps_arr))
        cv_qps = std_qps / mean_qps if mean_qps > 0 else 0.0

        repeatability_cv = float(np.mean([
            ttft_agg.cv_of_p95s, tpot_agg.cv_of_p95s, total_agg.cv_of_p95s
        ]))

        aggregated = AggregatedStats(
            num_runs=len(self._datasets),
            total_requests=sum(r.request_count for r in run_summaries),
            mean_qps=mean_qps,
            std_qps=std_qps,
            cv_qps=cv_qps,
            ttft=ttft_agg,
            tpot=tpot_agg,
            total_latency=total_agg,
            repeatability_cv=repeatability_cv,
        )

        # Identify most/least stable runs by sum of normalized P95 deviations
        deviations: list[float] = []
        for i in range(len(self._datasets)):
            dev = 0.0
            for key in ("ttft", "tpot", "total_latency"):
                p95_mean = float(np.mean(per_run_p95s[key]))
                if p95_mean > 0:
                    dev += abs(per_run_p95s[key][i] - p95_mean) / p95_mean
            deviations.append(dev)

        most_stable = int(np.argmin(deviations))
        least_stable = int(np.argmax(deviations))

        run_summaries[most_stable].stability = RunStability.MOST_STABLE
        run_summaries[least_stable].stability = RunStability.LEAST_STABLE

        return StatSummaryReport(
            runs=run_summaries,
            aggregated=aggregated,
            most_stable_run=most_stable,
            least_stable_run=least_stable,
        )


def summarize_stats(datasets: list[BenchmarkData]) -> dict:
    """Programmatic API: summarize cross-run statistics.

    Args:
        datasets: List of BenchmarkData objects (at least 2).

    Returns:
        Dictionary with the full StatSummaryReport.
    """
    analyzer = StatSummaryAnalyzer(datasets)
    report = analyzer.summarize()
    return report.model_dump()

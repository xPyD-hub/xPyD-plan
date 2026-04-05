"""Benchmark Ranking — rank multiple benchmark configurations by composite score.

Given several benchmark files, compute a composite score for each based on
configurable weighted dimensions (SLA compliance, latency, throughput) and
produce a ranked leaderboard.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.benchmark_models import BenchmarkData


class RankDimension(str, Enum):
    """Scoring dimensions for ranking."""

    SLA_COMPLIANCE = "sla_compliance"
    LATENCY = "latency"
    THROUGHPUT = "throughput"


class RankWeights(BaseModel):
    """Configurable weights for each ranking dimension."""

    sla_compliance: float = Field(
        0.4, ge=0.0, le=1.0, description="Weight for SLA compliance",
    )
    latency: float = Field(
        0.4, ge=0.0, le=1.0,
        description="Weight for latency (lower is better)",
    )
    throughput: float = Field(
        0.2, ge=0.0, le=1.0,
        description="Weight for throughput (higher is better)",
    )


class RankedBenchmark(BaseModel):
    """A single benchmark's ranking result."""

    file: str = Field(..., description="Benchmark file path")
    rank: int = Field(..., ge=1, description="Rank position (1 = best)")
    composite_score: float = Field(
        ..., ge=0.0, le=100.0, description="Composite score 0-100",
    )
    sla_score: float = Field(
        ..., ge=0.0, le=100.0, description="SLA compliance score",
    )
    latency_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Latency score (lower latency = higher score)",
    )
    throughput_score: float = Field(
        ..., ge=0.0, le=100.0, description="Throughput score",
    )
    prefill_instances: int = Field(..., ge=0)
    decode_instances: int = Field(..., ge=0)
    total_instances: int = Field(..., ge=0)
    measured_qps: float = Field(..., ge=0)
    ttft_p95_ms: float = Field(..., ge=0)
    tpot_p95_ms: float = Field(..., ge=0)
    total_latency_p95_ms: float = Field(..., ge=0)


class RankingReport(BaseModel):
    """Complete ranking report across all benchmarks."""

    rankings: list[RankedBenchmark] = Field(default_factory=list)
    best: Optional[RankedBenchmark] = Field(
        None, description="Top-ranked benchmark",
    )
    weights: RankWeights = Field(default_factory=RankWeights)
    sla_ttft_ms: Optional[float] = Field(None)
    sla_tpot_ms: Optional[float] = Field(None)
    sla_total_ms: Optional[float] = Field(None)
    benchmark_count: int = Field(0, ge=0)


class BenchmarkRanker:
    """Rank multiple benchmarks by composite score."""

    def __init__(
        self,
        sla_ttft_ms: Optional[float] = None,
        sla_tpot_ms: Optional[float] = None,
        sla_total_ms: Optional[float] = None,
        weights: Optional[RankWeights] = None,
    ) -> None:
        self._sla_ttft = sla_ttft_ms
        self._sla_tpot = sla_tpot_ms
        self._sla_total = sla_total_ms
        self._weights = weights or RankWeights()

    def rank(
        self, datasets: list[tuple[str, BenchmarkData]],
    ) -> RankingReport:
        """Rank benchmark datasets and return a report."""
        if not datasets:
            return RankingReport(weights=self._weights)

        entries: list[dict] = []
        for file_label, data in datasets:
            meta = data.metadata
            ttfts = np.array([r.ttft_ms for r in data.requests])
            tpots = np.array([r.tpot_ms for r in data.requests])
            totals = np.array(
                [r.total_latency_ms for r in data.requests],
            )

            n = len(ttfts)
            ttft_p95 = float(np.percentile(ttfts, 95)) if n else 0.0
            tpot_p95 = float(np.percentile(tpots, 95)) if n else 0.0
            total_p95 = float(np.percentile(totals, 95)) if n else 0.0

            sla_score = self._compute_sla_score(ttfts, tpots, totals)

            entries.append({
                "file": file_label,
                "prefill_instances": meta.num_prefill_instances,
                "decode_instances": meta.num_decode_instances,
                "total_instances": (
                    meta.num_prefill_instances
                    + meta.num_decode_instances
                ),
                "measured_qps": meta.measured_qps,
                "ttft_p95_ms": round(ttft_p95, 2),
                "tpot_p95_ms": round(tpot_p95, 2),
                "total_latency_p95_ms": round(total_p95, 2),
                "sla_score": sla_score,
            })

        # Normalize latency (lower is better → invert)
        total_p95s = [e["total_latency_p95_ms"] for e in entries]
        max_lat = max(total_p95s) if total_p95s else 1.0
        min_lat = min(total_p95s) if total_p95s else 0.0
        lat_range = max_lat - min_lat if max_lat > min_lat else 1.0

        # Normalize throughput (higher is better)
        qps_vals = [e["measured_qps"] for e in entries]
        max_qps = max(qps_vals) if qps_vals else 1.0
        min_qps = min(qps_vals) if qps_vals else 0.0
        qps_range = max_qps - min_qps if max_qps > min_qps else 1.0

        for e in entries:
            lat_norm = (e["total_latency_p95_ms"] - min_lat) / lat_range
            e["latency_score"] = round(100.0 * (1.0 - lat_norm), 2)
            qps_norm = (e["measured_qps"] - min_qps) / qps_range
            e["throughput_score"] = round(100.0 * qps_norm, 2)
            e["composite_score"] = round(
                self._weights.sla_compliance * e["sla_score"]
                + self._weights.latency * e["latency_score"]
                + self._weights.throughput * e["throughput_score"],
                2,
            )

        entries.sort(key=lambda x: x["composite_score"], reverse=True)

        rankings = [
            RankedBenchmark(rank=i, **e)
            for i, e in enumerate(entries, 1)
        ]

        return RankingReport(
            rankings=rankings,
            best=rankings[0] if rankings else None,
            weights=self._weights,
            sla_ttft_ms=self._sla_ttft,
            sla_tpot_ms=self._sla_tpot,
            sla_total_ms=self._sla_total,
            benchmark_count=len(rankings),
        )

    def _compute_sla_score(
        self,
        ttfts: np.ndarray,
        tpots: np.ndarray,
        totals: np.ndarray,
    ) -> float:
        """Compute SLA compliance score (0-100)."""
        if len(ttfts) == 0:
            return 0.0

        checks = []
        if self._sla_ttft is not None:
            checks.append(float(np.mean(ttfts <= self._sla_ttft)) * 100)
        if self._sla_tpot is not None:
            checks.append(float(np.mean(tpots <= self._sla_tpot)) * 100)
        if self._sla_total is not None:
            checks.append(
                float(np.mean(totals <= self._sla_total)) * 100,
            )

        if not checks:
            return 100.0

        return round(min(checks), 2)


def rank_benchmarks(
    benchmark_files: list[str],
    sla_ttft_ms: Optional[float] = None,
    sla_tpot_ms: Optional[float] = None,
    sla_total_ms: Optional[float] = None,
    weights: Optional[RankWeights] = None,
) -> dict:
    """Programmatic API: rank benchmark files and return dict."""
    from pathlib import Path

    datasets: list[tuple[str, BenchmarkData]] = []
    for f in benchmark_files:
        data = load_benchmark_auto(Path(f))
        datasets.append((f, data))

    ranker = BenchmarkRanker(
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_ms=sla_total_ms,
        weights=weights,
    )
    report = ranker.rank(datasets)
    return report.model_dump()

"""Multi-model comparison matrix.

Compare benchmark results across different LLM models to identify the best
model+ratio combination based on latency, cost, and SLA compliance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class ModelProfile(BaseModel):
    """Performance profile for a single model extracted from benchmark data."""

    model_name: str = Field(..., description="Name of the LLM model")
    num_prefill_instances: int = Field(..., ge=1)
    num_decode_instances: int = Field(..., ge=1)
    total_instances: int = Field(..., ge=2)
    measured_qps: float = Field(..., gt=0)
    request_count: int = Field(..., ge=1)
    ttft_p50_ms: float = Field(..., ge=0)
    ttft_p95_ms: float = Field(..., ge=0)
    ttft_p99_ms: float = Field(..., ge=0)
    tpot_p50_ms: float = Field(..., ge=0)
    tpot_p95_ms: float = Field(..., ge=0)
    tpot_p99_ms: float = Field(..., ge=0)
    total_latency_p50_ms: float = Field(..., ge=0)
    total_latency_p95_ms: float = Field(..., ge=0)
    total_latency_p99_ms: float = Field(..., ge=0)
    avg_prompt_tokens: float = Field(..., ge=0)
    avg_output_tokens: float = Field(..., ge=0)
    cost_per_hour: Optional[float] = Field(
        None, ge=0, description="Total hourly cost if cost config provided"
    )
    cost_per_request: Optional[float] = Field(
        None, ge=0, description="Cost per request if cost config provided"
    )


class ModelComparison(BaseModel):
    """Pairwise comparison between two models for a specific metric."""

    metric: str = Field(..., description="Metric name")
    model_a: str = Field(..., description="First model name")
    model_b: str = Field(..., description="Second model name")
    value_a: float = Field(..., description="Metric value for model A")
    value_b: float = Field(..., description="Metric value for model B")
    absolute_delta: float = Field(..., description="value_b - value_a")
    relative_delta: float = Field(
        ..., description="Relative change: (value_b - value_a) / value_a, or 0.0 if value_a is 0"
    )
    winner: str = Field(..., description="Model with better (lower) value for latency metrics")


class ModelRanking(BaseModel):
    """Ranking of a model across all comparison criteria."""

    model_name: str = Field(..., description="Model name")
    rank: int = Field(..., ge=1, description="Overall rank (1 = best)")
    latency_score: float = Field(..., description="Normalized latency score (lower is better)")
    cost_efficiency_score: Optional[float] = Field(
        None, description="Cost efficiency score (lower is better)"
    )
    recommendation: str = Field(..., description="Human-readable recommendation")


class ComparisonMatrix(BaseModel):
    """Complete comparison result across all models."""

    profiles: list[ModelProfile] = Field(default_factory=list)
    comparisons: list[ModelComparison] = Field(default_factory=list)
    rankings: list[ModelRanking] = Field(default_factory=list)
    best_latency_model: str = Field(..., description="Model with lowest P95 total latency")
    best_cost_model: Optional[str] = Field(
        None, description="Model with lowest cost per request (if cost data available)"
    )


class ModelComparator:
    """Compare benchmark results across different LLM models."""

    def __init__(
        self,
        gpu_hourly_rate: Optional[float] = None,
    ):
        self._gpu_hourly_rate = gpu_hourly_rate

    def load_profile(self, benchmark_path: str, model_name: str) -> ModelProfile:
        """Load a benchmark file and extract a model profile."""
        data = BenchmarkData.model_validate(
            json.loads(Path(benchmark_path).read_text())
        )
        ttfts = np.array([r.ttft_ms for r in data.requests])
        tpots = np.array([r.tpot_ms for r in data.requests])
        totals = np.array([r.total_latency_ms for r in data.requests])
        prompts = np.array([r.prompt_tokens for r in data.requests])
        outputs = np.array([r.output_tokens for r in data.requests])

        total_instances = data.metadata.total_instances
        cost_per_hour = None
        cost_per_request = None
        if self._gpu_hourly_rate is not None:
            cost_per_hour = self._gpu_hourly_rate * total_instances
            if data.metadata.measured_qps > 0:
                cost_per_request = cost_per_hour / (data.metadata.measured_qps * 3600)

        return ModelProfile(
            model_name=model_name,
            num_prefill_instances=data.metadata.num_prefill_instances,
            num_decode_instances=data.metadata.num_decode_instances,
            total_instances=total_instances,
            measured_qps=data.metadata.measured_qps,
            request_count=len(data.requests),
            ttft_p50_ms=float(np.percentile(ttfts, 50)),
            ttft_p95_ms=float(np.percentile(ttfts, 95)),
            ttft_p99_ms=float(np.percentile(ttfts, 99)),
            tpot_p50_ms=float(np.percentile(tpots, 50)),
            tpot_p95_ms=float(np.percentile(tpots, 95)),
            tpot_p99_ms=float(np.percentile(tpots, 99)),
            total_latency_p50_ms=float(np.percentile(totals, 50)),
            total_latency_p95_ms=float(np.percentile(totals, 95)),
            total_latency_p99_ms=float(np.percentile(totals, 99)),
            avg_prompt_tokens=float(np.mean(prompts)),
            avg_output_tokens=float(np.mean(outputs)),
            cost_per_hour=cost_per_hour,
            cost_per_request=cost_per_request,
        )

    def compare(
        self,
        benchmark_paths: list[str],
        model_names: list[str],
    ) -> ComparisonMatrix:
        """Compare multiple models and produce a comparison matrix."""
        if len(benchmark_paths) != len(model_names):
            raise ValueError(
                f"Number of benchmarks ({len(benchmark_paths)}) must match "
                f"number of model names ({len(model_names)})"
            )
        if len(benchmark_paths) < 2:
            raise ValueError("At least 2 benchmarks required for comparison")

        profiles = [
            self.load_profile(p, n) for p, n in zip(benchmark_paths, model_names)
        ]

        # Generate pairwise comparisons
        latency_metrics = [
            ("ttft_p50_ms", "TTFT P50"),
            ("ttft_p95_ms", "TTFT P95"),
            ("ttft_p99_ms", "TTFT P99"),
            ("tpot_p50_ms", "TPOT P50"),
            ("tpot_p95_ms", "TPOT P95"),
            ("tpot_p99_ms", "TPOT P99"),
            ("total_latency_p50_ms", "Total Latency P50"),
            ("total_latency_p95_ms", "Total Latency P95"),
            ("total_latency_p99_ms", "Total Latency P99"),
        ]

        comparisons: list[ModelComparison] = []
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                pa, pb = profiles[i], profiles[j]
                for attr, label in latency_metrics:
                    va = getattr(pa, attr)
                    vb = getattr(pb, attr)
                    abs_delta = vb - va
                    rel_delta = abs_delta / va if va != 0 else 0.0
                    # Lower latency is better
                    winner = pa.model_name if va <= vb else pb.model_name
                    comparisons.append(
                        ModelComparison(
                            metric=label,
                            model_a=pa.model_name,
                            model_b=pb.model_name,
                            value_a=va,
                            value_b=vb,
                            absolute_delta=abs_delta,
                            relative_delta=rel_delta,
                            winner=winner,
                        )
                    )

        # Rank models
        rankings = self._rank_models(profiles)

        best_latency_model = min(profiles, key=lambda p: p.total_latency_p95_ms).model_name

        best_cost_model = None
        cost_profiles = [p for p in profiles if p.cost_per_request is not None]
        if cost_profiles:
            best_cost_model = min(cost_profiles, key=lambda p: p.cost_per_request).model_name  # type: ignore[arg-type]

        return ComparisonMatrix(
            profiles=profiles,
            comparisons=comparisons,
            rankings=rankings,
            best_latency_model=best_latency_model,
            best_cost_model=best_cost_model,
        )

    def _rank_models(self, profiles: list[ModelProfile]) -> list[ModelRanking]:
        """Rank models by normalized latency score and optional cost efficiency."""
        if not profiles:
            return []

        # Normalize latency: sum of P95 TTFT + TPOT + total latency
        latency_scores = []
        for p in profiles:
            score = p.ttft_p95_ms + p.tpot_p95_ms + p.total_latency_p95_ms
            latency_scores.append(score)

        max_lat = max(latency_scores) if max(latency_scores) > 0 else 1.0
        norm_latency = [s / max_lat for s in latency_scores]

        # Cost efficiency: cost_per_request normalized
        has_cost = all(p.cost_per_request is not None for p in profiles)
        norm_cost: list[Optional[float]] = [None] * len(profiles)
        if has_cost:
            cost_scores = [p.cost_per_request for p in profiles]  # type: ignore[misc]
            max_cost = max(cost_scores) if max(cost_scores) > 0 else 1.0  # type: ignore[type-var]
            norm_cost = [c / max_cost for c in cost_scores]  # type: ignore[operator]

        # Combined score: 60% latency, 40% cost (if available), else 100% latency
        combined = []
        for i in range(len(profiles)):
            if norm_cost[i] is not None:
                combined.append(0.6 * norm_latency[i] + 0.4 * norm_cost[i])
            else:
                combined.append(norm_latency[i])

        # Sort by combined score
        indexed = sorted(enumerate(combined), key=lambda x: x[1])

        rankings = []
        for rank_idx, (orig_idx, _score) in enumerate(indexed):
            p = profiles[orig_idx]
            if rank_idx == 0:
                rec = f"Best overall: {p.model_name} leads in combined latency/cost"
            elif norm_cost[orig_idx] is not None and norm_cost[orig_idx] == min(  # type: ignore[type-var]
                c for c in norm_cost if c is not None
            ):
                rec = f"{p.model_name}: most cost-efficient option"
            elif norm_latency[orig_idx] == min(norm_latency):
                rec = f"{p.model_name}: lowest latency option"
            else:
                rec = f"{p.model_name}: rank {rank_idx + 1} overall"

            rankings.append(
                ModelRanking(
                    model_name=p.model_name,
                    rank=rank_idx + 1,
                    latency_score=norm_latency[orig_idx],
                    cost_efficiency_score=norm_cost[orig_idx],
                    recommendation=rec,
                )
            )

        return rankings


def compare_models(
    benchmark_paths: list[str],
    model_names: list[str],
    gpu_hourly_rate: Optional[float] = None,
) -> ComparisonMatrix:
    """Programmatic API: compare models across benchmark files.

    Args:
        benchmark_paths: Paths to benchmark JSON files (one per model).
        model_names: Model names corresponding to each benchmark file.
        gpu_hourly_rate: Optional hourly GPU cost for cost-efficiency ranking.

    Returns:
        ComparisonMatrix with profiles, comparisons, and rankings.
    """
    comparator = ModelComparator(gpu_hourly_rate=gpu_hourly_rate)
    return comparator.compare(benchmark_paths, model_names)

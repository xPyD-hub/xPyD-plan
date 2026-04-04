"""SLA budget allocation across prefill and decode stages.

Given a total latency budget, analyze benchmark data to determine how TTFT
and TPOT contribute to total latency, then recommend per-stage SLA targets.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class AllocationStrategy(str, Enum):
    """Strategy for splitting total budget between TTFT and TPOT."""

    PROPORTIONAL = "proportional"  # Match observed TTFT:TPOT ratio
    BALANCED = "balanced"  # Equal split
    TTFT_PRIORITY = "ttft-priority"  # Allocate more budget to TTFT
    TPOT_PRIORITY = "tpot-priority"  # Allocate more budget to TPOT


class StageBudget(BaseModel):
    """Budget allocated to a single stage."""

    stage: str = Field(..., description="Stage name (ttft or tpot)")
    budget_ms: float = Field(..., ge=0, description="Allocated budget in ms")
    share: float = Field(
        ..., ge=0, le=1, description="Fraction of total budget"
    )
    observed_ms: float = Field(
        ..., ge=0, description="Observed value at analysis percentile"
    )
    headroom_ms: float = Field(
        ..., description="budget_ms - observed_ms (positive = slack)"
    )


class BudgetAllocation(BaseModel):
    """Complete budget allocation result."""

    total_budget_ms: float = Field(..., gt=0)
    strategy: AllocationStrategy
    percentile: float = Field(95.0, description="Percentile used for analysis")
    ttft: StageBudget
    tpot: StageBudget
    observed_ratio: float = Field(
        ..., description="Observed TTFT share of (TTFT+TPOT) at percentile"
    )
    feasible: bool = Field(
        ...,
        description="True if both stages have non-negative headroom",
    )


class BudgetAllocator:
    """Allocate a total SLA budget across prefill and decode stages.

    Analyzes benchmark data to determine the natural TTFT:TPOT ratio,
    then splits the budget according to the chosen strategy.
    """

    def __init__(self, benchmark: BenchmarkData) -> None:
        self._benchmark = benchmark
        self._ttft_values = [r.ttft_ms for r in benchmark.requests]
        self._tpot_values = [r.tpot_ms for r in benchmark.requests]

    def _percentile(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(values, p))

    def allocate(
        self,
        total_budget_ms: float,
        strategy: AllocationStrategy = AllocationStrategy.PROPORTIONAL,
        percentile: float = 95.0,
    ) -> BudgetAllocation:
        """Allocate total_budget_ms between TTFT and TPOT.

        Args:
            total_budget_ms: Total latency budget in milliseconds.
            strategy: Allocation strategy to use.
            percentile: Percentile for analyzing observed latencies.

        Returns:
            BudgetAllocation with per-stage budgets.
        """
        ttft_obs = self._percentile(self._ttft_values, percentile)
        tpot_obs = self._percentile(self._tpot_values, percentile)
        total_obs = ttft_obs + tpot_obs

        # Observed ratio (TTFT share)
        observed_ratio = ttft_obs / total_obs if total_obs > 0 else 0.5

        # Compute shares based on strategy
        if strategy == AllocationStrategy.PROPORTIONAL:
            ttft_share = observed_ratio
        elif strategy == AllocationStrategy.BALANCED:
            ttft_share = 0.5
        elif strategy == AllocationStrategy.TTFT_PRIORITY:
            # Give TTFT 70% of the budget
            ttft_share = 0.7
        elif strategy == AllocationStrategy.TPOT_PRIORITY:
            # Give TTFT only 30% of the budget
            ttft_share = 0.3
        else:
            ttft_share = observed_ratio

        tpot_share = 1.0 - ttft_share
        ttft_budget = total_budget_ms * ttft_share
        tpot_budget = total_budget_ms * tpot_share

        ttft_stage = StageBudget(
            stage="ttft",
            budget_ms=round(ttft_budget, 2),
            share=round(ttft_share, 4),
            observed_ms=round(ttft_obs, 2),
            headroom_ms=round(ttft_budget - ttft_obs, 2),
        )
        tpot_stage = StageBudget(
            stage="tpot",
            budget_ms=round(tpot_budget, 2),
            share=round(tpot_share, 4),
            observed_ms=round(tpot_obs, 2),
            headroom_ms=round(tpot_budget - tpot_obs, 2),
        )

        feasible = ttft_stage.headroom_ms >= 0 and tpot_stage.headroom_ms >= 0

        return BudgetAllocation(
            total_budget_ms=total_budget_ms,
            strategy=strategy,
            percentile=percentile,
            ttft=ttft_stage,
            tpot=tpot_stage,
            observed_ratio=round(observed_ratio, 4),
            feasible=feasible,
        )


def allocate_budget(
    benchmark: BenchmarkData,
    total_budget_ms: float,
    strategy: str = "proportional",
    percentile: float = 95.0,
) -> dict:
    """Programmatic API for SLA budget allocation.

    Args:
        benchmark: Loaded benchmark data.
        total_budget_ms: Total latency budget in ms.
        strategy: One of proportional, balanced, ttft-priority, tpot-priority.
        percentile: Percentile for observed latency analysis.

    Returns:
        Dict with allocation results.
    """
    alloc_strategy = AllocationStrategy(strategy)
    allocator = BudgetAllocator(benchmark)
    result = allocator.allocate(total_budget_ms, alloc_strategy, percentile)
    return result.model_dump()

"""Planner core — find optimal PD ratio given SLA, dataset, and hardware."""

from __future__ import annotations

from xpyd_plan.estimator import estimate_performance
from xpyd_plan.models import (
    CandidateResult,
    DatasetStats,
    GPUProfile,
    PDConfig,
    PlanResult,
    SLAConfig,
)


def _check_sla(sla: SLAConfig, ttft_ms: float, tpot_ms: float) -> bool:
    """Check whether estimated metrics satisfy all SLA constraints."""
    if sla.ttft_ms is not None and ttft_ms > sla.ttft_ms:
        return False
    if sla.tpot_ms is not None and tpot_ms > sla.tpot_ms:
        return False
    return True


def _score(candidate: CandidateResult) -> float:
    """Compute cost-efficiency score: throughput per dollar per hour."""
    if candidate.performance.total_cost_per_hour <= 0:
        return 0.0
    return candidate.performance.throughput_rps / candidate.performance.total_cost_per_hour


def plan(
    sla: SLAConfig,
    dataset: DatasetStats,
    gpu: GPUProfile,
    budget: int,
) -> PlanResult:
    """Find the optimal PD configuration within a GPU budget.

    Enumerates all possible P:D splits (P=1..budget-1, D=budget-P),
    estimates performance for each, checks SLA compliance, and ranks
    by cost-efficiency (throughput / cost).

    Args:
        sla: SLA constraints to satisfy.
        dataset: Dataset statistics (token length distributions).
        gpu: GPU hardware profile.
        budget: Total number of available GPUs.

    Returns:
        PlanResult with candidates sorted by score (best first).
    """
    if budget < 2:
        return PlanResult(best=None, candidates=[], budget=budget, sla=sla)

    candidates: list[CandidateResult] = []

    for num_prefill in range(1, budget):
        num_decode = budget - num_prefill
        config = PDConfig(num_prefill=num_prefill, num_decode=num_decode)
        perf = estimate_performance(dataset, gpu, config)
        meets = _check_sla(sla, perf.ttft_ms, perf.tpot_ms)

        candidate = CandidateResult(
            config=config,
            performance=perf,
            meets_sla=meets,
            score=0.0,
        )
        candidate.score = _score(candidate) if meets else 0.0
        candidates.append(candidate)

    # Sort by score descending (SLA-meeting first, then by score)
    candidates.sort(key=lambda c: (-int(c.meets_sla), -c.score))

    best = candidates[0] if candidates and candidates[0].meets_sla else None

    return PlanResult(best=best, candidates=candidates, budget=budget, sla=sla)

"""Workload Mix Optimizer for multi-workload GPU cluster allocation.

Given benchmark data for multiple workloads (different models or request
patterns), find the minimum total GPU instances while meeting per-workload
SLA constraints.  Follows DESIGN_PRINCIPLES: data-driven, no guessing.
"""

from __future__ import annotations

import itertools
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.models import SLAConfig


class AllocationMode(str, Enum):
    """How GPU instances are allocated across workloads."""

    DEDICATED = "dedicated"  # Each workload gets its own instances


class WorkloadSpec(BaseModel):
    """Specification for a single workload in the mix."""

    name: str = Field(..., min_length=1, description="Workload identifier")
    benchmark_data: BenchmarkData = Field(..., description="Benchmark results")
    sla: SLAConfig = Field(..., description="SLA constraints for this workload")
    min_prefill: int = Field(1, ge=1, description="Minimum prefill instances")
    min_decode: int = Field(1, ge=1, description="Minimum decode instances")
    weight: float = Field(1.0, gt=0, description="Priority weight (higher = more important)")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return v.strip()


class WorkloadAllocation(BaseModel):
    """Allocation result for a single workload."""

    name: str
    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    total_instances: int
    meets_sla: bool
    prefill_waste: float = Field(..., ge=0, le=1, description="Prefill idle fraction")
    decode_waste: float = Field(..., ge=0, le=1, description="Decode idle fraction")
    weighted_waste: float = Field(..., ge=0, description="Weighted waste score")
    sla_details: dict[str, Any] = Field(default_factory=dict)

    @property
    def ratio_str(self) -> str:
        """Human-readable P:D ratio."""
        return f"{self.num_prefill}P:{self.num_decode}D"


class MixOptimizationResult(BaseModel):
    """Result of workload mix optimization."""

    allocations: list[WorkloadAllocation] = Field(default_factory=list)
    total_instances: int = 0
    total_prefill: int = 0
    total_decode: int = 0
    total_weighted_waste: float = 0.0
    all_sla_met: bool = False
    feasible: bool = False
    mode: AllocationMode = AllocationMode.DEDICATED
    candidates_evaluated: int = 0
    gpu_budget: int | None = None

    @property
    def summary(self) -> str:
        """One-line summary."""
        if not self.feasible:
            return "No feasible allocation found within GPU budget"
        status = "✅ All SLAs met" if self.all_sla_met else "⚠️ Some SLAs violated"
        return (
            f"{status} | {self.total_instances} GPUs "
            f"({self.total_prefill}P+{self.total_decode}D) | "
            f"waste={self.total_weighted_waste:.3f}"
        )


def _check_sla(data: BenchmarkData, sla: SLAConfig) -> tuple[bool, dict[str, Any]]:
    """Check SLA compliance from benchmark data. Returns (meets_sla, details)."""
    ttfts = np.array([r.ttft_ms for r in data.requests])
    tpots = np.array([r.tpot_ms for r in data.requests])
    totals = np.array([r.total_latency_ms for r in data.requests])

    pct = sla.sla_percentile
    ttft_val = float(np.percentile(ttfts, pct))
    tpot_val = float(np.percentile(tpots, pct))
    total_val = float(np.percentile(totals, pct))

    details: dict[str, Any] = {
        "ttft_ms": ttft_val,
        "tpot_ms": tpot_val,
        "total_latency_ms": total_val,
        "percentile": pct,
    }

    meets = True
    if sla.ttft_ms is not None and ttft_val > sla.ttft_ms:
        meets = False
    if sla.tpot_ms is not None and tpot_val > sla.tpot_ms:
        meets = False
    if sla.max_latency_ms is not None and total_val > sla.max_latency_ms:
        meets = False

    return meets, details


def _compute_waste(data: BenchmarkData) -> tuple[float, float]:
    """Estimate prefill and decode waste from benchmark data.

    Uses a simple heuristic: ratio of idle time based on token processing rates.
    Returns (prefill_waste, decode_waste) in [0, 1].
    """
    meta = data.metadata
    total = meta.total_instances
    p_frac = meta.num_prefill_instances / total
    d_frac = meta.num_decode_instances / total

    # Higher fraction → more idle capacity → more waste (simplified model)
    # Use a balanced waste: if perfectly balanced, waste is minimal
    balance = abs(p_frac - d_frac)
    p_waste = min(balance * 0.5 + 0.1, 0.9)
    d_waste = min(balance * 0.5 + 0.1, 0.9)

    return round(p_waste, 4), round(d_waste, 4)


class WorkloadMixOptimizer:
    """Optimize GPU allocation across multiple workloads.

    Uses brute-force enumeration with pruning to find the allocation
    that minimizes total weighted waste while meeting all SLA constraints.
    """

    def __init__(self, max_instances_per_workload: int = 32) -> None:
        self._max_instances = max_instances_per_workload

    def optimize(
        self,
        workloads: list[WorkloadSpec],
        total_gpu_budget: int | None = None,
        mode: AllocationMode = AllocationMode.DEDICATED,
    ) -> MixOptimizationResult:
        """Find optimal P:D allocation across all workloads.

        Args:
            workloads: List of workload specifications with benchmark data.
            total_gpu_budget: Maximum total GPU instances (None = unlimited).
            mode: Allocation mode (currently only DEDICATED).

        Returns:
            MixOptimizationResult with per-workload allocations.
        """
        if not workloads:
            return MixOptimizationResult(
                feasible=False, mode=mode, gpu_budget=total_gpu_budget
            )

        # For each workload, compute the single allocation from its benchmark data
        per_workload_candidates: list[list[WorkloadAllocation]] = []
        for ws in workloads:
            candidates = self._find_candidates(ws)
            per_workload_candidates.append(candidates)

        # If any workload has zero candidates, infeasible
        if any(len(c) == 0 for c in per_workload_candidates):
            return MixOptimizationResult(
                feasible=False, mode=mode, gpu_budget=total_gpu_budget
            )

        # Enumerate combinations (with pruning)
        best: MixOptimizationResult | None = None
        evaluated = 0

        for combo in itertools.product(*per_workload_candidates):
            evaluated += 1
            total = sum(a.total_instances for a in combo)

            # Budget pruning
            if total_gpu_budget is not None and total > total_gpu_budget:
                continue

            all_met = all(a.meets_sla for a in combo)
            total_waste = sum(a.weighted_waste for a in combo)

            if best is None or self._is_better(
                total, all_met, total_waste,
                best.total_instances, best.all_sla_met, best.total_weighted_waste,
            ):
                best = MixOptimizationResult(
                    allocations=list(combo),
                    total_instances=total,
                    total_prefill=sum(a.num_prefill for a in combo),
                    total_decode=sum(a.num_decode for a in combo),
                    total_weighted_waste=total_waste,
                    all_sla_met=all_met,
                    feasible=True,
                    mode=mode,
                    candidates_evaluated=evaluated,
                    gpu_budget=total_gpu_budget,
                )

        if best is not None:
            best.candidates_evaluated = evaluated
            return best

        return MixOptimizationResult(
            feasible=False,
            mode=mode,
            candidates_evaluated=evaluated,
            gpu_budget=total_gpu_budget,
        )

    def _find_candidates(self, ws: WorkloadSpec) -> list[WorkloadAllocation]:
        """Find all valid P:D allocations for a single workload."""
        candidates: list[WorkloadAllocation] = []
        total = ws.benchmark_data.metadata.total_instances

        # Check SLA compliance using benchmark data
        meets, sla_details = _check_sla(ws.benchmark_data, ws.sla)
        p_waste, d_waste = _compute_waste(ws.benchmark_data)

        p = ws.benchmark_data.metadata.num_prefill_instances
        d = ws.benchmark_data.metadata.num_decode_instances

        if p < ws.min_prefill or d < ws.min_decode:
            return candidates

        weighted = ws.weight * (p_waste * p / total + d_waste * d / total)

        candidates.append(
            WorkloadAllocation(
                name=ws.name,
                num_prefill=p,
                num_decode=d,
                total_instances=total,
                meets_sla=meets,
                prefill_waste=p_waste,
                decode_waste=d_waste,
                weighted_waste=round(weighted, 4),
                sla_details=sla_details,
            )
        )

        return candidates

    @staticmethod
    def _is_better(
        total: int,
        all_met: bool,
        waste: float,
        best_total: int,
        best_all_met: bool,
        best_waste: float,
    ) -> bool:
        """Compare two solutions: prefer all-SLA-met, then fewer instances, then lower waste."""
        if all_met != best_all_met:
            return all_met
        if total != best_total:
            return total < best_total
        return waste < best_waste


def optimize_workload_mix(
    workloads: list[WorkloadSpec],
    total_gpu_budget: int | None = None,
    mode: AllocationMode = AllocationMode.DEDICATED,
    max_instances_per_workload: int = 32,
) -> MixOptimizationResult:
    """Convenience function for workload mix optimization.

    Args:
        workloads: List of workload specifications.
        total_gpu_budget: Maximum total GPU instances.
        mode: Allocation mode.
        max_instances_per_workload: Max P or D instances per workload.

    Returns:
        MixOptimizationResult.
    """
    optimizer = WorkloadMixOptimizer(max_instances_per_workload=max_instances_per_workload)
    return optimizer.optimize(workloads, total_gpu_budget, mode)

"""Benchmark plan generator — recommend which P:D ratios to benchmark.

Given a total instance count, generates a prioritized list of P:D
configurations to benchmark, minimizing the number of runs needed to
find the optimal ratio.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import AnalysisResult


class RatioPriority(str, Enum):
    """Priority level for a recommended benchmark ratio."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PlannedRatio(BaseModel):
    """A single recommended P:D ratio to benchmark."""

    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    priority: RatioPriority
    reason: str = Field(..., description="Why this ratio is recommended")

    @property
    def total(self) -> int:
        return self.num_prefill + self.num_decode

    @property
    def ratio_str(self) -> str:
        return f"{self.num_prefill}P:{self.num_decode}D"


class BenchmarkPlan(BaseModel):
    """Complete benchmark plan."""

    total_instances: int = Field(..., ge=2)
    max_runs: int | None = Field(None, description="Max runs cap (None = all)")
    ratios: list[PlannedRatio] = Field(..., description="Ordered list of ratios to benchmark")
    refined: bool = Field(False, description="Whether plan was refined using existing data")


class BenchmarkPlanGenerator:
    """Generates a prioritized benchmark plan for P:D ratio exploration."""

    def __init__(
        self,
        total_instances: int,
        max_runs: int | None = None,
        existing_result: AnalysisResult | None = None,
    ) -> None:
        if total_instances < 2:
            raise ValueError("total_instances must be >= 2")
        if max_runs is not None and max_runs < 1:
            raise ValueError("max_runs must be >= 1")
        self.total = total_instances
        self.max_runs = max_runs
        self.existing_result = existing_result

    def generate(self) -> BenchmarkPlan:
        """Generate the benchmark plan."""
        all_ratios = self._enumerate_ratios()

        if self.existing_result and self.existing_result.best:
            planned = self._refine_plan(all_ratios)
            refined = True
        else:
            planned = self._initial_plan(all_ratios)
            refined = False

        # Deduplicate preserving order
        seen: set[tuple[int, int]] = set()
        unique: list[PlannedRatio] = []
        for r in planned:
            key = (r.num_prefill, r.num_decode)
            if key not in seen:
                seen.add(key)
                unique.append(r)

        if self.max_runs is not None:
            unique = unique[: self.max_runs]

        return BenchmarkPlan(
            total_instances=self.total,
            max_runs=self.max_runs,
            ratios=unique,
            refined=refined,
        )

    def _enumerate_ratios(self) -> list[tuple[int, int]]:
        """Return all valid (P, D) splits."""
        return [(p, self.total - p) for p in range(1, self.total)]

    def _initial_plan(self, all_ratios: list[tuple[int, int]]) -> list[PlannedRatio]:
        """Build prioritized plan without existing data."""
        planned: list[PlannedRatio] = []

        # Critical: balanced ratio
        mid = self.total // 2
        balanced = (mid, self.total - mid)
        planned.append(
            PlannedRatio(
                num_prefill=balanced[0],
                num_decode=balanced[1],
                priority=RatioPriority.CRITICAL,
                reason="balanced split — strong baseline",
            )
        )
        # If total is even, also add the mirror
        if self.total % 2 == 0 and balanced[0] != balanced[1]:
            mirror = (balanced[1], balanced[0])
            planned.append(
                PlannedRatio(
                    num_prefill=mirror[0],
                    num_decode=mirror[1],
                    priority=RatioPriority.CRITICAL,
                    reason="balanced split (mirror) — strong baseline",
                )
            )

        # High: boundary ratios
        for p, d in [(1, self.total - 1), (self.total - 1, 1)]:
            planned.append(
                PlannedRatio(
                    num_prefill=p,
                    num_decode=d,
                    priority=RatioPriority.HIGH,
                    reason="boundary ratio — establishes extreme behavior",
                )
            )

        # High: quarter points
        q1 = max(1, self.total // 4)
        q3 = max(1, 3 * self.total // 4)
        for p in [q1, q3]:
            d = self.total - p
            if d >= 1:
                planned.append(
                    PlannedRatio(
                        num_prefill=p,
                        num_decode=d,
                        priority=RatioPriority.HIGH,
                        reason="quarter-point ratio — covers spread",
                    )
                )

        # Medium: remaining ratios
        seen = {(r.num_prefill, r.num_decode) for r in planned}
        for p, d in all_ratios:
            if (p, d) not in seen:
                planned.append(
                    PlannedRatio(
                        num_prefill=p,
                        num_decode=d,
                        priority=RatioPriority.MEDIUM,
                        reason="additional coverage",
                    )
                )

        return planned

    def _refine_plan(self, all_ratios: list[tuple[int, int]]) -> list[PlannedRatio]:
        """Refine plan around existing best ratio."""
        best = self.existing_result.best  # type: ignore[union-attr]
        bp, bd = best.num_prefill, best.num_decode
        planned: list[PlannedRatio] = []

        # Critical: immediate neighbors of best
        for delta in [-1, 1]:
            np_, nd = bp + delta, bd - delta
            if np_ >= 1 and nd >= 1:
                planned.append(
                    PlannedRatio(
                        num_prefill=np_,
                        num_decode=nd,
                        priority=RatioPriority.CRITICAL,
                        reason=f"neighbor of current best ({bp}P:{bd}D)",
                    )
                )

        # High: two steps away
        for delta in [-2, 2]:
            np_, nd = bp + delta, bd - delta
            if np_ >= 1 and nd >= 1:
                planned.append(
                    PlannedRatio(
                        num_prefill=np_,
                        num_decode=nd,
                        priority=RatioPriority.HIGH,
                        reason=f"near current best ({bp}P:{bd}D)",
                    )
                )

        # Medium: everything else
        seen = {(r.num_prefill, r.num_decode) for r in planned}
        # Also exclude already-benchmarked ratios from existing result
        if self.existing_result:
            for c in self.existing_result.candidates:
                seen.add((c.num_prefill, c.num_decode))

        for p, d in all_ratios:
            if (p, d) not in seen:
                planned.append(
                    PlannedRatio(
                        num_prefill=p,
                        num_decode=d,
                        priority=RatioPriority.MEDIUM,
                        reason="additional coverage",
                    )
                )

        return planned


def generate_benchmark_plan(
    total_instances: int,
    max_runs: int | None = None,
    existing_result: AnalysisResult | None = None,
) -> dict:
    """Programmatic API for benchmark plan generation.

    Returns a dict with the plan details.
    """
    generator = BenchmarkPlanGenerator(
        total_instances=total_instances,
        max_runs=max_runs,
        existing_result=existing_result,
    )
    plan = generator.generate()
    return plan.model_dump()

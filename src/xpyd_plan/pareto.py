"""Pareto frontier analysis for multi-objective P:D ratio optimization.

Identifies non-dominated P:D ratio candidates across latency, cost,
and utilization waste objectives, enabling informed trade-off decisions.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import AnalysisResult
from xpyd_plan.cost import CostAnalyzer, CostConfig


class ParetoObjective(str, Enum):
    """Objectives for Pareto analysis (all minimized)."""

    LATENCY = "latency"
    COST = "cost"
    WASTE = "waste"


class ParetoCandidate(BaseModel):
    """A P:D ratio candidate with objective values for Pareto analysis."""

    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    latency_ms: float = Field(..., ge=0, description="P95 total latency (ms)")
    hourly_cost: float | None = Field(None, ge=0, description="Hourly GPU cost")
    waste_rate: float = Field(..., ge=0, le=1, description="Utilization waste rate")
    meets_sla: bool = True
    is_dominated: bool = Field(False, description="True if dominated by another candidate")
    weighted_score: float | None = Field(None, description="Weighted score (lower is better)")

    @property
    def ratio_str(self) -> str:
        return f"{self.num_prefill}P:{self.num_decode}D"

    @property
    def total_instances(self) -> int:
        return self.num_prefill + self.num_decode


class ParetoFrontier(BaseModel):
    """Result of Pareto frontier analysis."""

    candidates: list[ParetoCandidate] = Field(default_factory=list)
    frontier: list[ParetoCandidate] = Field(
        default_factory=list, description="Non-dominated (Pareto-optimal) candidates"
    )
    dominated: list[ParetoCandidate] = Field(
        default_factory=list, description="Dominated candidates"
    )
    best_weighted: ParetoCandidate | None = Field(
        None, description="Best candidate by weighted score"
    )
    objectives_used: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)


def _dominates(
    a: ParetoCandidate,
    b: ParetoCandidate,
    objectives: list[ParetoObjective],
) -> bool:
    """Return True if candidate a dominates candidate b.

    a dominates b if a is <= b on all objectives and < b on at least one.
    All objectives are minimized.
    """
    values_a = _get_objective_values(a, objectives)
    values_b = _get_objective_values(b, objectives)

    at_least_one_better = False
    for va, vb in zip(values_a, values_b):
        if va > vb:
            return False
        if va < vb:
            at_least_one_better = True
    return at_least_one_better


def _get_objective_values(
    candidate: ParetoCandidate,
    objectives: list[ParetoObjective],
) -> list[float]:
    """Extract objective values from a candidate."""
    values: list[float] = []
    for obj in objectives:
        if obj == ParetoObjective.LATENCY:
            values.append(candidate.latency_ms)
        elif obj == ParetoObjective.COST:
            values.append(candidate.hourly_cost if candidate.hourly_cost is not None else 0.0)
        elif obj == ParetoObjective.WASTE:
            values.append(candidate.waste_rate)
    return values


class ParetoAnalyzer:
    """Analyze P:D ratio candidates using Pareto dominance."""

    def __init__(
        self,
        cost_config: CostConfig | None = None,
    ) -> None:
        self._cost_analyzer = CostAnalyzer(cost_config) if cost_config else None

    def analyze(
        self,
        analysis: AnalysisResult,
        measured_qps: float | None = None,
        objectives: list[ParetoObjective] | None = None,
        weights: dict[str, float] | None = None,
        sla_only: bool = True,
    ) -> ParetoFrontier:
        """Find the Pareto frontier across specified objectives.

        Args:
            analysis: AnalysisResult from BenchmarkAnalyzer.
            measured_qps: Measured QPS for cost calculations.
            objectives: Which objectives to use (default: latency + waste,
                        adds cost if cost_config provided).
            weights: Objective weights for scoring (default: equal).
                     Keys: 'latency', 'cost', 'waste'. Values: relative weights.
            sla_only: If True, only consider SLA-meeting candidates.

        Returns:
            ParetoFrontier with classified candidates.
        """
        if objectives is None:
            objectives = [ParetoObjective.LATENCY, ParetoObjective.WASTE]
            if self._cost_analyzer is not None:
                objectives.append(ParetoObjective.COST)

        if weights is None:
            weights = {obj.value: 1.0 for obj in objectives}

        # Build ParetoCandidate list from analysis
        candidates = self._build_candidates(analysis, measured_qps, sla_only)

        if not candidates:
            return ParetoFrontier(
                objectives_used=[o.value for o in objectives],
                weights=weights,
            )

        # Compute Pareto dominance
        frontier: list[ParetoCandidate] = []
        dominated: list[ParetoCandidate] = []

        for i, c in enumerate(candidates):
            is_dominated = False
            for j, other in enumerate(candidates):
                if i != j and _dominates(other, c, objectives):
                    is_dominated = True
                    break
            c_copy = c.model_copy(update={"is_dominated": is_dominated})
            if is_dominated:
                dominated.append(c_copy)
            else:
                frontier.append(c_copy)

        # Compute weighted scores (normalize then weight)
        all_classified = frontier + dominated
        self._compute_weighted_scores(all_classified, objectives, weights)

        # Sort frontier by weighted score
        frontier.sort(key=lambda c: c.weighted_score or float("inf"))
        dominated.sort(key=lambda c: c.weighted_score or float("inf"))

        best_weighted = frontier[0] if frontier else None

        return ParetoFrontier(
            candidates=all_classified,
            frontier=frontier,
            dominated=dominated,
            best_weighted=best_weighted,
            objectives_used=[o.value for o in objectives],
            weights=weights,
        )

    def _build_candidates(
        self,
        analysis: AnalysisResult,
        measured_qps: float | None,
        sla_only: bool,
    ) -> list[ParetoCandidate]:
        """Convert RatioCandidates to ParetoCandidates."""
        candidates: list[ParetoCandidate] = []
        for rc in analysis.candidates:
            if sla_only and not rc.meets_sla:
                continue

            latency = 0.0
            if rc.sla_check is not None:
                latency = rc.sla_check.total_latency_p95_ms

            hourly_cost = None
            if self._cost_analyzer is not None:
                cost_result = self._cost_analyzer.compute_cost(rc, measured_qps)
                hourly_cost = cost_result.hourly_cost

            candidates.append(
                ParetoCandidate(
                    num_prefill=rc.num_prefill,
                    num_decode=rc.num_decode,
                    latency_ms=latency,
                    hourly_cost=hourly_cost,
                    waste_rate=rc.waste_rate,
                    meets_sla=rc.meets_sla,
                )
            )
        return candidates

    def _compute_weighted_scores(
        self,
        candidates: list[ParetoCandidate],
        objectives: list[ParetoObjective],
        weights: dict[str, float],
    ) -> None:
        """Normalize objectives to [0,1] and compute weighted scores in-place."""
        if not candidates:
            return

        # Extract values per objective
        obj_values: dict[ParetoObjective, list[float]] = {}
        for obj in objectives:
            obj_values[obj] = _get_all_values(candidates, obj)

        # Compute min/max for normalization
        for c in candidates:
            score = 0.0
            total_weight = sum(weights.get(obj.value, 1.0) for obj in objectives)
            for obj in objectives:
                vals = obj_values[obj]
                min_v, max_v = min(vals), max(vals)
                raw = _get_objective_values(c, [obj])[0]
                normalized = (raw - min_v) / (max_v - min_v) if max_v > min_v else 0.0
                w = weights.get(obj.value, 1.0) / total_weight
                score += normalized * w
            c.weighted_score = round(score, 6)


def _get_all_values(
    candidates: list[ParetoCandidate], obj: ParetoObjective
) -> list[float]:
    """Get all values for a single objective."""
    return [_get_objective_values(c, [obj])[0] for c in candidates]


def find_pareto_frontier(
    analysis: AnalysisResult,
    measured_qps: float | None = None,
    cost_config: CostConfig | None = None,
    objectives: list[str] | None = None,
    weights: dict[str, float] | None = None,
    sla_only: bool = True,
) -> ParetoFrontier:
    """Programmatic API: find the Pareto frontier for P:D ratio analysis.

    Args:
        analysis: AnalysisResult from BenchmarkAnalyzer.
        measured_qps: Measured QPS for cost calculations.
        cost_config: CostConfig for cost objective. If None, cost is excluded.
        objectives: List of objective names ('latency', 'cost', 'waste').
        weights: Objective weights (default: equal).
        sla_only: Only consider SLA-meeting candidates.

    Returns:
        ParetoFrontier with classified candidates and recommendations.
    """
    analyzer = ParetoAnalyzer(cost_config=cost_config)
    parsed_objectives = None
    if objectives is not None:
        parsed_objectives = [ParetoObjective(o) for o in objectives]
    return analyzer.analyze(
        analysis,
        measured_qps=measured_qps,
        objectives=parsed_objectives,
        weights=weights,
        sla_only=sla_only,
    )

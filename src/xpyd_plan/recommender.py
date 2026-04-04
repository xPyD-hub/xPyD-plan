"""Recommendation engine for P:D ratio optimization.

Combines SLA compliance, cost analysis, Pareto frontier, and utilization
data to generate ranked, actionable recommendations.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import AnalysisResult, RatioCandidate
from xpyd_plan.cost import CostAnalyzer, CostConfig


class RecommendationPriority(str, Enum):
    """Priority level for a recommendation."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionCategory(str, Enum):
    """Action category for a recommendation."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REBALANCE = "rebalance"
    INVESTIGATE = "investigate"
    NO_ACTION = "no_action"


# Priority ordering for sorting (lower = more urgent)
_PRIORITY_ORDER = {
    RecommendationPriority.CRITICAL: 0,
    RecommendationPriority.HIGH: 1,
    RecommendationPriority.MEDIUM: 2,
    RecommendationPriority.LOW: 3,
}


class Recommendation(BaseModel):
    """A single actionable recommendation."""

    priority: RecommendationPriority
    action: ActionCategory
    title: str
    detail: str
    current_ratio: str | None = None
    suggested_ratio: str | None = None

    @property
    def priority_order(self) -> int:
        return _PRIORITY_ORDER[self.priority]


class RecommendationReport(BaseModel):
    """Complete recommendation report."""

    recommendations: list[Recommendation] = Field(default_factory=list)
    total_instances: int = 0
    current_ratio: str | None = None
    optimal_ratio: str | None = None
    analysis_summary: str = ""


class RecommendationEngine:
    """Generate actionable recommendations from analysis results.

    Examines SLA compliance, utilization waste, cost data, and Pareto
    frontier analysis to produce ranked recommendations.

    Args:
        cost_config: Optional cost configuration for cost-aware recommendations.
        waste_threshold: Waste rate above which REBALANCE is recommended (default 0.3).
        critical_waste_threshold: Waste rate above which recommendation is HIGH (default 0.5).
    """

    def __init__(
        self,
        cost_config: CostConfig | None = None,
        waste_threshold: float = 0.3,
        critical_waste_threshold: float = 0.5,
    ) -> None:
        self.cost_config = cost_config
        self.waste_threshold = waste_threshold
        self.critical_waste_threshold = critical_waste_threshold

    def analyze(
        self,
        analysis: AnalysisResult,
        measured_qps: float | None = None,
    ) -> RecommendationReport:
        """Generate recommendations from an analysis result.

        Args:
            analysis: AnalysisResult from BenchmarkAnalyzer.
            measured_qps: Measured QPS for cost calculations.

        Returns:
            RecommendationReport with ranked recommendations.
        """
        recommendations: list[Recommendation] = []
        current = analysis.current_sla_check
        current_ratio_str: str | None = None

        optimal_ratio_str: str | None = None
        if analysis.best:
            optimal_ratio_str = analysis.best.ratio_str

        # 1. SLA compliance checks
        recommendations.extend(self._check_sla(analysis, current_ratio_str))

        # 2. Waste / utilization checks
        recommendations.extend(
            self._check_waste(analysis, current_ratio_str)
        )

        # 3. Cost checks
        if self.cost_config and measured_qps:
            recommendations.extend(
                self._check_cost(analysis, measured_qps, current_ratio_str)
            )

        # 4. If no issues found, produce NO_ACTION
        if not recommendations:
            summary = "Current configuration is optimal."
            if current and current.meets_all and analysis.best:
                summary = (
                    f"Current configuration meets SLA and {analysis.best.ratio_str} "
                    "is already optimal."
                )
            recommendations.append(
                Recommendation(
                    priority=RecommendationPriority.LOW,
                    action=ActionCategory.NO_ACTION,
                    title="No changes needed",
                    detail=summary,
                    current_ratio=current_ratio_str,
                )
            )

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority_order)

        # Build summary
        sla_meeting = sum(1 for c in analysis.candidates if c.meets_sla)
        total_candidates = len(analysis.candidates)
        summary = (
            f"{sla_meeting}/{total_candidates} ratios meet SLA. "
            f"Optimal: {optimal_ratio_str or 'none'}."
        )

        return RecommendationReport(
            recommendations=recommendations,
            total_instances=analysis.total_instances,
            current_ratio=current_ratio_str,
            optimal_ratio=optimal_ratio_str,
            analysis_summary=summary,
        )

    def _check_sla(
        self,
        analysis: AnalysisResult,
        current_ratio: str | None,
    ) -> list[Recommendation]:
        """Check SLA compliance and generate recommendations."""
        recs: list[Recommendation] = []
        current = analysis.current_sla_check

        # No candidates meet SLA at all
        if not any(c.meets_sla for c in analysis.candidates):
            recs.append(
                Recommendation(
                    priority=RecommendationPriority.CRITICAL,
                    action=ActionCategory.SCALE_UP,
                    title="No P:D ratio meets SLA",
                    detail=(
                        f"None of the {len(analysis.candidates)} evaluated ratios "
                        "satisfy SLA constraints. Consider adding more instances "
                        "or relaxing SLA targets."
                    ),
                    current_ratio=current_ratio,
                )
            )
            return recs

        # Current config fails SLA but optimal exists
        if current and not current.meets_all and analysis.best:
            recs.append(
                Recommendation(
                    priority=RecommendationPriority.CRITICAL,
                    action=ActionCategory.REBALANCE,
                    title="Current ratio fails SLA",
                    detail=(
                        f"Current configuration does not meet SLA. "
                        f"Switch to {analysis.best.ratio_str} which meets SLA "
                        f"with {analysis.best.waste_rate:.1%} waste."
                    ),
                    current_ratio=current_ratio,
                    suggested_ratio=analysis.best.ratio_str,
                )
            )

        return recs

    def _check_waste(
        self,
        analysis: AnalysisResult,
        current_ratio: str | None,
    ) -> list[Recommendation]:
        """Check utilization waste and generate recommendations."""
        recs: list[Recommendation] = []

        if not analysis.best:
            return recs

        best = analysis.best

        # Current config has high waste but better option exists
        if analysis.current_config:
            current_waste = analysis.current_config.waste_rate
            if current_waste >= self.critical_waste_threshold and best.waste_rate < current_waste:
                recs.append(
                    Recommendation(
                        priority=RecommendationPriority.HIGH,
                        action=ActionCategory.REBALANCE,
                        title="High utilization waste",
                        detail=(
                            f"Current waste rate is {current_waste:.1%}, "
                            f"rebalance to {best.ratio_str} for {best.waste_rate:.1%} waste."
                        ),
                        current_ratio=current_ratio,
                        suggested_ratio=best.ratio_str,
                    )
                )
            elif current_waste >= self.waste_threshold and best.waste_rate < current_waste:
                recs.append(
                    Recommendation(
                        priority=RecommendationPriority.MEDIUM,
                        action=ActionCategory.REBALANCE,
                        title="Moderate utilization waste",
                        detail=(
                            f"Current waste rate is {current_waste:.1%}, "
                            f"rebalance to {best.ratio_str} for {best.waste_rate:.1%} waste."
                        ),
                        current_ratio=current_ratio,
                        suggested_ratio=best.ratio_str,
                    )
                )

        # Check if any candidate has very low utilization (possible scale-down)
        sla_candidates = [c for c in analysis.candidates if c.meets_sla]
        if sla_candidates:
            min_waste = min(c.waste_rate for c in sla_candidates)
            # All candidates have very low waste — might be over-provisioned
            if min_waste < 0.05 and len(sla_candidates) == len(analysis.candidates):
                recs.append(
                    Recommendation(
                        priority=RecommendationPriority.LOW,
                        action=ActionCategory.SCALE_DOWN,
                        title="Potential over-provisioning",
                        detail=(
                            f"All {len(sla_candidates)} ratios meet SLA with low waste "
                            f"(min {min_waste:.1%}). Consider reducing total instances."
                        ),
                        current_ratio=current_ratio,
                    )
                )

        return recs

    def _check_cost(
        self,
        analysis: AnalysisResult,
        measured_qps: float,
        current_ratio: str | None,
    ) -> list[Recommendation]:
        """Check cost efficiency and generate recommendations."""
        recs: list[Recommendation] = []

        if not self.cost_config or not analysis.best:
            return recs

        cost_analyzer = CostAnalyzer(self.cost_config)

        # Find cheapest SLA-meeting candidate
        sla_candidates = [c for c in analysis.candidates if c.meets_sla]
        if not sla_candidates:
            return recs

        def _cost(c: RatioCandidate) -> float:
            return cost_analyzer.compute_cost(c, measured_qps).hourly_cost

        cheapest = min(sla_candidates, key=_cost)
        cheapest_cost = _cost(cheapest)

        # If current config exists and is more expensive
        if analysis.current_config:
            current_total = analysis.total_instances
            current_cost = current_total * self.cost_config.gpu_hourly_rate
            savings = current_cost - cheapest_cost

            if savings > 0 and savings / current_cost > 0.1:
                recs.append(
                    Recommendation(
                        priority=RecommendationPriority.MEDIUM,
                        action=ActionCategory.REBALANCE,
                        title="Cost optimization available",
                        detail=(
                            f"Switch to {cheapest.ratio_str} to save "
                            f"{savings:.2f} {self.cost_config.currency}/hr "
                            f"({savings / current_cost:.0%} reduction)."
                        ),
                        current_ratio=current_ratio,
                        suggested_ratio=cheapest.ratio_str,
                    )
                )

        return recs


def get_recommendations(
    analysis: AnalysisResult,
    measured_qps: float | None = None,
    cost_config: CostConfig | None = None,
    waste_threshold: float = 0.3,
    critical_waste_threshold: float = 0.5,
) -> RecommendationReport:
    """Programmatic API: generate recommendations from analysis results.

    Args:
        analysis: AnalysisResult from BenchmarkAnalyzer.
        measured_qps: Measured QPS for cost calculations.
        cost_config: Optional CostConfig for cost-aware recommendations.
        waste_threshold: Waste rate threshold for MEDIUM recommendations.
        critical_waste_threshold: Waste rate threshold for HIGH recommendations.

    Returns:
        RecommendationReport with ranked recommendations.
    """
    engine = RecommendationEngine(
        cost_config=cost_config,
        waste_threshold=waste_threshold,
        critical_waste_threshold=critical_waste_threshold,
    )
    return engine.analyze(analysis, measured_qps=measured_qps)

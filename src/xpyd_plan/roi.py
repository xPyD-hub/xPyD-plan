"""Cost projection and ROI calculator.

Given current and optimal P:D ratio costs, project savings over time
and calculate break-even points for migration investments.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.cost import CostAnalyzer, CostConfig, CostResult
from xpyd_plan.models import SLAConfig


class CostProjection(BaseModel):
    """Cost projection for a specific P:D configuration over time."""

    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    total_instances: int
    hourly_cost: float = Field(..., ge=0)
    daily_cost: float = Field(..., ge=0)
    monthly_cost: float = Field(..., ge=0, description="30-day cost")
    yearly_cost: float = Field(..., ge=0, description="365-day cost")
    cost_per_request: float | None = None
    currency: str = "USD"
    meets_sla: bool = True

    @property
    def ratio_str(self) -> str:
        return f"{self.num_prefill}P:{self.num_decode}D"


class SavingsEstimate(BaseModel):
    """Estimated savings from migrating to a cheaper configuration."""

    current_ratio: str = Field(..., description="Current P:D ratio string")
    optimal_ratio: str = Field(..., description="Optimal P:D ratio string")
    hourly_savings: float = Field(..., description="Hourly cost savings")
    daily_savings: float = Field(..., description="Daily cost savings")
    monthly_savings: float = Field(..., description="Monthly (30-day) savings")
    yearly_savings: float = Field(..., description="Yearly (365-day) savings")
    savings_percent: float = Field(
        ..., description="Savings as percentage of current cost"
    )
    migration_cost: float = Field(0.0, ge=0, description="One-time migration cost")
    break_even_hours: float | None = Field(
        None,
        description="Hours until migration cost is recovered (None if no savings)",
    )
    currency: str = "USD"


class ROIReport(BaseModel):
    """Complete ROI analysis report."""

    current_projection: CostProjection
    optimal_projection: CostProjection
    savings: SavingsEstimate
    all_projections: list[CostProjection] = Field(default_factory=list)
    currency: str = "USD"


class ROICalculator:
    """Calculate cost projections and ROI for P:D ratio configurations."""

    def __init__(
        self,
        cost_config: CostConfig,
        sla: SLAConfig,
        *,
        migration_cost: float = 0.0,
    ) -> None:
        self._cost_config = cost_config
        self._sla = sla
        self._migration_cost = migration_cost
        self._cost_analyzer = CostAnalyzer(cost_config)

    def _project(self, cost_result: CostResult) -> CostProjection:
        """Create a time-based cost projection from a CostResult."""
        hourly = cost_result.hourly_cost
        return CostProjection(
            num_prefill=cost_result.num_prefill,
            num_decode=cost_result.num_decode,
            total_instances=cost_result.total_instances,
            hourly_cost=hourly,
            daily_cost=hourly * 24,
            monthly_cost=hourly * 24 * 30,
            yearly_cost=hourly * 24 * 365,
            cost_per_request=cost_result.cost_per_request,
            currency=cost_result.currency,
            meets_sla=cost_result.meets_sla,
        )

    def _estimate_savings(
        self,
        current: CostProjection,
        optimal: CostProjection,
    ) -> SavingsEstimate:
        """Compute savings and break-even from current to optimal."""
        hourly_savings = current.hourly_cost - optimal.hourly_cost
        savings_pct = (
            (hourly_savings / current.hourly_cost * 100)
            if current.hourly_cost > 0
            else 0.0
        )

        break_even = None
        if hourly_savings > 0 and self._migration_cost > 0:
            break_even = self._migration_cost / hourly_savings
        elif hourly_savings <= 0 and self._migration_cost > 0:
            break_even = None  # Never recovers

        return SavingsEstimate(
            current_ratio=current.ratio_str,
            optimal_ratio=optimal.ratio_str,
            hourly_savings=hourly_savings,
            daily_savings=hourly_savings * 24,
            monthly_savings=hourly_savings * 24 * 30,
            yearly_savings=hourly_savings * 24 * 365,
            savings_percent=savings_pct,
            migration_cost=self._migration_cost,
            break_even_hours=break_even,
            currency=current.currency,
        )

    def calculate(
        self,
        data: BenchmarkData,
    ) -> ROIReport:
        """Run full ROI analysis on benchmark data.

        Finds the current configuration's cost and the cost-optimal
        SLA-meeting configuration, then computes savings projections.

        Args:
            data: Benchmark data with measured results.

        Returns:
            ROIReport with projections, savings, and break-even analysis.
        """
        analyzer = BenchmarkAnalyzer()
        analyzer._data = data
        analysis = analyzer.find_optimal_ratio(
            data.metadata.total_instances, self._sla
        )
        measured_qps = data.metadata.measured_qps

        # Current config cost
        from xpyd_plan.benchmark_models import RatioCandidate

        current_candidate = None
        for c in analysis.candidates:
            if (
                c.num_prefill == data.metadata.num_prefill_instances
                and c.num_decode == data.metadata.num_decode_instances
            ):
                current_candidate = c
                break

        if current_candidate is None:
            # Fallback: create from metadata
            current_candidate = RatioCandidate(
                num_prefill=data.metadata.num_prefill_instances,
                num_decode=data.metadata.num_decode_instances,
                prefill_utilization=0.0,
                decode_utilization=0.0,
                waste_rate=1.0,
                meets_sla=False,
            )

        current_cost = self._cost_analyzer.compute_cost(current_candidate, measured_qps)
        current_proj = self._project(current_cost)

        # Cost-optimal config
        comparison = self._cost_analyzer.compare(analysis, measured_qps)

        if comparison.cost_optimal is not None:
            optimal_cost = comparison.cost_optimal
        else:
            # No SLA-meeting config; use current as "optimal"
            optimal_cost = current_cost

        optimal_proj = self._project(optimal_cost)

        # All projections
        all_costs = self._cost_analyzer.compute_all_costs(analysis, measured_qps)
        all_projections = [self._project(c) for c in all_costs]

        savings = self._estimate_savings(current_proj, optimal_proj)

        return ROIReport(
            current_projection=current_proj,
            optimal_projection=optimal_proj,
            savings=savings,
            all_projections=all_projections,
            currency=self._cost_config.currency,
        )


def calculate_roi(
    data: BenchmarkData,
    cost_config: CostConfig,
    sla: SLAConfig,
    *,
    migration_cost: float = 0.0,
) -> ROIReport:
    """Programmatic API for ROI calculation.

    Args:
        data: Benchmark data with measured results.
        cost_config: GPU cost configuration.
        sla: SLA configuration for compliance checks.
        migration_cost: One-time cost of migrating configurations.

    Returns:
        ROIReport with projections, savings, and break-even analysis.
    """
    calculator = ROICalculator(cost_config, sla, migration_cost=migration_cost)
    return calculator.calculate(data)

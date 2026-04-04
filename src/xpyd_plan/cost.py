"""Cost-aware optimization for P:D ratio analysis.

Adds GPU cost models to enable cost-per-request and total hourly cost
calculations, budget constraints, and cost-optimal vs SLA-optimal comparison.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import AnalysisResult, RatioCandidate


class CostConfig(BaseModel):
    """GPU cost configuration."""

    gpu_hourly_rate: float = Field(..., gt=0, description="Hourly cost per GPU instance (USD)")
    currency: str = Field("USD", description="Currency code")

    @classmethod
    def from_yaml(cls, path: str) -> CostConfig:
        """Load cost config from a YAML file."""
        from pathlib import Path

        import yaml

        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)


class CostResult(BaseModel):
    """Cost metrics for a specific P:D ratio."""

    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    total_instances: int
    hourly_cost: float = Field(..., ge=0, description="Total hourly cost")
    cost_per_request: float | None = Field(
        None, ge=0, description="Cost per request (hourly cost / requests per hour)"
    )
    currency: str = "USD"
    meets_sla: bool = True

    @property
    def ratio_str(self) -> str:
        return f"{self.num_prefill}P:{self.num_decode}D"


class CostComparisonResult(BaseModel):
    """Comparison between SLA-optimal and cost-optimal ratios."""

    sla_optimal: CostResult | None = None
    cost_optimal: CostResult | None = None
    all_costs: list[CostResult] = Field(default_factory=list)
    budget_ceiling: float | None = None


class CostAnalyzer:
    """Analyze costs for different P:D ratio configurations."""

    def __init__(self, cost_config: CostConfig) -> None:
        self._config = cost_config

    @property
    def config(self) -> CostConfig:
        return self._config

    def compute_cost(
        self,
        candidate: RatioCandidate,
        measured_qps: float | None = None,
    ) -> CostResult:
        """Compute cost metrics for a single P:D ratio candidate.

        Args:
            candidate: A RatioCandidate from the analyzer.
            measured_qps: If provided, used to compute cost_per_request.
        """
        total = candidate.num_prefill + candidate.num_decode
        hourly_cost = total * self._config.gpu_hourly_rate

        cost_per_request = None
        if measured_qps is not None and measured_qps > 0:
            requests_per_hour = measured_qps * 3600
            cost_per_request = hourly_cost / requests_per_hour

        return CostResult(
            num_prefill=candidate.num_prefill,
            num_decode=candidate.num_decode,
            total_instances=total,
            hourly_cost=hourly_cost,
            cost_per_request=cost_per_request,
            currency=self._config.currency,
            meets_sla=candidate.meets_sla,
        )

    def compute_all_costs(
        self,
        analysis: AnalysisResult,
        measured_qps: float | None = None,
        budget_ceiling: float | None = None,
    ) -> list[CostResult]:
        """Compute costs for all candidates, optionally filtering by budget.

        Args:
            analysis: The AnalysisResult from BenchmarkAnalyzer.
            measured_qps: If provided, used to compute cost_per_request.
            budget_ceiling: If set, exclude ratios with hourly cost above this.

        Returns:
            List of CostResult sorted by hourly_cost ascending.
        """
        results = []
        for candidate in analysis.candidates:
            cost = self.compute_cost(candidate, measured_qps)
            if budget_ceiling is not None and cost.hourly_cost > budget_ceiling:
                continue
            results.append(cost)
        return sorted(results, key=lambda c: c.hourly_cost)

    def find_cost_optimal(
        self,
        analysis: AnalysisResult,
        measured_qps: float | None = None,
        budget_ceiling: float | None = None,
    ) -> CostResult | None:
        """Find the cheapest P:D ratio that meets SLA.

        Args:
            analysis: The AnalysisResult from BenchmarkAnalyzer.
            measured_qps: If provided, used to compute cost_per_request.
            budget_ceiling: If set, exclude ratios with hourly cost above this.

        Returns:
            The cheapest CostResult meeting SLA, or None if none qualifies.
        """
        all_costs = self.compute_all_costs(analysis, measured_qps, budget_ceiling)
        sla_meeting = [c for c in all_costs if c.meets_sla]
        if not sla_meeting:
            return None
        return sla_meeting[0]  # Already sorted by hourly_cost

    def compare(
        self,
        analysis: AnalysisResult,
        measured_qps: float | None = None,
        budget_ceiling: float | None = None,
    ) -> CostComparisonResult:
        """Compare SLA-optimal vs cost-optimal ratio.

        SLA-optimal: the ratio with minimum waste that meets SLA (from analyzer).
        Cost-optimal: the cheapest ratio that meets SLA.
        """
        all_costs = self.compute_all_costs(analysis, measured_qps, budget_ceiling)

        # Cost-optimal: cheapest that meets SLA
        sla_meeting = [c for c in all_costs if c.meets_sla]
        cost_optimal = sla_meeting[0] if sla_meeting else None

        # SLA-optimal: compute cost for the analyzer's best pick
        sla_optimal = None
        if analysis.best is not None:
            sla_optimal = self.compute_cost(analysis.best, measured_qps)

        return CostComparisonResult(
            sla_optimal=sla_optimal,
            cost_optimal=cost_optimal,
            all_costs=all_costs,
            budget_ceiling=budget_ceiling,
        )

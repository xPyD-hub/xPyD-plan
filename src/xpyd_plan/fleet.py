"""Fleet sizing calculator for multi-GPU-type optimization.

Given a target QPS, multiple GPU types with benchmark data and costs,
find the cheapest fleet composition meeting SLA and QPS targets.
"""

from __future__ import annotations

import itertools
import math

from pydantic import BaseModel, Field

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.benchmark_models import AnalysisResult, BenchmarkData
from xpyd_plan.models import SLAConfig


class GPUTypeConfig(BaseModel):
    """Configuration for a single GPU type in the fleet."""

    name: str = Field(..., description="GPU type name (e.g. 'A100-80G')")
    benchmark: BenchmarkData = Field(..., description="Benchmark data for this GPU type")
    hourly_rate: float = Field(..., gt=0, description="Hourly cost per instance")
    max_instances: int = Field(64, ge=1, description="Max instances of this GPU type")
    currency: str = Field("USD", description="Currency code")


class FleetAllocation(BaseModel):
    """Allocation for a single GPU type in the fleet."""

    gpu_type: str
    num_prefill: int = Field(..., ge=0)
    num_decode: int = Field(..., ge=0)
    total_instances: int = Field(..., ge=0)
    estimated_qps: float = Field(..., ge=0, description="QPS this allocation handles")
    hourly_cost: float = Field(..., ge=0)
    meets_sla: bool
    ratio_str: str
    currency: str = "USD"


class FleetOption(BaseModel):
    """A complete fleet configuration option."""

    allocations: list[FleetAllocation] = Field(default_factory=list)
    total_instances: int = Field(..., ge=0)
    total_hourly_cost: float = Field(..., ge=0)
    total_qps: float = Field(..., ge=0, description="Total QPS across all GPU types")
    meets_target_qps: bool
    meets_sla: bool
    currency: str = "USD"

    @property
    def cost_per_qps(self) -> float | None:
        """Cost per unit of QPS."""
        if self.total_qps <= 0:
            return None
        return self.total_hourly_cost / self.total_qps


class FleetReport(BaseModel):
    """Complete fleet sizing report."""

    target_qps: float
    budget_ceiling: float | None = None
    gpu_types: list[str] = Field(default_factory=list)
    best: FleetOption | None = None
    options: list[FleetOption] = Field(default_factory=list)
    currency: str = "USD"


class FleetCalculator:
    """Calculate optimal fleet composition across multiple GPU types.

    For each GPU type, analyzes benchmark data to find the best P:D ratio
    and estimates QPS capacity. Then combines GPU types to find the cheapest
    fleet meeting the target QPS and SLA.
    """

    def __init__(
        self,
        gpu_configs: list[GPUTypeConfig],
        sla: SLAConfig,
        *,
        budget_ceiling: float | None = None,
        max_options: int = 20,
    ) -> None:
        self._gpu_configs = gpu_configs
        self._sla = sla
        self._budget_ceiling = budget_ceiling
        self._max_options = max_options
        # Pre-compute per-GPU-type analysis
        self._gpu_analyses: list[_GPUAnalysis] = []
        for cfg in gpu_configs:
            self._gpu_analyses.append(self._analyze_gpu(cfg))

    def _analyze_gpu(self, cfg: GPUTypeConfig) -> _GPUAnalysis:
        """Analyze a single GPU type: find best ratio and QPS per instance."""
        analyzer = BenchmarkAnalyzer()
        analyzer._data = cfg.benchmark
        total = cfg.benchmark.metadata.total_instances
        analysis = analyzer.find_optimal_ratio(total, self._sla)

        # Estimate QPS per instance from benchmark metadata
        measured_qps = cfg.benchmark.metadata.measured_qps
        qps_per_instance = measured_qps / total if total > 0 else 0

        best_ratio = None
        if analysis.best is not None:
            best_ratio = (analysis.best.num_prefill, analysis.best.num_decode)

        return _GPUAnalysis(
            config=cfg,
            analysis=analysis,
            qps_per_instance=qps_per_instance,
            best_ratio=best_ratio,
            meets_sla=analysis.best is not None and analysis.best.meets_sla,
        )

    def calculate(self, target_qps: float) -> FleetReport:
        """Find fleet options meeting the target QPS.

        Enumerates combinations of GPU types and instance counts,
        returns options sorted by total hourly cost.
        """
        gpu_names = [cfg.name for cfg in self._gpu_configs]
        options: list[FleetOption] = []

        # For each GPU type, determine possible instance counts
        # Range: 0 to min(max_instances, ceil(target_qps / qps_per_instance) * 2)
        instance_ranges: list[list[int]] = []
        for ga in self._gpu_analyses:
            if ga.qps_per_instance <= 0 or not ga.meets_sla:
                instance_ranges.append([0])
                continue
            max_needed = math.ceil(target_qps / ga.qps_per_instance) + 2
            max_allowed = min(ga.config.max_instances, max_needed)
            instance_ranges.append(list(range(0, max_allowed + 1)))

        # Enumerate combinations (use step size to limit search space)
        stepped_ranges: list[list[int]] = []
        for r in instance_ranges:
            if len(r) <= 20:
                stepped_ranges.append(r)
            else:
                # Sample: 0, then step through
                step = max(1, len(r) // 15)
                sampled = list(range(0, len(r), step))
                if (len(r) - 1) not in sampled:
                    sampled.append(len(r) - 1)
                stepped_ranges.append([r[i] for i in sampled])

        for combo in itertools.product(*stepped_ranges):
            total_count = sum(combo)
            if total_count == 0:
                continue

            allocations: list[FleetAllocation] = []
            total_qps = 0.0
            total_cost = 0.0
            all_meet_sla = True

            for i, count in enumerate(combo):
                if count == 0:
                    continue
                ga = self._gpu_analyses[i]
                cfg = ga.config

                # Determine P:D split for this count
                if ga.best_ratio is not None and count >= 2:
                    p_ratio = ga.best_ratio[0] / (ga.best_ratio[0] + ga.best_ratio[1])
                    num_p = max(1, round(count * p_ratio))
                    num_d = max(1, count - num_p)
                    # Adjust if over
                    if num_p + num_d > count:
                        if num_p > num_d:
                            num_p = count - num_d
                        else:
                            num_d = count - num_p
                    ratio_str = f"{num_p}P:{num_d}D"
                    meets = ga.meets_sla
                elif count == 1:
                    num_p, num_d = 1, 0
                    ratio_str = "1P:0D"
                    meets = False
                    all_meet_sla = False
                else:
                    num_p, num_d = 1, count - 1
                    ratio_str = f"{num_p}P:{num_d}D"
                    meets = False
                    all_meet_sla = False

                est_qps = ga.qps_per_instance * count
                hourly = cfg.hourly_rate * count

                if not meets:
                    all_meet_sla = False

                allocations.append(FleetAllocation(
                    gpu_type=cfg.name,
                    num_prefill=num_p,
                    num_decode=num_d,
                    total_instances=count,
                    estimated_qps=est_qps,
                    hourly_cost=hourly,
                    meets_sla=meets,
                    ratio_str=ratio_str,
                    currency=cfg.currency,
                ))
                total_qps += est_qps
                total_cost += hourly

            meets_target = total_qps >= target_qps
            if not meets_target:
                continue

            if self._budget_ceiling is not None and total_cost > self._budget_ceiling:
                continue

            option = FleetOption(
                allocations=allocations,
                total_instances=total_count,
                total_hourly_cost=round(total_cost, 4),
                total_qps=round(total_qps, 2),
                meets_target_qps=meets_target,
                meets_sla=all_meet_sla,
                currency=self._gpu_configs[0].currency if self._gpu_configs else "USD",
            )
            options.append(option)

        # Sort by cost, prefer SLA-meeting options
        options.sort(key=lambda o: (not o.meets_sla, o.total_hourly_cost))

        # Limit output
        options = options[: self._max_options]

        best = options[0] if options else None

        return FleetReport(
            target_qps=target_qps,
            budget_ceiling=self._budget_ceiling,
            gpu_types=gpu_names,
            best=best,
            options=options,
            currency=self._gpu_configs[0].currency if self._gpu_configs else "USD",
        )


class _GPUAnalysis:
    """Internal: per-GPU-type analysis results."""

    def __init__(
        self,
        config: GPUTypeConfig,
        analysis: AnalysisResult,
        qps_per_instance: float,
        best_ratio: tuple[int, int] | None,
        meets_sla: bool,
    ) -> None:
        self.config = config
        self.analysis = analysis
        self.qps_per_instance = qps_per_instance
        self.best_ratio = best_ratio
        self.meets_sla = meets_sla


def calculate_fleet(
    gpu_configs: list[GPUTypeConfig],
    target_qps: float,
    sla: SLAConfig,
    *,
    budget_ceiling: float | None = None,
    max_options: int = 20,
) -> FleetReport:
    """Programmatic API for fleet sizing.

    Args:
        gpu_configs: List of GPU type configurations with benchmark data.
        target_qps: Target QPS to achieve.
        sla: SLA configuration for compliance checks.
        budget_ceiling: Optional max hourly cost.
        max_options: Max fleet options to return.

    Returns:
        FleetReport with ranked fleet options.
    """
    calculator = FleetCalculator(
        gpu_configs, sla, budget_ceiling=budget_ceiling, max_options=max_options
    )
    return calculator.calculate(target_qps)

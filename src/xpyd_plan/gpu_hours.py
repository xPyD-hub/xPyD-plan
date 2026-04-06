"""GPU Hour Calculator — estimate GPU hours and costs from traffic profiles."""

from __future__ import annotations

import math
from typing import List

from pydantic import BaseModel, Field, field_validator

from xpyd_plan.benchmark_models import BenchmarkData


class HourlyTraffic(BaseModel):
    """Traffic specification for a single hour."""

    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    qps: float = Field(..., ge=0, description="Expected QPS during this hour")


class TrafficProfile(BaseModel):
    """24-hour traffic profile."""

    hours: List[HourlyTraffic] = Field(
        ..., min_length=1, max_length=24, description="Hourly traffic specs"
    )
    name: str = Field(default="default", description="Profile name")

    @field_validator("hours")
    @classmethod
    def validate_unique_hours(cls, v: list[HourlyTraffic]) -> list[HourlyTraffic]:
        """Ensure no duplicate hours."""
        seen = set()
        for ht in v:
            if ht.hour in seen:
                raise ValueError(f"Duplicate hour: {ht.hour}")
            seen.add(ht.hour)
        return sorted(v, key=lambda x: x.hour)


class HourBreakdown(BaseModel):
    """Per-hour resource and cost breakdown."""

    hour: int = Field(..., description="Hour of day")
    qps: float = Field(..., description="Traffic QPS")
    required_instances: int = Field(..., description="Instances needed")
    gpu_hours: float = Field(..., description="GPU hours consumed")
    cost: float = Field(..., description="Cost for this hour")


class ScalingSavings(BaseModel):
    """Savings from auto-scaling vs fixed provisioning."""

    fixed_daily_gpu_hours: float = Field(..., description="GPU hours with fixed instances")
    dynamic_daily_gpu_hours: float = Field(
        ..., description="GPU hours with auto-scaling"
    )
    saved_gpu_hours: float = Field(..., description="GPU hours saved per day")
    savings_percent: float = Field(..., description="Percentage savings")
    fixed_daily_cost: float = Field(..., description="Daily cost with fixed instances")
    dynamic_daily_cost: float = Field(
        ..., description="Daily cost with auto-scaling"
    )
    saved_cost: float = Field(..., description="Cost saved per day")


class GPUHourReport(BaseModel):
    """Complete GPU hour calculation report."""

    profile_name: str = Field(..., description="Traffic profile name")
    gpu_cost_per_hour: float = Field(..., description="GPU hourly rate")
    currency: str = Field(default="USD", description="Currency")
    peak_qps: float = Field(..., description="Peak traffic QPS")
    peak_instances: int = Field(..., description="Instances needed at peak")
    off_peak_qps: float = Field(..., description="Minimum traffic QPS")
    off_peak_instances: int = Field(..., description="Instances needed at off-peak")
    daily_gpu_hours: float = Field(..., description="Total GPU hours per day")
    monthly_gpu_hours: float = Field(..., description="Total GPU hours per month (30d)")
    daily_cost: float = Field(..., description="Total cost per day")
    monthly_cost: float = Field(..., description="Total cost per month (30d)")
    avg_utilization: float = Field(
        ..., description="Average utilization (dynamic/peak)"
    )
    hourly_breakdown: List[HourBreakdown] = Field(
        ..., description="Per-hour breakdown"
    )
    scaling_savings: ScalingSavings = Field(
        ..., description="Auto-scaling savings analysis"
    )
    qps_per_instance: float = Field(
        ..., description="Estimated QPS capacity per instance"
    )


class GPUHourCalculator:
    """Calculate GPU hours and costs from benchmark data and traffic profiles."""

    def __init__(self, data: BenchmarkData) -> None:
        self._data = data
        self._qps_per_instance = self._estimate_qps_per_instance()

    def _estimate_qps_per_instance(self) -> float:
        """Estimate QPS capacity per instance from benchmark data."""
        total_instances = (
            self._data.metadata.num_prefill_instances
            + self._data.metadata.num_decode_instances
        )
        if total_instances <= 0:
            return 1.0
        measured_qps = self._data.metadata.measured_qps
        if measured_qps <= 0:
            return 1.0
        return measured_qps / total_instances

    def _instances_for_qps(self, qps: float) -> int:
        """Calculate minimum instances needed for a given QPS."""
        if qps <= 0:
            return 0
        raw = qps / self._qps_per_instance
        return max(1, math.ceil(raw))

    def calculate(
        self,
        profile: TrafficProfile,
        gpu_cost_per_hour: float = 2.0,
        currency: str = "USD",
    ) -> GPUHourReport:
        """Calculate GPU hours and costs for a traffic profile."""
        # Build full 24-hour schedule (default to 0 QPS for unspecified hours)
        hour_map: dict[int, float] = {ht.hour: ht.qps for ht in profile.hours}

        hourly_breakdown: list[HourBreakdown] = []
        total_gpu_hours = 0.0
        total_cost = 0.0
        peak_qps = 0.0
        peak_instances = 0
        min_qps = float("inf")
        min_instances = float("inf")
        instance_sum = 0

        for h in range(24):
            qps = hour_map.get(h, 0.0)
            instances = self._instances_for_qps(qps)
            gpu_h = float(instances)  # 1 hour per instance
            cost = gpu_h * gpu_cost_per_hour

            hourly_breakdown.append(
                HourBreakdown(
                    hour=h,
                    qps=qps,
                    required_instances=instances,
                    gpu_hours=gpu_h,
                    cost=round(cost, 4),
                )
            )

            total_gpu_hours += gpu_h
            total_cost += cost
            instance_sum += instances

            if qps > peak_qps:
                peak_qps = qps
                peak_instances = instances
            if qps < min_qps:
                min_qps = qps
                min_instances = instances

        if min_qps == float("inf"):
            min_qps = 0.0
        if min_instances == float("inf"):
            min_instances = 0

        # Fixed provisioning = peak instances * 24h
        fixed_gpu_hours = float(peak_instances) * 24.0
        fixed_cost = fixed_gpu_hours * gpu_cost_per_hour
        saved_gpu_hours = fixed_gpu_hours - total_gpu_hours
        savings_pct = (saved_gpu_hours / fixed_gpu_hours * 100.0) if fixed_gpu_hours > 0 else 0.0

        avg_util = (instance_sum / (peak_instances * 24.0)) if peak_instances > 0 else 0.0

        scaling_savings = ScalingSavings(
            fixed_daily_gpu_hours=fixed_gpu_hours,
            dynamic_daily_gpu_hours=total_gpu_hours,
            saved_gpu_hours=round(saved_gpu_hours, 2),
            savings_percent=round(savings_pct, 2),
            fixed_daily_cost=round(fixed_cost, 2),
            dynamic_daily_cost=round(total_cost, 2),
            saved_cost=round(fixed_cost - total_cost, 2),
        )

        return GPUHourReport(
            profile_name=profile.name,
            gpu_cost_per_hour=gpu_cost_per_hour,
            currency=currency,
            peak_qps=peak_qps,
            peak_instances=peak_instances,
            off_peak_qps=min_qps,
            off_peak_instances=int(min_instances),
            daily_gpu_hours=round(total_gpu_hours, 2),
            monthly_gpu_hours=round(total_gpu_hours * 30, 2),
            daily_cost=round(total_cost, 2),
            monthly_cost=round(total_cost * 30, 2),
            avg_utilization=round(avg_util, 4),
            hourly_breakdown=hourly_breakdown,
            scaling_savings=scaling_savings,
            qps_per_instance=round(self._qps_per_instance, 4),
        )


def calculate_gpu_hours(
    data: BenchmarkData,
    profile: TrafficProfile,
    gpu_cost_per_hour: float = 2.0,
    currency: str = "USD",
) -> dict:
    """Convenience function for GPU hour calculation.

    Args:
        data: Benchmark data with measured QPS and instance counts.
        profile: 24-hour traffic profile.
        gpu_cost_per_hour: Cost per GPU instance per hour.
        currency: Currency label.

    Returns:
        Dictionary with GPU hour report.
    """
    calc = GPUHourCalculator(data)
    report = calc.calculate(profile, gpu_cost_per_hour, currency)
    return report.model_dump()

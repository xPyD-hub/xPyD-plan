"""Capacity planning module for xpyd-plan.

Given benchmark data at various cluster sizes, recommends the minimum
total instances and optimal P:D ratio to meet a target QPS and SLA.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.models import SLAConfig


class ConfidenceLevel(str, Enum):
    """Confidence level of the capacity recommendation."""

    HIGH = "high"  # interpolation between known data points
    MEDIUM = "medium"  # slight extrapolation beyond known range
    LOW = "low"  # significant extrapolation beyond known range


class ScalingDataPoint(BaseModel):
    """A data point mapping cluster size to measured performance."""

    total_instances: int
    num_prefill: int
    num_decode: int
    measured_qps: float
    max_qps_meeting_sla: float | None = None
    ttft_p95_ms: float
    tpot_p95_ms: float
    waste_rate: float


class CapacityRecommendation(BaseModel):
    """Result of capacity planning."""

    target_qps: float = Field(..., description="Requested target QPS")
    recommended_instances: int = Field(..., description="Minimum total instances needed")
    recommended_ratio: str = Field(..., description="Recommended P:D ratio string")
    num_prefill: int
    num_decode: int
    confidence: ConfidenceLevel
    estimated_headroom_pct: float = Field(
        ..., description="Estimated QPS headroom above target (%)"
    )
    scaling_points: list[ScalingDataPoint] = Field(
        default_factory=list, description="Data points used for scaling model"
    )
    notes: list[str] = Field(default_factory=list)


class CapacityPlanner:
    """Plans capacity based on benchmark data at different cluster sizes.

    Usage:
        planner = CapacityPlanner()
        planner.fit(benchmark_datasets)
        rec = planner.recommend(target_qps=50.0, sla=SLAConfig(...))
    """

    def __init__(self) -> None:
        self._points: list[ScalingDataPoint] = []
        self._fitted = False

    @property
    def points(self) -> list[ScalingDataPoint]:
        """Return fitted scaling data points."""
        return list(self._points)

    def fit(
        self,
        datasets: list[BenchmarkData],
        sla: SLAConfig | None = None,
    ) -> None:
        """Learn scaling characteristics from benchmark datasets.

        Args:
            datasets: Benchmark data from runs at different cluster sizes/QPS levels.
            sla: SLA config for determining which configs meet SLA.
        """
        if not datasets:
            raise ValueError("At least one benchmark dataset is required")

        if sla is None:
            sla = SLAConfig()

        points: list[ScalingDataPoint] = []

        for data in datasets:
            analyzer = BenchmarkAnalyzer()
            analyzer._data = data

            sla_check = analyzer.check_sla(sla)
            util = analyzer.compute_utilization()

            max_qps = data.metadata.measured_qps if sla_check.meets_all else None

            points.append(
                ScalingDataPoint(
                    total_instances=data.metadata.total_instances,
                    num_prefill=data.metadata.num_prefill_instances,
                    num_decode=data.metadata.num_decode_instances,
                    measured_qps=data.metadata.measured_qps,
                    max_qps_meeting_sla=max_qps,
                    ttft_p95_ms=sla_check.ttft_p95_ms,
                    tpot_p95_ms=sla_check.tpot_p95_ms,
                    waste_rate=util.waste_rate,
                )
            )

        # Sort by total instances
        points.sort(key=lambda p: (p.total_instances, p.measured_qps))
        self._points = points
        self._fitted = True

    def _estimate_qps_per_instance(self) -> float:
        """Estimate throughput per instance from data points that meet SLA."""
        valid = [p for p in self._points if p.max_qps_meeting_sla is not None]
        if not valid:
            # Fall back to all points
            valid = self._points
            values = [p.measured_qps / p.total_instances for p in valid]
        else:
            values = [p.max_qps_meeting_sla / p.total_instances for p in valid]  # type: ignore[operator]

        return sum(values) / len(values)

    def _determine_confidence(self, target_qps: float) -> ConfidenceLevel:
        """Determine confidence based on how far target is from known data."""
        all_qps = [p.measured_qps for p in self._points]
        min_qps = min(all_qps)
        max_qps = max(all_qps)
        qps_range = max_qps - min_qps if max_qps > min_qps else max_qps

        if min_qps <= target_qps <= max_qps:
            return ConfidenceLevel.HIGH
        elif target_qps < min_qps:
            # Below range — usually safe
            return ConfidenceLevel.HIGH

        # Extrapolation above
        overshoot = (target_qps - max_qps) / qps_range if qps_range > 0 else float("inf")
        if overshoot <= 0.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _find_best_ratio(self, target_instances: int) -> tuple[int, int]:
        """Find the best P:D split for the given total from known data.

        Uses the ratio from the closest data point that meets SLA,
        or falls back to the ratio with lowest waste.
        """
        valid = [p for p in self._points if p.max_qps_meeting_sla is not None]
        if not valid:
            valid = self._points

        # Find closest point by total_instances
        closest = min(valid, key=lambda p: abs(p.total_instances - target_instances))
        ratio = closest.num_prefill / (closest.num_prefill + closest.num_decode)

        num_prefill = max(1, round(ratio * target_instances))
        num_decode = max(1, target_instances - num_prefill)

        return num_prefill, num_decode

    def recommend(
        self,
        target_qps: float,
        sla: SLAConfig | None = None,
        max_instances: int = 64,
    ) -> CapacityRecommendation:
        """Recommend minimum instances and P:D ratio for target QPS.

        Args:
            target_qps: Target queries per second to support.
            sla: SLA constraints (used for confidence, not re-checked).
            max_instances: Upper bound on instance search (default: 64).

        Returns:
            CapacityRecommendation with minimum instances and ratio.

        Raises:
            RuntimeError: If fit() hasn't been called.
            ValueError: If target_qps is not positive.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before recommend()")

        if target_qps <= 0:
            raise ValueError("target_qps must be positive")

        qps_per_instance = self._estimate_qps_per_instance()
        raw_instances = target_qps / qps_per_instance

        # Add 20% headroom and round up
        min_instances = max(2, math.ceil(raw_instances * 1.2))

        # Cap at max_instances
        recommended = min(min_instances, max_instances)

        num_prefill, num_decode = self._find_best_ratio(recommended)
        actual_total = num_prefill + num_decode
        estimated_qps = actual_total * qps_per_instance
        headroom = ((estimated_qps - target_qps) / target_qps) * 100.0

        confidence = self._determine_confidence(target_qps)

        notes: list[str] = []
        if confidence == ConfidenceLevel.LOW:
            notes.append(
                "Significant extrapolation beyond benchmark data range. "
                "Run additional benchmarks at higher instance counts to improve accuracy."
            )
        if recommended >= max_instances:
            notes.append(
                f"Reached max_instances limit ({max_instances}). "
                "Target QPS may not be achievable."
            )

        return CapacityRecommendation(
            target_qps=target_qps,
            recommended_instances=actual_total,
            recommended_ratio=f"{num_prefill}P:{num_decode}D",
            num_prefill=num_prefill,
            num_decode=num_decode,
            confidence=confidence,
            estimated_headroom_pct=round(headroom, 1),
            scaling_points=self._points,
            notes=notes,
        )


def plan_capacity(
    benchmark_paths: list[str],
    target_qps: float,
    sla_config: SLAConfig | dict | None = None,
    max_instances: int = 64,
) -> dict[str, Any]:
    """Programmatic API for capacity planning.

    Args:
        benchmark_paths: Paths to benchmark JSON files at different scales.
        target_qps: Target QPS to achieve.
        sla_config: SLA configuration.
        max_instances: Maximum instances to consider.

    Returns:
        Dictionary with capacity recommendation.
    """
    import json

    from xpyd_plan.bench_adapter import load_benchmark_auto

    if isinstance(sla_config, dict):
        sla_config = SLAConfig(**sla_config)
    elif sla_config is None:
        sla_config = SLAConfig()

    datasets = [load_benchmark_auto(p) for p in benchmark_paths]

    planner = CapacityPlanner()
    planner.fit(datasets, sla=sla_config)
    rec = planner.recommend(target_qps=target_qps, sla=sla_config, max_instances=max_instances)

    return json.loads(rec.model_dump_json())

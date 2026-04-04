"""Throughput scaling analysis across benchmark runs with different instance counts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class ScalingPoint(BaseModel):
    """A single data point on the scaling curve."""

    total_instances: int = Field(..., ge=2, description="Total instance count")
    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    measured_qps: float = Field(..., gt=0, description="Measured throughput (QPS)")
    per_instance_qps: float = Field(..., gt=0, description="QPS per instance")
    scaling_efficiency: float = Field(
        ...,
        ge=0,
        le=1,
        description="Actual throughput / ideal linear throughput (relative to baseline)",
    )


class ScalingCurve(BaseModel):
    """Scaling curve with all data points and analysis."""

    points: list[ScalingPoint] = Field(..., min_length=1)
    baseline_per_instance_qps: float = Field(
        ..., gt=0, description="Per-instance QPS of the smallest configuration (baseline)"
    )
    knee_point: ScalingPoint | None = Field(
        None,
        description="First point where scaling efficiency drops below threshold",
    )
    knee_threshold: float = Field(
        0.8, description="Efficiency threshold used for knee detection"
    )
    optimal_point: ScalingPoint = Field(
        ..., description="Point with highest total QPS while above knee threshold"
    )


class ScalingReport(BaseModel):
    """Complete scaling analysis report."""

    curve: ScalingCurve
    recommendation: str = Field(..., description="Human-readable recommendation")


class ScalingAnalyzer:
    """Analyze throughput scaling across benchmark runs with different instance counts."""

    def __init__(self, knee_threshold: float = 0.8) -> None:
        """Initialize with knee-point detection threshold.

        Args:
            knee_threshold: Scaling efficiency below this triggers knee detection.
                Must be between 0 and 1 (exclusive). Default 0.8 (80%).
        """
        if not 0 < knee_threshold < 1:
            msg = f"knee_threshold must be between 0 and 1 exclusive, got {knee_threshold}"
            raise ValueError(msg)
        self.knee_threshold = knee_threshold

    def analyze(self, benchmarks: list[BenchmarkData]) -> ScalingReport:
        """Analyze scaling across multiple benchmark runs.

        Args:
            benchmarks: List of benchmark datasets with different instance counts.
                Must contain at least 2 benchmarks with distinct total_instances.

        Returns:
            ScalingReport with curve, knee point, and recommendation.

        Raises:
            ValueError: If fewer than 2 benchmarks or all have same instance count.
        """
        if len(benchmarks) < 2:
            msg = "Need at least 2 benchmark datasets for scaling analysis"
            raise ValueError(msg)

        # Sort by total instances
        sorted_benchmarks = sorted(benchmarks, key=lambda b: b.metadata.total_instances)

        # Check distinct instance counts
        instance_counts = {b.metadata.total_instances for b in sorted_benchmarks}
        if len(instance_counts) < 2:
            msg = "Need benchmarks with at least 2 distinct total_instances values"
            raise ValueError(msg)

        # Baseline: smallest configuration
        baseline = sorted_benchmarks[0]
        baseline_per_instance = baseline.metadata.measured_qps / baseline.metadata.total_instances

        # Build scaling points
        points: list[ScalingPoint] = []
        for bench in sorted_benchmarks:
            meta = bench.metadata
            per_inst = meta.measured_qps / meta.total_instances
            ideal_qps = baseline_per_instance * meta.total_instances
            efficiency = meta.measured_qps / ideal_qps if ideal_qps > 0 else 0.0
            # Clamp efficiency to [0, 1]
            efficiency = min(1.0, max(0.0, efficiency))

            points.append(
                ScalingPoint(
                    total_instances=meta.total_instances,
                    num_prefill=meta.num_prefill_instances,
                    num_decode=meta.num_decode_instances,
                    measured_qps=meta.measured_qps,
                    per_instance_qps=round(per_inst, 4),
                    scaling_efficiency=round(efficiency, 4),
                )
            )

        # Find knee point (first point below threshold)
        knee_point: ScalingPoint | None = None
        for point in points:
            if point.scaling_efficiency < self.knee_threshold:
                knee_point = point
                break

        # Optimal point: highest QPS among points at or above threshold
        above_threshold = [p for p in points if p.scaling_efficiency >= self.knee_threshold]
        if above_threshold:
            optimal = max(above_threshold, key=lambda p: p.measured_qps)
        else:
            # All below threshold, pick the one with best efficiency
            optimal = max(points, key=lambda p: p.scaling_efficiency)

        curve = ScalingCurve(
            points=points,
            baseline_per_instance_qps=round(baseline_per_instance, 4),
            knee_point=knee_point,
            knee_threshold=self.knee_threshold,
            optimal_point=optimal,
        )

        # Build recommendation
        recommendation = self._build_recommendation(curve)

        return ScalingReport(curve=curve, recommendation=recommendation)

    def _build_recommendation(self, curve: ScalingCurve) -> str:
        """Generate a human-readable recommendation from the scaling curve."""
        opt = curve.optimal_point
        if curve.knee_point is None:
            return (
                f"Scaling is efficient across all tested configurations. "
                f"Optimal: {opt.total_instances} instances "
                f"({opt.num_prefill}P:{opt.num_decode}D) at {opt.measured_qps:.1f} QPS "
                f"with {opt.scaling_efficiency:.0%} efficiency."
            )
        knee = curve.knee_point
        return (
            f"Diminishing returns detected at {knee.total_instances} instances "
            f"({knee.scaling_efficiency:.0%} efficiency). "
            f"Recommended: {opt.total_instances} instances "
            f"({opt.num_prefill}P:{opt.num_decode}D) at {opt.measured_qps:.1f} QPS "
            f"with {opt.scaling_efficiency:.0%} efficiency."
        )


def analyze_scaling(
    benchmarks: list[BenchmarkData],
    knee_threshold: float = 0.8,
) -> dict:
    """Programmatic API for scaling analysis.

    Args:
        benchmarks: List of benchmark datasets with different instance counts.
        knee_threshold: Efficiency threshold for knee-point detection.

    Returns:
        Dictionary representation of ScalingReport.
    """
    analyzer = ScalingAnalyzer(knee_threshold=knee_threshold)
    report = analyzer.analyze(benchmarks)
    return report.model_dump()

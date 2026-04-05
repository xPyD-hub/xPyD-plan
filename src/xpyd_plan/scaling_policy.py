"""Auto-scaling policy generator for P:D disaggregated inference clusters.

Analyzes benchmark data at multiple QPS levels and generates scaling rules
that map observed metrics (latency, utilization) to instance scaling actions.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.benchmark_models import BenchmarkData


class PolicyFormat(str, Enum):
    """Output format for scaling policies."""

    JSON = "json"
    YAML = "yaml"
    TABLE = "table"


class ScalingDirection(str, Enum):
    """Direction of a scaling action."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


class ScalingTrigger(BaseModel):
    """Condition that triggers a scaling action."""

    metric: str = Field(..., description="Metric name (e.g., ttft_p95_ms, tpot_p95_ms, qps)")
    threshold: float = Field(..., description="Threshold value")
    comparator: str = Field(..., description="Comparison operator (gt, lt, gte, lte)")
    sustained_seconds: int = Field(
        60, ge=0, description="Seconds the condition must hold before triggering"
    )


class ScalingAction(BaseModel):
    """Action to take when a trigger fires."""

    direction: ScalingDirection
    prefill_delta: int = Field(0, description="Change in prefill instances")
    decode_delta: int = Field(0, description="Change in decode instances")
    reason: str = Field("", description="Human-readable reason for this action")


class ScalingRule(BaseModel):
    """A single scaling rule: trigger → action."""

    name: str
    trigger: ScalingTrigger
    action: ScalingAction
    priority: int = Field(1, ge=1, description="Higher priority rules evaluated first")


class ScalingPolicy(BaseModel):
    """Complete auto-scaling policy for a P:D cluster."""

    rules: list[ScalingRule] = Field(default_factory=list)
    cooldown_seconds: int = Field(
        300, ge=0, description="Minimum seconds between scaling actions"
    )
    min_prefill_instances: int = Field(1, ge=1)
    min_decode_instances: int = Field(1, ge=1)
    max_prefill_instances: int = Field(16, ge=1)
    max_decode_instances: int = Field(16, ge=1)
    evaluation_interval_seconds: int = Field(
        30, ge=1, description="How often to evaluate scaling rules"
    )
    notes: list[str] = Field(default_factory=list)


def _compute_p95(values: list[float]) -> float:
    """Compute P95 of a list of values."""
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def _compute_p50(values: list[float]) -> float:
    """Compute P50 of a list of values."""
    if not values:
        return 0.0
    return float(np.percentile(values, 50))


class ScalingPolicyGenerator:
    """Generate auto-scaling policies from benchmark data.

    Analyzes benchmark data at multiple QPS levels to determine when to
    scale prefill/decode instances up or down based on SLA thresholds.
    """

    def __init__(
        self,
        benchmark_data: list[BenchmarkData],
        *,
        sla_ttft_ms: float = 500.0,
        sla_tpot_ms: float = 100.0,
        sla_total_ms: float = 5000.0,
        min_instances: int = 1,
        max_instances: int = 16,
    ) -> None:
        if not benchmark_data:
            msg = "At least one benchmark file is required"
            raise ValueError(msg)
        self._data = sorted(benchmark_data, key=lambda d: d.metadata.measured_qps)
        self._sla_ttft = sla_ttft_ms
        self._sla_tpot = sla_tpot_ms
        self._sla_total = sla_total_ms
        self._min_instances = min_instances
        self._max_instances = max_instances

    def generate(self) -> ScalingPolicy:
        """Generate a scaling policy from benchmark data."""
        rules: list[ScalingRule] = []
        notes: list[str] = []

        # Analyze each benchmark to understand latency at different loads
        profiles = []
        for bm in self._data:
            reqs = bm.requests
            if not reqs:
                continue
            ttft_p95 = _compute_p95([r.ttft_ms for r in reqs])
            tpot_p95 = _compute_p95([r.tpot_ms for r in reqs])
            total_p95 = _compute_p95([r.total_latency_ms for r in reqs])
            qps = bm.metadata.measured_qps
            prefill = bm.metadata.num_prefill_instances
            decode = bm.metadata.num_decode_instances
            profiles.append({
                "qps": qps,
                "ttft_p95": ttft_p95,
                "tpot_p95": tpot_p95,
                "total_p95": total_p95,
                "prefill": prefill,
                "decode": decode,
            })

        if not profiles:
            msg = "No valid benchmark data with requests found"
            raise ValueError(msg)

        # Scale-up rules: trigger when latency approaches SLA
        # Use 80% of SLA as warning threshold
        warning_factor = 0.8

        ttft_warning = self._sla_ttft * warning_factor
        tpot_warning = self._sla_tpot * warning_factor

        rules.append(ScalingRule(
            name="scale-up-ttft-warning",
            trigger=ScalingTrigger(
                metric="ttft_p95_ms",
                threshold=round(ttft_warning, 1),
                comparator="gte",
                sustained_seconds=60,
            ),
            action=ScalingAction(
                direction=ScalingDirection.SCALE_UP,
                prefill_delta=1,
                decode_delta=0,
                reason=f"TTFT P95 approaching SLA ({self._sla_ttft}ms), add prefill capacity",
            ),
            priority=2,
        ))

        rules.append(ScalingRule(
            name="scale-up-tpot-warning",
            trigger=ScalingTrigger(
                metric="tpot_p95_ms",
                threshold=round(tpot_warning, 1),
                comparator="gte",
                sustained_seconds=60,
            ),
            action=ScalingAction(
                direction=ScalingDirection.SCALE_UP,
                prefill_delta=0,
                decode_delta=1,
                reason=f"TPOT P95 approaching SLA ({self._sla_tpot}ms), add decode capacity",
            ),
            priority=2,
        ))

        rules.append(ScalingRule(
            name="scale-up-total-critical",
            trigger=ScalingTrigger(
                metric="total_latency_p95_ms",
                threshold=round(self._sla_total * 0.9, 1),
                comparator="gte",
                sustained_seconds=30,
            ),
            action=ScalingAction(
                direction=ScalingDirection.SCALE_UP,
                prefill_delta=1,
                decode_delta=1,
                reason=f"Total latency P95 near SLA ({self._sla_total}ms), add both instance types",
            ),
            priority=3,
        ))

        # Scale-down rules: based on low utilization
        # Find the lowest QPS profile that still meets SLA
        sla_meeting_profiles = [
            p for p in profiles
            if p["ttft_p95"] <= self._sla_ttft
            and p["tpot_p95"] <= self._sla_tpot
            and p["total_p95"] <= self._sla_total
        ]

        if sla_meeting_profiles:
            # Use the lowest-QPS SLA-meeting profile for scale-down thresholds
            low_profile = sla_meeting_profiles[0]
            # Scale down when TTFT is well below warning (< 50% of SLA)
            scale_down_ttft = self._sla_ttft * 0.4
            scale_down_tpot = self._sla_tpot * 0.4

            rules.append(ScalingRule(
                name="scale-down-ttft-idle",
                trigger=ScalingTrigger(
                    metric="ttft_p95_ms",
                    threshold=round(scale_down_ttft, 1),
                    comparator="lte",
                    sustained_seconds=300,
                ),
                action=ScalingAction(
                    direction=ScalingDirection.SCALE_DOWN,
                    prefill_delta=-1,
                    decode_delta=0,
                    reason="TTFT P95 well below SLA, reduce prefill capacity to save cost",
                ),
                priority=1,
            ))

            rules.append(ScalingRule(
                name="scale-down-tpot-idle",
                trigger=ScalingTrigger(
                    metric="tpot_p95_ms",
                    threshold=round(scale_down_tpot, 1),
                    comparator="lte",
                    sustained_seconds=300,
                ),
                action=ScalingAction(
                    direction=ScalingDirection.SCALE_DOWN,
                    prefill_delta=0,
                    decode_delta=-1,
                    reason="TPOT P95 well below SLA, reduce decode capacity to save cost",
                ),
                priority=1,
            ))

            notes.append(
                f"Scale-down thresholds derived from lowest SLA-meeting benchmark "
                f"(QPS={low_profile['qps']:.1f})."
            )
        else:
            notes.append(
                "No benchmark meets all SLA constraints. Scale-down rules not generated. "
                "Consider relaxing SLA or adding capacity."
            )

        # Estimate cooldown from data: use the spread of QPS levels
        if len(profiles) >= 2:
            cooldown = 300  # default 5 minutes
            notes.append(
                f"Cooldown set to {cooldown}s. Adjust based on observed instance startup time."
            )
        else:
            cooldown = 300
            notes.append(
                "Single benchmark provided. Policy thresholds may be less accurate. "
                "More QPS levels recommended."
            )

        # Sort rules by priority (highest first)
        rules.sort(key=lambda r: r.priority, reverse=True)

        return ScalingPolicy(
            rules=rules,
            cooldown_seconds=cooldown,
            min_prefill_instances=self._min_instances,
            min_decode_instances=self._min_instances,
            max_prefill_instances=self._max_instances,
            max_decode_instances=self._max_instances,
            evaluation_interval_seconds=30,
            notes=notes,
        )


def generate_scaling_policy(
    benchmark_paths: list[str],
    *,
    sla_ttft_ms: float = 500.0,
    sla_tpot_ms: float = 100.0,
    sla_total_ms: float = 5000.0,
    min_instances: int = 1,
    max_instances: int = 16,
) -> dict:
    """Programmatic API for scaling policy generation.

    Args:
        benchmark_paths: Paths to benchmark JSON files.
        sla_ttft_ms: TTFT SLA threshold in milliseconds.
        sla_tpot_ms: TPOT SLA threshold in milliseconds.
        sla_total_ms: Total latency SLA threshold in milliseconds.
        min_instances: Minimum instances per type.
        max_instances: Maximum instances per type.

    Returns:
        Dictionary representation of ScalingPolicy.
    """
    data = [load_benchmark_auto(p) for p in benchmark_paths]
    gen = ScalingPolicyGenerator(
        data,
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_ms=sla_total_ms,
        min_instances=min_instances,
        max_instances=max_instances,
    )
    policy = gen.generate()
    return policy.model_dump()

"""Sensitivity analysis — P:D ratio vs SLA satisfaction curves.

Identifies the "cliff" where SLA starts failing and recommends
P:D ratios with safety margins.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.benchmark_models import RatioCandidate, SLACheck
from xpyd_plan.models import SLAConfig


class SensitivityPoint(BaseModel):
    """Sensitivity data for a single P:D ratio."""

    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    ratio_str: str
    meets_sla: bool
    sla_check: SLACheck

    # Margin: how far each metric is from the SLA threshold.
    # Positive = headroom (good), negative = violation.
    # None if that SLA constraint is not set.
    ttft_margin_ms: float | None = None
    tpot_margin_ms: float | None = None
    total_latency_margin_ms: float | None = None

    # Relative margin as fraction of threshold (e.g. 0.2 = 20% headroom)
    ttft_margin_pct: float | None = None
    tpot_margin_pct: float | None = None
    total_latency_margin_pct: float | None = None

    waste_rate: float


class CliffPoint(BaseModel):
    """The critical P:D ratio where SLA transitions from pass to fail."""

    last_pass: SensitivityPoint | None = Field(
        None, description="Last P:D ratio that still meets SLA"
    )
    first_fail: SensitivityPoint | None = Field(
        None, description="First P:D ratio that fails SLA"
    )
    failing_metric: str | None = Field(
        None, description="Which SLA metric fails first (ttft/tpot/total_latency)"
    )
    direction: str = Field(
        "prefill_heavy",
        description="Direction of cliff: 'prefill_heavy' means too many P instances causes fail",
    )


class SafetyRecommendation(BaseModel):
    """P:D ratio recommendation with safety margin."""

    recommended: SensitivityPoint | None = None
    optimal: SensitivityPoint | None = None
    cliff_distance: int = Field(
        0, description="How many P:D steps away from the cliff"
    )
    min_margin_pct: float | None = Field(
        None, description="Minimum margin percentage across all SLA metrics"
    )


class SensitivityResult(BaseModel):
    """Complete sensitivity analysis result."""

    points: list[SensitivityPoint] = Field(default_factory=list)
    cliffs: list[CliffPoint] = Field(default_factory=list)
    recommendation: SafetyRecommendation | None = None
    total_instances: int


def _compute_margins(
    sla_check: SLACheck, sla: SLAConfig
) -> dict[str, float | None]:
    """Compute absolute and percentage margins for each SLA metric.

    Uses the evaluated percentile values from the SLA check (which respects
    the configured ``sla_percentile``), falling back to P95 when the
    evaluated values are not available (backward compatibility).
    """
    margins: dict[str, float | None] = {}

    ttft_val = (
        sla_check.ttft_evaluated_ms
        if sla_check.ttft_evaluated_ms is not None
        else sla_check.ttft_p95_ms
    )
    tpot_val = (
        sla_check.tpot_evaluated_ms
        if sla_check.tpot_evaluated_ms is not None
        else sla_check.tpot_p95_ms
    )
    total_val = (
        sla_check.total_latency_evaluated_ms
        if sla_check.total_latency_evaluated_ms is not None
        else sla_check.total_latency_p95_ms
    )

    if sla.ttft_ms is not None:
        margins["ttft_margin_ms"] = sla.ttft_ms - ttft_val
        margins["ttft_margin_pct"] = margins["ttft_margin_ms"] / sla.ttft_ms
    else:
        margins["ttft_margin_ms"] = None
        margins["ttft_margin_pct"] = None

    if sla.tpot_ms is not None:
        margins["tpot_margin_ms"] = sla.tpot_ms - tpot_val
        margins["tpot_margin_pct"] = margins["tpot_margin_ms"] / sla.tpot_ms
    else:
        margins["tpot_margin_ms"] = None
        margins["tpot_margin_pct"] = None

    if sla.max_latency_ms is not None:
        margins["total_latency_margin_ms"] = sla.max_latency_ms - total_val
        margins["total_latency_margin_pct"] = (
            margins["total_latency_margin_ms"] / sla.max_latency_ms
        )
    else:
        margins["total_latency_margin_ms"] = None
        margins["total_latency_margin_pct"] = None

    return margins


def analyze_sensitivity(
    analyzer: BenchmarkAnalyzer,
    total_instances: int,
    sla: SLAConfig,
) -> SensitivityResult:
    """Run sensitivity analysis across all P:D ratios.

    For each possible P:D split, computes SLA margins and identifies
    cliff points where SLA transitions from pass to fail.

    Args:
        analyzer: BenchmarkAnalyzer with loaded data.
        total_instances: Total number of instances.
        sla: SLA configuration.

    Returns:
        SensitivityResult with points, cliffs, and recommendation.
    """
    if total_instances < 2:
        return SensitivityResult(points=[], cliffs=[], total_instances=total_instances)

    points: list[SensitivityPoint] = []

    for num_p in range(1, total_instances):
        num_d = total_instances - num_p
        candidate: RatioCandidate = analyzer._estimate_ratio_performance(num_p, num_d, sla)
        assert candidate.sla_check is not None

        margins = _compute_margins(candidate.sla_check, sla)

        point = SensitivityPoint(
            num_prefill=num_p,
            num_decode=num_d,
            ratio_str=candidate.ratio_str,
            meets_sla=candidate.meets_sla,
            sla_check=candidate.sla_check,
            waste_rate=candidate.waste_rate,
            **margins,
        )
        points.append(point)

    # Detect cliffs: transitions from pass -> fail
    cliffs = _detect_cliffs(points)

    # Generate safety recommendation
    recommendation = _generate_recommendation(points, cliffs)

    return SensitivityResult(
        points=points,
        cliffs=cliffs,
        recommendation=recommendation,
        total_instances=total_instances,
    )


def _detect_cliffs(points: list[SensitivityPoint]) -> list[CliffPoint]:
    """Detect cliff points where SLA transitions from pass to fail.

    Scans from left (1P) to right (max P) looking for transitions.
    """
    cliffs: list[CliffPoint] = []

    for i in range(len(points) - 1):
        curr = points[i]
        nxt = points[i + 1]

        if curr.meets_sla and not nxt.meets_sla:
            failing = _identify_failing_metric(nxt)
            cliffs.append(CliffPoint(
                last_pass=curr,
                first_fail=nxt,
                failing_metric=failing,
                direction="prefill_heavy",
            ))
        elif not curr.meets_sla and nxt.meets_sla:
            failing = _identify_failing_metric(curr)
            cliffs.append(CliffPoint(
                last_pass=nxt,
                first_fail=curr,
                failing_metric=failing,
                direction="decode_heavy",
            ))

    return cliffs


def _identify_failing_metric(point: SensitivityPoint) -> str | None:
    """Identify which SLA metric is most violated at a given point."""
    candidates = []
    if point.ttft_margin_pct is not None and point.ttft_margin_pct < 0:
        candidates.append(("ttft", point.ttft_margin_pct))
    if point.tpot_margin_pct is not None and point.tpot_margin_pct < 0:
        candidates.append(("tpot", point.tpot_margin_pct))
    if (
        point.total_latency_margin_pct is not None
        and point.total_latency_margin_pct < 0
    ):
        candidates.append(("total_latency", point.total_latency_margin_pct))

    if not candidates:
        if point.ttft_margin_ms is not None and point.ttft_margin_ms < 0:
            return "ttft"
        if point.tpot_margin_ms is not None and point.tpot_margin_ms < 0:
            return "tpot"
        if (
            point.total_latency_margin_ms is not None
            and point.total_latency_margin_ms < 0
        ):
            return "total_latency"
        return None

    return min(candidates, key=lambda x: x[1])[0]


def _generate_recommendation(
    points: list[SensitivityPoint],
    cliffs: list[CliffPoint],
) -> SafetyRecommendation | None:
    """Generate a safety-margin recommendation.

    Strategy: find the optimal point (meets SLA, min waste), then ensure
    it's at least 1 step away from any cliff. If not, shift towards safety.
    """
    passing = [p for p in points if p.meets_sla]
    if not passing:
        return SafetyRecommendation()

    # Optimal = min waste among passing
    optimal = min(passing, key=lambda p: p.waste_rate)

    # Compute cliff distance for optimal
    cliff_indices = set()
    for cliff in cliffs:
        if cliff.first_fail:
            idx = next(
                (i for i, p in enumerate(points) if p.num_prefill == cliff.first_fail.num_prefill),
                None,
            )
            if idx is not None:
                cliff_indices.add(idx)

    optimal_idx = next(
        i for i, p in enumerate(points) if p.num_prefill == optimal.num_prefill
    )

    min_cliff_dist = float("inf")
    for ci in cliff_indices:
        dist = abs(optimal_idx - ci)
        if dist < min_cliff_dist:
            min_cliff_dist = dist

    if min_cliff_dist == float("inf"):
        min_cliff_dist = len(points)

    # If cliff distance >= 2, optimal is safe enough
    if min_cliff_dist >= 2:
        margins = []
        for attr in ("ttft_margin_pct", "tpot_margin_pct", "total_latency_margin_pct"):
            val = getattr(optimal, attr)
            if val is not None:
                margins.append(val)
        min_margin = min(margins) if margins else None

        return SafetyRecommendation(
            recommended=optimal,
            optimal=optimal,
            cliff_distance=int(min_cliff_dist),
            min_margin_pct=min_margin,
        )

    # Need to shift away from cliff
    best_safe = None
    best_safe_dist = 0
    best_safe_waste = float("inf")

    for i, p in enumerate(points):
        if not p.meets_sla:
            continue
        dist = min((abs(i - ci) for ci in cliff_indices), default=len(points))
        if dist > best_safe_dist or (dist == best_safe_dist and p.waste_rate < best_safe_waste):
            best_safe = p
            best_safe_dist = dist
            best_safe_waste = p.waste_rate

    recommended = best_safe if best_safe else optimal
    rec_margins = []
    for attr in ("ttft_margin_pct", "tpot_margin_pct", "total_latency_margin_pct"):
        val = getattr(recommended, attr)
        if val is not None:
            rec_margins.append(val)

    return SafetyRecommendation(
        recommended=recommended,
        optimal=optimal,
        cliff_distance=int(best_safe_dist) if best_safe else int(min_cliff_dist),
        min_margin_pct=min(rec_margins) if rec_margins else None,
    )

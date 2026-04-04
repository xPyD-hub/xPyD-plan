"""Workload characterization and clustering — segment benchmark requests by token patterns."""

from __future__ import annotations

import enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData, BenchmarkRequest
from .models import SLAConfig


class WorkloadCategory(str, enum.Enum):
    """Workload class based on token distribution patterns."""

    PREFILL_HEAVY = "prefill_heavy"
    DECODE_HEAVY = "decode_heavy"
    BALANCED = "balanced"
    SHORT = "short"
    LONG = "long"


class WorkloadClass(BaseModel):
    """A single workload cluster with its statistics."""

    category: WorkloadCategory = Field(..., description="Workload category label")
    request_count: int = Field(..., ge=0, description="Number of requests in this class")
    fraction: float = Field(..., ge=0, le=1, description="Fraction of total requests")
    avg_prompt_tokens: float = Field(..., description="Mean prompt tokens")
    avg_output_tokens: float = Field(..., description="Mean output tokens")
    prompt_output_ratio: float = Field(..., description="Mean prompt/output token ratio")
    ttft_p50_ms: float = Field(..., description="TTFT P50 (ms)")
    ttft_p95_ms: float = Field(..., description="TTFT P95 (ms)")
    ttft_p99_ms: float = Field(..., description="TTFT P99 (ms)")
    tpot_p50_ms: float = Field(..., description="TPOT P50 (ms)")
    tpot_p95_ms: float = Field(..., description="TPOT P95 (ms)")
    tpot_p99_ms: float = Field(..., description="TPOT P99 (ms)")
    total_latency_p50_ms: float = Field(..., description="Total latency P50 (ms)")
    total_latency_p95_ms: float = Field(..., description="Total latency P95 (ms)")
    total_latency_p99_ms: float = Field(..., description="Total latency P99 (ms)")
    meets_sla: Optional[bool] = Field(None, description="Whether this class meets SLA")
    sla_margin_ms: Optional[float] = Field(
        None, description="SLA margin (negative = violation) in ms"
    )


class WorkloadProfile(BaseModel):
    """Overall workload profile summary."""

    total_requests: int = Field(..., description="Total requests analyzed")
    num_classes: int = Field(..., description="Number of non-empty workload classes")
    dominant_class: WorkloadCategory = Field(
        ..., description="Category with the most requests"
    )
    classification_thresholds: dict[str, float] = Field(
        ..., description="Thresholds used for classification"
    )


class WorkloadReport(BaseModel):
    """Complete workload characterization report."""

    profile: WorkloadProfile = Field(..., description="Overall workload profile")
    classes: list[WorkloadClass] = Field(..., description="Per-class statistics")
    bottleneck: Optional[WorkloadClass] = Field(
        None, description="Class with worst SLA margin (bottleneck)"
    )


# --- Thresholds ---
# prompt/output ratio thresholds
_RATIO_HIGH = 3.0  # prompt >> output → prefill_heavy
_RATIO_LOW = 0.33  # output >> prompt → decode_heavy
# total token thresholds (prompt + output)
_TOTAL_SHORT = 256
_TOTAL_LONG = 2048


def _classify_request(req: BenchmarkRequest) -> WorkloadCategory:
    """Classify a single request into a workload category."""
    total = req.prompt_tokens + req.output_tokens
    if total <= _TOTAL_SHORT:
        return WorkloadCategory.SHORT
    if total >= _TOTAL_LONG:
        return WorkloadCategory.LONG
    ratio = req.prompt_tokens / req.output_tokens
    if ratio >= _RATIO_HIGH:
        return WorkloadCategory.PREFILL_HEAVY
    if ratio <= _RATIO_LOW:
        return WorkloadCategory.DECODE_HEAVY
    return WorkloadCategory.BALANCED


def _compute_percentiles(values: list[float]) -> tuple[float, float, float]:
    """Return P50, P95, P99."""
    arr = np.array(values)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def _build_class(
    category: WorkloadCategory,
    requests: list[BenchmarkRequest],
    total_requests: int,
    sla: Optional[SLAConfig],
) -> WorkloadClass:
    """Build a WorkloadClass from a list of requests."""
    n = len(requests)
    if n == 0:
        return WorkloadClass(
            category=category,
            request_count=0,
            fraction=0.0,
            avg_prompt_tokens=0.0,
            avg_output_tokens=0.0,
            prompt_output_ratio=0.0,
            ttft_p50_ms=0.0,
            ttft_p95_ms=0.0,
            ttft_p99_ms=0.0,
            tpot_p50_ms=0.0,
            tpot_p95_ms=0.0,
            tpot_p99_ms=0.0,
            total_latency_p50_ms=0.0,
            total_latency_p95_ms=0.0,
            total_latency_p99_ms=0.0,
            meets_sla=None,
            sla_margin_ms=None,
        )

    avg_prompt = float(np.mean([r.prompt_tokens for r in requests]))
    avg_output = float(np.mean([r.output_tokens for r in requests]))
    ratio = avg_prompt / avg_output if avg_output > 0 else float("inf")

    ttft_p50, ttft_p95, ttft_p99 = _compute_percentiles([r.ttft_ms for r in requests])
    tpot_p50, tpot_p95, tpot_p99 = _compute_percentiles([r.tpot_ms for r in requests])
    tlat_p50, tlat_p95, tlat_p99 = _compute_percentiles(
        [r.total_latency_ms for r in requests]
    )

    meets_sla = None
    sla_margin_ms = None
    if sla is not None and sla.has_constraints():
        pctl = sla.sla_percentile
        ttft_at_p = float(np.percentile([r.ttft_ms for r in requests], pctl))
        tpot_at_p = float(np.percentile([r.tpot_ms for r in requests], pctl))
        tlat_at_p = float(np.percentile([r.total_latency_ms for r in requests], pctl))

        margins: list[float] = []
        if sla.ttft_ms is not None:
            margins.append(sla.ttft_ms - ttft_at_p)
        if sla.tpot_ms is not None:
            margins.append(sla.tpot_ms - tpot_at_p)
        if sla.max_latency_ms is not None:
            margins.append(sla.max_latency_ms - tlat_at_p)

        if margins:
            sla_margin_ms = min(margins)
            meets_sla = sla_margin_ms >= 0

    return WorkloadClass(
        category=category,
        request_count=n,
        fraction=n / total_requests,
        avg_prompt_tokens=round(avg_prompt, 1),
        avg_output_tokens=round(avg_output, 1),
        prompt_output_ratio=round(ratio, 2),
        ttft_p50_ms=round(ttft_p50, 2),
        ttft_p95_ms=round(ttft_p95, 2),
        ttft_p99_ms=round(ttft_p99, 2),
        tpot_p50_ms=round(tpot_p50, 2),
        tpot_p95_ms=round(tpot_p95, 2),
        tpot_p99_ms=round(tpot_p99, 2),
        total_latency_p50_ms=round(tlat_p50, 2),
        total_latency_p95_ms=round(tlat_p95, 2),
        total_latency_p99_ms=round(tlat_p99, 2),
        meets_sla=meets_sla,
        sla_margin_ms=round(sla_margin_ms, 2) if sla_margin_ms is not None else None,
    )


class WorkloadClassifier:
    """Segment benchmark requests into workload clusters by token distribution patterns."""

    def __init__(self, sla: Optional[SLAConfig] = None) -> None:
        self.sla = sla

    def classify(self, data: BenchmarkData) -> WorkloadReport:
        """Classify all requests and produce a workload report."""
        buckets: dict[WorkloadCategory, list[BenchmarkRequest]] = {
            cat: [] for cat in WorkloadCategory
        }
        for req in data.requests:
            cat = _classify_request(req)
            buckets[cat].append(req)

        total = len(data.requests)
        classes: list[WorkloadClass] = []
        for cat in WorkloadCategory:
            wc = _build_class(cat, buckets[cat], total, self.sla)
            classes.append(wc)

        non_empty = [c for c in classes if c.request_count > 0]
        dominant = max(non_empty, key=lambda c: c.request_count) if non_empty else classes[0]

        bottleneck = None
        if self.sla is not None and self.sla.has_constraints():
            candidates = [c for c in non_empty if c.sla_margin_ms is not None]
            if candidates:
                bottleneck = min(candidates, key=lambda c: c.sla_margin_ms)  # type: ignore[arg-type]

        profile = WorkloadProfile(
            total_requests=total,
            num_classes=len(non_empty),
            dominant_class=dominant.category,
            classification_thresholds={
                "ratio_high": _RATIO_HIGH,
                "ratio_low": _RATIO_LOW,
                "total_short": _TOTAL_SHORT,
                "total_long": _TOTAL_LONG,
            },
        )

        return WorkloadReport(
            profile=profile,
            classes=classes,
            bottleneck=bottleneck,
        )


def classify_workload(
    data: BenchmarkData,
    sla: Optional[SLAConfig] = None,
) -> WorkloadReport:
    """Programmatic API: classify benchmark requests into workload clusters.

    Args:
        data: Benchmark data to analyze.
        sla: Optional SLA config for per-class compliance checking.

    Returns:
        WorkloadReport with per-class statistics and bottleneck identification.
    """
    return WorkloadClassifier(sla=sla).classify(data)

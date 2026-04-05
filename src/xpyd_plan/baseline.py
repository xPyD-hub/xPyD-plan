"""Latency baseline manager — save, load, and compare golden baselines.

Establish a reference latency profile from a known-good benchmark run,
persist it as JSON, and compare future runs against it with configurable
regression thresholds.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class BaselineVerdict(str, Enum):
    """Overall comparison verdict."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class MetricBaseline(BaseModel):
    """Latency percentiles for a single metric."""

    p50_ms: float = Field(..., description="P50 latency in ms")
    p95_ms: float = Field(..., description="P95 latency in ms")
    p99_ms: float = Field(..., description="P99 latency in ms")


class BaselineProfile(BaseModel):
    """Saved baseline profile from a golden benchmark run."""

    ttft: MetricBaseline = Field(..., description="TTFT percentiles")
    tpot: MetricBaseline = Field(..., description="TPOT percentiles")
    total_latency: MetricBaseline = Field(..., description="Total latency percentiles")
    qps: float = Field(..., description="Measured QPS")
    request_count: int = Field(..., description="Number of requests")
    num_prefill_instances: int = Field(..., description="Prefill instance count")
    num_decode_instances: int = Field(..., description="Decode instance count")


class MetricDelta(BaseModel):
    """Comparison result for a single metric at a single percentile."""

    metric: str = Field(..., description="Metric name")
    percentile: str = Field(..., description="Percentile label (p50, p95, p99)")
    baseline_ms: float = Field(..., description="Baseline value in ms")
    current_ms: float = Field(..., description="Current value in ms")
    delta_ms: float = Field(..., description="Absolute delta (current - baseline)")
    delta_pct: float = Field(..., description="Relative delta as %")
    verdict: BaselineVerdict = Field(..., description="Pass/warn/fail for this metric")


class BaselineComparison(BaseModel):
    """Full comparison report against a saved baseline."""

    deltas: list[MetricDelta] = Field(..., description="Per-metric-percentile deltas")
    overall_verdict: BaselineVerdict = Field(..., description="Overall verdict")
    regression_threshold_pct: float = Field(
        ..., description="Configured regression threshold %"
    )
    warn_threshold_pct: float = Field(
        ..., description="Warning threshold % (half of regression)"
    )
    summary: str = Field(..., description="Human-readable summary")


def _percentile_value(values: list[float], pct: float) -> float:
    """Compute percentile using linear interpolation."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    k = (pct / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)


def _extract_metric(data: BenchmarkData, attr: str) -> MetricBaseline:
    """Extract P50/P95/P99 for a given request attribute."""
    values = [getattr(r, attr) for r in data.requests]
    return MetricBaseline(
        p50_ms=_percentile_value(values, 50.0),
        p95_ms=_percentile_value(values, 95.0),
        p99_ms=_percentile_value(values, 99.0),
    )


class BaselineManager:
    """Save, load, and compare latency baselines."""

    def save(self, data: BenchmarkData) -> BaselineProfile:
        """Create a baseline profile from benchmark data."""
        profile = BaselineProfile(
            ttft=_extract_metric(data, "ttft_ms"),
            tpot=_extract_metric(data, "tpot_ms"),
            total_latency=_extract_metric(data, "total_latency_ms"),
            qps=data.metadata.measured_qps,
            request_count=len(data.requests),
            num_prefill_instances=data.metadata.num_prefill_instances,
            num_decode_instances=data.metadata.num_decode_instances,
        )
        return profile

    def save_to_file(self, data: BenchmarkData, path: str | Path) -> BaselineProfile:
        """Create baseline profile and write to JSON file."""
        profile = self.save(data)
        path = Path(path)
        path.write_text(json.dumps(profile.model_dump(), indent=2) + "\n")
        return profile

    def load(self, path: str | Path) -> BaselineProfile:
        """Load a baseline profile from JSON file."""
        path = Path(path)
        raw = json.loads(path.read_text())
        return BaselineProfile.model_validate(raw)

    def compare(
        self,
        data: BenchmarkData,
        baseline: BaselineProfile,
        regression_threshold_pct: float = 10.0,
    ) -> BaselineComparison:
        """Compare benchmark data against a baseline profile.

        Args:
            data: Current benchmark data.
            baseline: Saved baseline profile.
            regression_threshold_pct: Percentage increase that constitutes a
                regression (default 10%). Warning threshold is half this value.

        Returns:
            BaselineComparison with per-metric deltas and overall verdict.
        """
        warn_threshold = regression_threshold_pct / 2.0
        current = self.save(data)

        deltas: list[MetricDelta] = []
        metrics = [
            ("ttft", current.ttft, baseline.ttft),
            ("tpot", current.tpot, baseline.tpot),
            ("total_latency", current.total_latency, baseline.total_latency),
        ]

        for metric_name, cur_metric, base_metric in metrics:
            for pct_label, cur_val, base_val in [
                ("p50", cur_metric.p50_ms, base_metric.p50_ms),
                ("p95", cur_metric.p95_ms, base_metric.p95_ms),
                ("p99", cur_metric.p99_ms, base_metric.p99_ms),
            ]:
                delta_ms = cur_val - base_val
                if base_val > 0:
                    delta_pct = (delta_ms / base_val) * 100.0
                else:
                    delta_pct = 0.0 if cur_val == 0 else 100.0

                if delta_pct >= regression_threshold_pct:
                    verdict = BaselineVerdict.FAIL
                elif delta_pct >= warn_threshold:
                    verdict = BaselineVerdict.WARN
                else:
                    verdict = BaselineVerdict.PASS

                deltas.append(
                    MetricDelta(
                        metric=metric_name,
                        percentile=pct_label,
                        baseline_ms=base_val,
                        current_ms=cur_val,
                        delta_ms=round(delta_ms, 3),
                        delta_pct=round(delta_pct, 3),
                        verdict=verdict,
                    )
                )

        # Overall verdict: worst across all deltas
        if any(d.verdict == BaselineVerdict.FAIL for d in deltas):
            overall = BaselineVerdict.FAIL
        elif any(d.verdict == BaselineVerdict.WARN for d in deltas):
            overall = BaselineVerdict.WARN
        else:
            overall = BaselineVerdict.PASS

        fail_count = sum(1 for d in deltas if d.verdict == BaselineVerdict.FAIL)
        warn_count = sum(1 for d in deltas if d.verdict == BaselineVerdict.WARN)

        if overall == BaselineVerdict.PASS:
            summary = "All metrics within baseline tolerance. No regression detected."
        elif overall == BaselineVerdict.WARN:
            summary = (
                f"{warn_count} metric(s) approaching regression threshold. "
                "Monitor closely."
            )
        else:
            summary = (
                f"{fail_count} metric(s) regressed beyond {regression_threshold_pct}% "
                f"threshold. Investigation recommended."
            )

        return BaselineComparison(
            deltas=deltas,
            overall_verdict=overall,
            regression_threshold_pct=regression_threshold_pct,
            warn_threshold_pct=warn_threshold,
            summary=summary,
        )


def save_baseline(data: BenchmarkData) -> BaselineProfile:
    """Programmatic API: create a baseline profile from benchmark data."""
    return BaselineManager().save(data)


def compare_baseline(
    data: BenchmarkData,
    baseline: BaselineProfile,
    regression_threshold_pct: float = 10.0,
) -> BaselineComparison:
    """Programmatic API: compare benchmark data against a baseline."""
    return BaselineManager().compare(data, baseline, regression_threshold_pct)

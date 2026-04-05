"""Benchmark Diff Report — comprehensive side-by-side comparison of two benchmarks.

Generates a structured diff report combining summary stats, latency percentiles,
SLA compliance, throughput, and token distribution comparisons with
regression/improvement annotations.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class ChangeDirection(str, Enum):
    """Direction of a metric change between baseline and target."""

    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    UNCHANGED = "UNCHANGED"


class MetricDiff(BaseModel):
    """Diff for a single metric between baseline and target."""

    metric_name: str = Field(..., description="Human-readable metric name")
    baseline_value: float = Field(..., description="Value in baseline benchmark")
    target_value: float = Field(..., description="Value in target benchmark")
    absolute_delta: float = Field(..., description="target - baseline")
    relative_delta_pct: Optional[float] = Field(
        None, description="Percentage change (None if baseline is 0)"
    )
    direction: ChangeDirection = Field(
        ..., description="Whether change is improvement, regression, or unchanged"
    )


class LatencyDiffSection(BaseModel):
    """Latency percentile diffs for one latency metric (TTFT/TPOT/total)."""

    metric: str = Field(..., description="Latency metric name")
    p50: MetricDiff = Field(..., description="P50 diff")
    p95: MetricDiff = Field(..., description="P95 diff")
    p99: MetricDiff = Field(..., description="P99 diff")


class DiffSummary(BaseModel):
    """High-level summary of the diff."""

    total_diffs: int = Field(..., ge=0, description="Total metrics compared")
    improvements: int = Field(..., ge=0, description="Metrics that improved")
    regressions: int = Field(..., ge=0, description="Metrics that regressed")
    unchanged: int = Field(..., ge=0, description="Metrics unchanged")
    verdict: str = Field(
        ..., description="Overall verdict: BETTER, WORSE, MIXED, EQUIVALENT"
    )


class DiffReport(BaseModel):
    """Complete benchmark diff report."""

    baseline_file: str = Field(..., description="Baseline benchmark identifier")
    target_file: str = Field(..., description="Target benchmark identifier")
    baseline_requests: int = Field(..., ge=0, description="Request count in baseline")
    target_requests: int = Field(..., ge=0, description="Request count in target")
    baseline_qps: float = Field(..., ge=0, description="Baseline measured QPS")
    target_qps: float = Field(..., ge=0, description="Target measured QPS")
    qps_diff: MetricDiff = Field(..., description="QPS comparison")
    latency_diffs: list[LatencyDiffSection] = Field(
        ..., description="Per-metric latency diffs"
    )
    token_diffs: list[MetricDiff] = Field(
        ..., description="Token distribution diffs"
    )
    summary: DiffSummary = Field(..., description="Overall diff summary")
    regression_threshold_pct: float = Field(
        ..., description="Threshold used for regression detection"
    )


class DiffReporter:
    """Generate comprehensive diff reports between two benchmarks."""

    def __init__(
        self,
        regression_threshold_pct: float = 5.0,
    ) -> None:
        """Initialize the reporter.

        Args:
            regression_threshold_pct: Percentage change threshold above which
                a latency increase is classified as a regression (default 5%).
        """
        if regression_threshold_pct < 0:
            raise ValueError("regression_threshold_pct must be >= 0")
        self._threshold = regression_threshold_pct

    def compare(
        self,
        baseline: BenchmarkData,
        target: BenchmarkData,
        baseline_name: str = "baseline",
        target_name: str = "target",
    ) -> DiffReport:
        """Compare two benchmarks and produce a diff report.

        Args:
            baseline: The reference benchmark.
            target: The benchmark to compare against baseline.
            baseline_name: Label for baseline.
            target_name: Label for target.

        Returns:
            A DiffReport with all comparisons.
        """
        b_reqs = baseline.requests
        t_reqs = target.requests

        # QPS diff (higher is better)
        qps_diff = self._make_diff(
            "measured_qps",
            baseline.metadata.measured_qps,
            target.metadata.measured_qps,
            higher_is_better=True,
        )

        # Latency diffs (lower is better)
        latency_diffs = []
        for metric, field_p in [
            ("TTFT", "ttft_ms"),
            ("TPOT", "tpot_ms"),
            ("Total Latency", "total_latency_ms"),
        ]:
            b_vals = np.array([getattr(r, field_p) for r in b_reqs])
            t_vals = np.array([getattr(r, field_p) for r in t_reqs])

            section = LatencyDiffSection(
                metric=metric,
                p50=self._make_diff(
                    f"{metric} P50",
                    float(np.percentile(b_vals, 50)),
                    float(np.percentile(t_vals, 50)),
                    higher_is_better=False,
                ),
                p95=self._make_diff(
                    f"{metric} P95",
                    float(np.percentile(b_vals, 95)),
                    float(np.percentile(t_vals, 95)),
                    higher_is_better=False,
                ),
                p99=self._make_diff(
                    f"{metric} P99",
                    float(np.percentile(b_vals, 99)),
                    float(np.percentile(t_vals, 99)),
                    higher_is_better=False,
                ),
            )
            latency_diffs.append(section)

        # Token distribution diffs (informational — no direction)
        token_diffs = []
        for label, field in [
            ("Mean Prompt Tokens", "prompt_tokens"),
            ("Mean Output Tokens", "output_tokens"),
        ]:
            b_mean = float(np.mean([getattr(r, field) for r in b_reqs]))
            t_mean = float(np.mean([getattr(r, field) for r in t_reqs]))
            token_diffs.append(
                self._make_diff(label, b_mean, t_mean, higher_is_better=None)
            )

        # Aggregate summary
        all_diffs: list[MetricDiff] = [qps_diff]
        for section in latency_diffs:
            all_diffs.extend([section.p50, section.p95, section.p99])
        # Don't count token diffs in verdict — they're informational

        improvements = sum(
            1 for d in all_diffs if d.direction == ChangeDirection.IMPROVED
        )
        regressions = sum(
            1 for d in all_diffs if d.direction == ChangeDirection.REGRESSED
        )
        unchanged = sum(
            1 for d in all_diffs if d.direction == ChangeDirection.UNCHANGED
        )
        total = len(all_diffs)

        if regressions == 0 and improvements > 0:
            verdict = "BETTER"
        elif improvements == 0 and regressions > 0:
            verdict = "WORSE"
        elif improvements == 0 and regressions == 0:
            verdict = "EQUIVALENT"
        else:
            verdict = "MIXED"

        return DiffReport(
            baseline_file=baseline_name,
            target_file=target_name,
            baseline_requests=len(b_reqs),
            target_requests=len(t_reqs),
            baseline_qps=baseline.metadata.measured_qps,
            target_qps=target.metadata.measured_qps,
            qps_diff=qps_diff,
            latency_diffs=latency_diffs,
            token_diffs=token_diffs,
            summary=DiffSummary(
                total_diffs=total,
                improvements=improvements,
                regressions=regressions,
                unchanged=unchanged,
                verdict=verdict,
            ),
            regression_threshold_pct=self._threshold,
        )

    def to_markdown(self, report: DiffReport) -> str:
        """Render a DiffReport as a Markdown string."""
        lines: list[str] = []
        lines.append(
            f"# Benchmark Diff: {report.baseline_file} → {report.target_file}"
        )
        lines.append("")
        lines.append(f"**Verdict: {report.summary.verdict}** "
                      f"({report.summary.improvements} improved, "
                      f"{report.summary.regressions} regressed, "
                      f"{report.summary.unchanged} unchanged)")
        lines.append("")

        # Overview
        lines.append("## Overview")
        lines.append("")
        lines.append("| | Baseline | Target | Delta |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| Requests | {report.baseline_requests} "
            f"| {report.target_requests} "
            f"| {report.target_requests - report.baseline_requests:+d} |"
        )
        lines.append(self._md_diff_row("QPS", report.qps_diff))
        lines.append("")

        # Latency
        lines.append("## Latency Comparison")
        for section in report.latency_diffs:
            lines.append("")
            lines.append(f"### {section.metric}")
            lines.append("")
            lines.append("| Percentile | Baseline (ms) | Target (ms) | Delta | Change |")
            lines.append("|---|---|---|---|---|")
            for label, diff in [("P50", section.p50), ("P95", section.p95), ("P99", section.p99)]:
                icon = self._direction_icon(diff.direction)
                rel = (
                    f"{diff.relative_delta_pct:+.1f}%"
                    if diff.relative_delta_pct is not None
                    else "N/A"
                )
                lines.append(
                    f"| {label} | {diff.baseline_value:.2f} "
                    f"| {diff.target_value:.2f} "
                    f"| {rel} | {icon} {diff.direction.value} |"
                )
        lines.append("")

        # Tokens
        lines.append("## Token Distribution")
        lines.append("")
        lines.append("| Metric | Baseline | Target | Delta |")
        lines.append("|---|---|---|---|")
        for diff in report.token_diffs:
            rel = (
                f"{diff.relative_delta_pct:+.1f}%"
                if diff.relative_delta_pct is not None
                else "N/A"
            )
            lines.append(
                f"| {diff.metric_name} | {diff.baseline_value:.1f} "
                f"| {diff.target_value:.1f} | {rel} |"
            )
        lines.append("")

        return "\n".join(lines)

    def _make_diff(
        self,
        name: str,
        baseline: float,
        target: float,
        higher_is_better: Optional[bool],
    ) -> MetricDiff:
        absolute = target - baseline
        if baseline != 0:
            relative = (absolute / abs(baseline)) * 100.0
        else:
            relative = None

        if higher_is_better is None:
            direction = ChangeDirection.UNCHANGED
        elif relative is None:
            direction = ChangeDirection.UNCHANGED
        elif abs(relative) <= self._threshold:
            direction = ChangeDirection.UNCHANGED
        elif higher_is_better:
            direction = (
                ChangeDirection.IMPROVED if absolute > 0 else ChangeDirection.REGRESSED
            )
        else:
            direction = (
                ChangeDirection.IMPROVED if absolute < 0 else ChangeDirection.REGRESSED
            )

        return MetricDiff(
            metric_name=name,
            baseline_value=baseline,
            target_value=target,
            absolute_delta=absolute,
            relative_delta_pct=relative,
            direction=direction,
        )

    @staticmethod
    def _direction_icon(direction: ChangeDirection) -> str:
        return {
            ChangeDirection.IMPROVED: "✅",
            ChangeDirection.REGRESSED: "❌",
            ChangeDirection.UNCHANGED: "➖",
        }[direction]

    @staticmethod
    def _md_diff_row(label: str, diff: MetricDiff) -> str:
        rel = (
            f"{diff.relative_delta_pct:+.1f}%"
            if diff.relative_delta_pct is not None
            else "N/A"
        )
        icon = DiffReporter._direction_icon(diff.direction)
        return (
            f"| {label} | {diff.baseline_value:.2f} "
            f"| {diff.target_value:.2f} "
            f"| {rel} {icon} |"
        )


def generate_diff_report(
    baseline: BenchmarkData,
    target: BenchmarkData,
    baseline_name: str = "baseline",
    target_name: str = "target",
    regression_threshold_pct: float = 5.0,
) -> dict:
    """Programmatic API for benchmark diff report.

    Args:
        baseline: Reference benchmark data.
        target: Benchmark to compare against baseline.
        baseline_name: Label for baseline.
        target_name: Label for target.
        regression_threshold_pct: Threshold for regression detection.

    Returns:
        Dictionary representation of DiffReport.
    """
    reporter = DiffReporter(regression_threshold_pct=regression_threshold_pct)
    report = reporter.compare(baseline, target, baseline_name, target_name)
    return report.model_dump()

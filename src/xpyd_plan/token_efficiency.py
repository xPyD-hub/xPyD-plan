"""Token efficiency analysis for benchmark data.

Computes per-request and aggregate token throughput metrics to identify
how efficiently GPU instances convert compute time into generated tokens.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class EfficiencyGrade(str, Enum):
    """Token efficiency grade classification."""

    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"


class PerRequestEfficiency(BaseModel):
    """Token efficiency metrics for a single request."""

    request_id: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    output_tokens_per_sec: float = Field(
        ..., description="Output tokens generated per second"
    )
    total_tokens_per_sec: float = Field(
        ..., description="Total tokens processed per second (prompt + output)"
    )
    decode_tokens_per_sec: float = Field(
        ..., description="Decode throughput: output_tokens / (total_latency - ttft) per sec"
    )


class AggregateEfficiency(BaseModel):
    """Aggregate token efficiency statistics."""

    output_tps_mean: float
    output_tps_p50: float
    output_tps_p95: float
    output_tps_p99: float
    output_tps_min: float
    output_tps_max: float

    decode_tps_mean: float
    decode_tps_p50: float
    decode_tps_p95: float
    decode_tps_p99: float

    total_tps_mean: float
    total_tps_p50: float
    total_tps_p95: float

    total_output_tokens: int = Field(
        ..., description="Sum of output tokens across all requests"
    )
    total_prompt_tokens: int = Field(
        ..., description="Sum of prompt tokens across all requests"
    )
    total_tokens: int = Field(
        ..., description="Sum of all tokens (prompt + output)"
    )


class InstanceEfficiency(BaseModel):
    """Per-instance token throughput."""

    output_tokens_per_instance: float = Field(
        ..., description="Total output tokens / total instances"
    )
    output_tokens_per_decode_instance: float = Field(
        ..., description="Total output tokens / decode instances"
    )
    prompt_tokens_per_prefill_instance: float = Field(
        ..., description="Total prompt tokens / prefill instances"
    )


class TokenEfficiencyReport(BaseModel):
    """Complete token efficiency analysis report."""

    aggregate: AggregateEfficiency
    instance_efficiency: InstanceEfficiency
    grade: EfficiencyGrade
    grade_reason: str
    num_requests: int
    per_request: list[PerRequestEfficiency] = Field(
        default_factory=list,
        description="Per-request details (empty unless include_details=True)",
    )


class TokenEfficiencyAnalyzer:
    """Analyze token generation efficiency from benchmark data."""

    def __init__(self, data: BenchmarkData) -> None:
        self._data = data

    def analyze(self, *, include_details: bool = False) -> TokenEfficiencyReport:
        """Run token efficiency analysis.

        Args:
            include_details: If True, include per-request efficiency metrics.

        Returns:
            TokenEfficiencyReport with aggregate and instance-level metrics.
        """
        requests = self._data.requests
        meta = self._data.metadata

        per_request: list[PerRequestEfficiency] = []
        output_tps_list: list[float] = []
        decode_tps_list: list[float] = []
        total_tps_list: list[float] = []

        total_output_tokens = 0
        total_prompt_tokens = 0

        for req in requests:
            total_latency_sec = req.total_latency_ms / 1000.0
            ttft_sec = req.ttft_ms / 1000.0
            decode_time_sec = total_latency_sec - ttft_sec

            total_tokens = req.prompt_tokens + req.output_tokens
            total_output_tokens += req.output_tokens
            total_prompt_tokens += req.prompt_tokens

            output_tps = (
                req.output_tokens / total_latency_sec if total_latency_sec > 0 else 0.0
            )
            total_tps = (
                total_tokens / total_latency_sec if total_latency_sec > 0 else 0.0
            )
            decode_tps = (
                req.output_tokens / decode_time_sec
                if decode_time_sec > 0
                else 0.0
            )

            output_tps_list.append(output_tps)
            decode_tps_list.append(decode_tps)
            total_tps_list.append(total_tps)

            if include_details:
                per_request.append(
                    PerRequestEfficiency(
                        request_id=req.request_id,
                        prompt_tokens=req.prompt_tokens,
                        output_tokens=req.output_tokens,
                        total_tokens=total_tokens,
                        output_tokens_per_sec=round(output_tps, 2),
                        total_tokens_per_sec=round(total_tps, 2),
                        decode_tokens_per_sec=round(decode_tps, 2),
                    )
                )

        output_arr = np.array(output_tps_list)
        decode_arr = np.array(decode_tps_list)
        total_arr = np.array(total_tps_list)

        aggregate = AggregateEfficiency(
            output_tps_mean=round(float(np.mean(output_arr)), 2),
            output_tps_p50=round(float(np.percentile(output_arr, 50)), 2),
            output_tps_p95=round(float(np.percentile(output_arr, 95)), 2),
            output_tps_p99=round(float(np.percentile(output_arr, 99)), 2),
            output_tps_min=round(float(np.min(output_arr)), 2),
            output_tps_max=round(float(np.max(output_arr)), 2),
            decode_tps_mean=round(float(np.mean(decode_arr)), 2),
            decode_tps_p50=round(float(np.percentile(decode_arr, 50)), 2),
            decode_tps_p95=round(float(np.percentile(decode_arr, 95)), 2),
            decode_tps_p99=round(float(np.percentile(decode_arr, 99)), 2),
            total_tps_mean=round(float(np.mean(total_arr)), 2),
            total_tps_p50=round(float(np.percentile(total_arr, 50)), 2),
            total_tps_p95=round(float(np.percentile(total_arr, 95)), 2),
            total_output_tokens=total_output_tokens,
            total_prompt_tokens=total_prompt_tokens,
            total_tokens=total_output_tokens + total_prompt_tokens,
        )

        instance_efficiency = InstanceEfficiency(
            output_tokens_per_instance=round(
                total_output_tokens / meta.total_instances, 2
            ),
            output_tokens_per_decode_instance=round(
                total_output_tokens / meta.num_decode_instances, 2
            ),
            prompt_tokens_per_prefill_instance=round(
                total_prompt_tokens / meta.num_prefill_instances, 2
            ),
        )

        # Grade based on output TPS P50 stability relative to P95
        # High ratio of P95/P50 means inconsistent efficiency
        p50 = aggregate.output_tps_p50
        p95 = aggregate.output_tps_p95
        if p50 > 0:
            ratio = p95 / p50
        else:
            ratio = float("inf")

        if ratio <= 1.3:
            grade = EfficiencyGrade.EXCELLENT
            grade_reason = (
                f"Consistent token throughput (P95/P50 ratio: {ratio:.2f} ≤ 1.3)"
            )
        elif ratio <= 1.8:
            grade = EfficiencyGrade.GOOD
            grade_reason = (
                f"Mostly consistent token throughput (P95/P50 ratio: {ratio:.2f} ≤ 1.8)"
            )
        elif ratio <= 2.5:
            grade = EfficiencyGrade.FAIR
            grade_reason = (
                f"Variable token throughput (P95/P50 ratio: {ratio:.2f} ≤ 2.5)"
            )
        else:
            grade = EfficiencyGrade.POOR
            grade_reason = (
                f"Highly variable token throughput (P95/P50 ratio: {ratio:.2f} > 2.5)"
            )

        return TokenEfficiencyReport(
            aggregate=aggregate,
            instance_efficiency=instance_efficiency,
            grade=grade,
            grade_reason=grade_reason,
            num_requests=len(requests),
            per_request=per_request,
        )


def analyze_token_efficiency(
    data: BenchmarkData,
    *,
    include_details: bool = False,
) -> dict:
    """Programmatic API for token efficiency analysis.

    Args:
        data: Benchmark data to analyze.
        include_details: Include per-request breakdown.

    Returns:
        Dictionary with analysis results.
    """
    analyzer = TokenEfficiencyAnalyzer(data)
    report = analyzer.analyze(include_details=include_details)
    return report.model_dump()

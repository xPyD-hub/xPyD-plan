"""Tail latency analysis for benchmark data."""

from __future__ import annotations

import statistics
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class TailClassification(str, Enum):
    """Classification of tail heaviness based on P99/P50 ratio."""

    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    EXTREME = "extreme"


class TailMetric(BaseModel):
    """Extended percentile metrics for a single latency field."""

    field: str = Field(description="Latency field name (ttft_ms, tpot_ms, total_latency_ms)")
    p50: float = Field(description="50th percentile (median)")
    p90: float = Field(description="90th percentile")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")
    p999: float = Field(description="99.9th percentile")
    p9999: float = Field(description="99.99th percentile")
    tail_ratio_p99: float = Field(description="P99 / P50 ratio")
    tail_ratio_p999: float = Field(description="P99.9 / P50 ratio")
    classification: TailClassification = Field(description="Tail heaviness classification")


class LongTailProfile(BaseModel):
    """Characterization of requests in the P99+ tail."""

    field: str = Field(description="Latency field used for tail selection")
    tail_count: int = Field(description="Number of requests in the tail (P99+)")
    total_count: int = Field(description="Total number of requests")
    avg_prompt_tokens: float = Field(description="Mean prompt tokens of tail requests")
    avg_output_tokens: float = Field(description="Mean output tokens of tail requests")
    median_prompt_tokens: float = Field(description="Median prompt tokens of tail requests")
    median_output_tokens: float = Field(description="Median output tokens of tail requests")
    avg_prompt_tokens_all: float = Field(description="Mean prompt tokens of all requests")
    avg_output_tokens_all: float = Field(description="Mean output tokens of all requests")
    prompt_token_ratio: float = Field(
        description="Ratio of tail avg prompt tokens to overall avg"
    )
    output_token_ratio: float = Field(
        description="Ratio of tail avg output tokens to overall avg"
    )


class TailReport(BaseModel):
    """Complete tail latency analysis report."""

    metrics: list[TailMetric] = Field(description="Extended percentile metrics per latency field")
    long_tail_profiles: list[LongTailProfile] = Field(
        description="Characterization of P99+ tail requests per field"
    )
    worst_tail: Optional[str] = Field(
        default=None, description="Field with the heaviest tail classification"
    )
    total_requests: int = Field(description="Total requests analyzed")


def _classify_tail(ratio: float) -> TailClassification:
    """Classify tail heaviness based on P99/P50 ratio."""
    if ratio < 2.0:
        return TailClassification.LIGHT
    elif ratio < 5.0:
        return TailClassification.MODERATE
    elif ratio < 10.0:
        return TailClassification.HEAVY
    else:
        return TailClassification.EXTREME


_CLASSIFICATION_SEVERITY = {
    TailClassification.LIGHT: 0,
    TailClassification.MODERATE: 1,
    TailClassification.HEAVY: 2,
    TailClassification.EXTREME: 3,
}


class TailAnalyzer:
    """Analyze tail latency behavior in benchmark data."""

    LATENCY_FIELDS = ["ttft_ms", "tpot_ms", "total_latency_ms"]

    def __init__(self, data: BenchmarkData) -> None:
        self._data = data
        self._requests = data.requests

    def analyze(self) -> TailReport:
        """Run full tail latency analysis."""
        metrics: list[TailMetric] = []
        profiles: list[LongTailProfile] = []

        for field in self.LATENCY_FIELDS:
            values = np.array([getattr(r, field) for r in self._requests], dtype=np.float64)
            if len(values) == 0:
                continue

            p50 = float(np.percentile(values, 50))
            p90 = float(np.percentile(values, 90))
            p95 = float(np.percentile(values, 95))
            p99 = float(np.percentile(values, 99))
            p999 = float(np.percentile(values, 99.9))
            p9999 = float(np.percentile(values, 99.99))

            ratio_p99 = p99 / p50 if p50 > 0 else 0.0
            ratio_p999 = p999 / p50 if p50 > 0 else 0.0
            classification = _classify_tail(ratio_p99)

            metrics.append(
                TailMetric(
                    field=field,
                    p50=round(p50, 2),
                    p90=round(p90, 2),
                    p95=round(p95, 2),
                    p99=round(p99, 2),
                    p999=round(p999, 2),
                    p9999=round(p9999, 2),
                    tail_ratio_p99=round(ratio_p99, 3),
                    tail_ratio_p999=round(ratio_p999, 3),
                    classification=classification,
                )
            )

            # Long-tail profile: requests at or above P99
            threshold = p99
            tail_reqs = [r for r in self._requests if getattr(r, field) >= threshold]
            if not tail_reqs:
                continue

            tail_prompt = [r.prompt_tokens for r in tail_reqs]
            tail_output = [r.output_tokens for r in tail_reqs]
            all_prompt = [r.prompt_tokens for r in self._requests]
            all_output = [r.output_tokens for r in self._requests]

            avg_prompt_all = statistics.mean(all_prompt)
            avg_output_all = statistics.mean(all_output)
            avg_prompt_tail = statistics.mean(tail_prompt)
            avg_output_tail = statistics.mean(tail_output)

            profiles.append(
                LongTailProfile(
                    field=field,
                    tail_count=len(tail_reqs),
                    total_count=len(self._requests),
                    avg_prompt_tokens=round(avg_prompt_tail, 1),
                    avg_output_tokens=round(avg_output_tail, 1),
                    median_prompt_tokens=float(statistics.median(tail_prompt)),
                    median_output_tokens=float(statistics.median(tail_output)),
                    avg_prompt_tokens_all=round(avg_prompt_all, 1),
                    avg_output_tokens_all=round(avg_output_all, 1),
                    prompt_token_ratio=round(
                        avg_prompt_tail / avg_prompt_all if avg_prompt_all > 0 else 0.0, 3
                    ),
                    output_token_ratio=round(
                        avg_output_tail / avg_output_all if avg_output_all > 0 else 0.0, 3
                    ),
                )
            )

        # Determine worst tail
        worst_tail = None
        if metrics:
            worst = max(metrics, key=lambda m: _CLASSIFICATION_SEVERITY[m.classification])
            worst_tail = worst.field

        return TailReport(
            metrics=metrics,
            long_tail_profiles=profiles,
            worst_tail=worst_tail,
            total_requests=len(self._requests),
        )


def analyze_tail(data: BenchmarkData) -> dict:
    """Programmatic API: analyze tail latency behavior.

    Args:
        data: Loaded benchmark data.

    Returns:
        Dictionary with tail analysis results.
    """
    analyzer = TailAnalyzer(data)
    report = analyzer.analyze()
    return report.model_dump()

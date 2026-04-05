"""Prefill-Decode imbalance detector.

Analyze multiple benchmarks at different P:D ratios to determine whether the
system is prefill-starved, decode-starved, or balanced.  Sensitivity of TTFT
to prefill instance count and TPOT to decode instance count drives the
classification.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.benchmark_models import BenchmarkData


class ImbalanceLevel(str, Enum):
    """Severity of the detected P:D imbalance."""

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ImbalanceClassification(str, Enum):
    """High-level imbalance classification."""

    PREFILL_STARVED = "prefill_starved"
    DECODE_STARVED = "decode_starved"
    BALANCED = "balanced"
    INSUFFICIENT_DATA = "insufficient_data"


class MetricSensitivity(BaseModel):
    """Sensitivity of a latency metric to instance count changes."""

    metric: str = Field(..., description="Metric name (ttft_p95 or tpot_p95)")
    instance_type: str = Field(..., description="Instance type (prefill or decode)")
    sensitivity_ms_per_instance: float = Field(
        ...,
        description="Average latency change (ms) per additional instance (negative = improvement)",
    )
    data_points: int = Field(..., ge=0, description="Number of data points used")
    r_squared: Optional[float] = Field(
        None, ge=0, le=1, description="R² of the linear fit"
    )


class ImbalanceReport(BaseModel):
    """Full imbalance analysis report."""

    classification: ImbalanceClassification = Field(
        ..., description="Overall imbalance classification"
    )
    level: ImbalanceLevel = Field(..., description="Imbalance severity")
    ttft_sensitivity: Optional[MetricSensitivity] = Field(
        None, description="TTFT sensitivity to prefill instance count"
    )
    tpot_sensitivity: Optional[MetricSensitivity] = Field(
        None, description="TPOT sensitivity to decode instance count"
    )
    sensitivity_ratio: Optional[float] = Field(
        None,
        description="Ratio of TTFT sensitivity magnitude to TPOT sensitivity magnitude",
    )
    recommendation: str = Field(..., description="Actionable recommendation")
    num_benchmarks: int = Field(..., ge=0, description="Number of benchmarks analyzed")


class PDImbalanceDetector:
    """Detect prefill-decode imbalance from multi-ratio benchmark data."""

    def __init__(
        self,
        datasets: Sequence[BenchmarkData],
        *,
        severity_threshold_mild: float = 1.5,
        severity_threshold_moderate: float = 3.0,
        severity_threshold_severe: float = 5.0,
    ) -> None:
        self._datasets = list(datasets)
        self._mild = severity_threshold_mild
        self._moderate = severity_threshold_moderate
        self._severe = severity_threshold_severe

    def analyze(self) -> ImbalanceReport:
        """Run the imbalance analysis."""
        if len(self._datasets) < 2:
            return ImbalanceReport(
                classification=ImbalanceClassification.INSUFFICIENT_DATA,
                level=ImbalanceLevel.NONE,
                recommendation="At least 2 benchmarks with different P:D ratios are required.",
                num_benchmarks=len(self._datasets),
            )

        # Extract per-benchmark stats
        points: list[dict] = []
        for ds in self._datasets:
            meta = ds.metadata
            n_prefill = meta.num_prefill_instances
            n_decode = meta.num_decode_instances
            latencies = ds.requests
            if not latencies:
                continue
            ttft_vals = sorted(r.ttft_ms for r in latencies)
            tpot_vals = sorted(r.tpot_ms for r in latencies)
            p95_idx = max(0, int(len(ttft_vals) * 0.95) - 1)
            points.append(
                {
                    "prefill": n_prefill,
                    "decode": n_decode,
                    "ttft_p95": ttft_vals[p95_idx],
                    "tpot_p95": tpot_vals[p95_idx],
                }
            )

        if len(points) < 2:
            return ImbalanceReport(
                classification=ImbalanceClassification.INSUFFICIENT_DATA,
                level=ImbalanceLevel.NONE,
                recommendation="At least 2 benchmarks with different P:D ratios are required.",
                num_benchmarks=len(self._datasets),
            )

        # Compute sensitivities via linear regression
        ttft_sens = self._compute_sensitivity(
            points, instance_key="prefill", metric_key="ttft_p95"
        )
        tpot_sens = self._compute_sensitivity(
            points, instance_key="decode", metric_key="tpot_p95"
        )

        ttft_result = MetricSensitivity(
            metric="ttft_p95",
            instance_type="prefill",
            sensitivity_ms_per_instance=ttft_sens["slope"],
            data_points=ttft_sens["n"],
            r_squared=ttft_sens["r_squared"],
        )
        tpot_result = MetricSensitivity(
            metric="tpot_p95",
            instance_type="decode",
            sensitivity_ms_per_instance=tpot_sens["slope"],
            data_points=tpot_sens["n"],
            r_squared=tpot_sens["r_squared"],
        )

        # Classification based on sensitivity magnitudes
        ttft_mag = abs(ttft_sens["slope"])
        tpot_mag = abs(tpot_sens["slope"])

        # Avoid division by zero
        if tpot_mag < 1e-9 and ttft_mag < 1e-9:
            classification = ImbalanceClassification.BALANCED
            ratio = 1.0
        elif tpot_mag < 1e-9:
            classification = ImbalanceClassification.PREFILL_STARVED
            ratio = float("inf")
        elif ttft_mag < 1e-9:
            classification = ImbalanceClassification.DECODE_STARVED
            ratio = 0.0
        else:
            ratio = ttft_mag / tpot_mag
            if ratio > self._mild:
                classification = ImbalanceClassification.PREFILL_STARVED
            elif ratio < 1.0 / self._mild:
                classification = ImbalanceClassification.DECODE_STARVED
            else:
                classification = ImbalanceClassification.BALANCED

        # Determine severity
        effective_ratio = ratio if classification == ImbalanceClassification.PREFILL_STARVED else (
            1.0 / ratio if ratio > 0 else float("inf")
        ) if classification == ImbalanceClassification.DECODE_STARVED else 1.0

        if classification == ImbalanceClassification.BALANCED:
            level = ImbalanceLevel.NONE
        elif effective_ratio >= self._severe:
            level = ImbalanceLevel.SEVERE
        elif effective_ratio >= self._moderate:
            level = ImbalanceLevel.MODERATE
        elif effective_ratio >= self._mild:
            level = ImbalanceLevel.MILD
        else:
            level = ImbalanceLevel.NONE

        recommendation = self._make_recommendation(classification, level)

        return ImbalanceReport(
            classification=classification,
            level=level,
            ttft_sensitivity=ttft_result,
            tpot_sensitivity=tpot_result,
            sensitivity_ratio=ratio if ratio != float("inf") else None,
            recommendation=recommendation,
            num_benchmarks=len(self._datasets),
        )

    @staticmethod
    def _compute_sensitivity(
        points: list[dict], *, instance_key: str, metric_key: str
    ) -> dict:
        """Linear regression of metric vs instance count."""
        x = np.array([p[instance_key] for p in points], dtype=float)
        y = np.array([p[metric_key] for p in points], dtype=float)
        n = len(x)

        if n < 2 or np.std(x) < 1e-9:
            return {"slope": 0.0, "r_squared": None, "n": n}

        # OLS
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        slope = float(ss_xy / ss_xx)

        # R²
        y_pred = x_mean + slope * (x - x_mean)  # simplified: intercept = y_mean - slope*x_mean
        y_pred = (y_mean - slope * x_mean) + slope * x
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else None

        return {"slope": slope, "r_squared": r_squared, "n": n}

    @staticmethod
    def _make_recommendation(
        classification: ImbalanceClassification, level: ImbalanceLevel
    ) -> str:
        if classification == ImbalanceClassification.BALANCED:
            return "P:D ratio is well-balanced. No adjustment needed."
        if classification == ImbalanceClassification.INSUFFICIENT_DATA:
            return "Insufficient data to determine imbalance."

        direction = (
            "prefill"
            if classification == ImbalanceClassification.PREFILL_STARVED
            else "decode"
        )
        metric = "TTFT" if direction == "prefill" else "TPOT"

        severity_text = {
            ImbalanceLevel.MILD: "slightly",
            ImbalanceLevel.MODERATE: "moderately",
            ImbalanceLevel.SEVERE: "severely",
            ImbalanceLevel.NONE: "",
        }

        return (
            f"System is {severity_text[level]} {direction}-starved. "
            f"{metric} is highly sensitive to {direction} instance count. "
            f"Consider increasing the number of {direction} instances."
        )


def detect_pd_imbalance(
    benchmark_paths: Sequence[str | Path],
    *,
    severity_threshold_mild: float = 1.5,
    severity_threshold_moderate: float = 3.0,
    severity_threshold_severe: float = 5.0,
) -> ImbalanceReport:
    """Programmatic API: detect P:D imbalance from benchmark files.

    Args:
        benchmark_paths: Paths to benchmark JSON files (different P:D ratios).
        severity_threshold_mild: Sensitivity ratio threshold for mild imbalance.
        severity_threshold_moderate: Threshold for moderate imbalance.
        severity_threshold_severe: Threshold for severe imbalance.

    Returns:
        ImbalanceReport with classification, sensitivities, and recommendation.
    """
    datasets = [load_benchmark_auto(str(p)) for p in benchmark_paths]
    detector = PDImbalanceDetector(
        datasets,
        severity_threshold_mild=severity_threshold_mild,
        severity_threshold_moderate=severity_threshold_moderate,
        severity_threshold_severe=severity_threshold_severe,
    )
    return detector.analyze()

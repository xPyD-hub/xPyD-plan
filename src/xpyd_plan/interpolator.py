"""Performance interpolation model for predicting untested P:D ratios.

Given benchmark data at a few P:D configurations, fits a performance model
per metric and predicts performance at untested ratios. Follows DESIGN_PRINCIPLES:
  - Analyze data, fit/interpolate, build performance model
  - Extrapolate predicted performance for all possible P:D ratios
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class InterpolationMethod(str, Enum):
    """Supported interpolation methods."""

    LINEAR = "linear"
    SPLINE = "spline"


class InterpolationConfidence(str, Enum):
    """Confidence level of an interpolated prediction."""

    HIGH = "high"  # Within range of measured data (interpolation)
    MEDIUM = "medium"  # Near extrapolation (≤20% beyond range)
    LOW = "low"  # Far extrapolation (>20% beyond range)


class PredictedPerformance(BaseModel):
    """Predicted performance for a single P:D ratio."""

    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    total_instances: int = Field(..., ge=2)
    prefill_fraction: float = Field(
        ..., ge=0, le=1, description="Fraction of instances used for prefill"
    )
    ttft_p95_ms: float = Field(..., ge=0, description="Predicted TTFT P95 (ms)")
    tpot_p95_ms: float = Field(..., ge=0, description="Predicted TPOT P95 (ms)")
    total_latency_p95_ms: float = Field(
        ..., ge=0, description="Predicted total latency P95 (ms)"
    )
    throughput_qps: float = Field(..., ge=0, description="Predicted throughput (QPS)")
    confidence: InterpolationConfidence
    is_measured: bool = Field(False, description="Whether this ratio was actually measured")


class InterpolationResult(BaseModel):
    """Result of performance interpolation across P:D ratios."""

    method: InterpolationMethod
    measured_points: int = Field(..., ge=2, description="Number of measured data points used")
    total_instances: int
    predictions: list[PredictedPerformance] = Field(default_factory=list)
    best_predicted: PredictedPerformance | None = Field(
        None, description="Ratio with best predicted throughput"
    )
    notes: list[str] = Field(default_factory=list)


def _compute_percentile(values: list[float], p: float) -> float:
    """Compute percentile from a list of values."""
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _classify_confidence(
    x: float, x_min: float, x_max: float
) -> InterpolationConfidence:
    """Classify confidence based on how far x is from measured range."""
    if x_min <= x <= x_max:
        return InterpolationConfidence.HIGH
    span = x_max - x_min
    if span == 0:
        return InterpolationConfidence.LOW
    if x < x_min:
        dist = x_min - x
    else:
        dist = x - x_max
    ratio = dist / span
    if ratio <= 0.2:
        return InterpolationConfidence.MEDIUM
    return InterpolationConfidence.LOW


class PerformanceInterpolator:
    """Interpolate performance for untested P:D ratios.

    Fits per-metric models (TTFT P95, TPOT P95, total latency P95, QPS)
    as functions of prefill_fraction = num_prefill / total_instances.

    Usage:
        interpolator = PerformanceInterpolator()
        interpolator.fit(datasets)
        result = interpolator.predict(
            target_ratios=[(1, 3), (1, 5), (2, 6)],
            method=InterpolationMethod.LINEAR,
        )
    """

    def __init__(self) -> None:
        self._points: list[dict[str, Any]] = []
        self._total_instances: int | None = None
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, datasets: list[BenchmarkData]) -> None:
        """Fit the interpolation model from benchmark datasets.

        All datasets must have the same total_instances count.
        Need at least 2 datasets at different P:D ratios.

        Args:
            datasets: List of benchmark datasets at different P:D ratios.

        Raises:
            ValueError: If fewer than 2 datasets or inconsistent total instances.
        """
        if len(datasets) < 2:
            raise ValueError("Need at least 2 benchmark datasets for interpolation.")

        total_set = {d.metadata.total_instances for d in datasets}
        if len(total_set) > 1:
            raise ValueError(
                f"All datasets must have the same total_instances. Got: {sorted(total_set)}"
            )

        self._total_instances = datasets[0].metadata.total_instances
        self._points = []

        for ds in datasets:
            meta = ds.metadata
            frac = meta.num_prefill_instances / meta.total_instances
            ttfts = [r.ttft_ms for r in ds.requests]
            tpots = [r.tpot_ms for r in ds.requests]
            totals = [r.total_latency_ms for r in ds.requests]

            self._points.append(
                {
                    "prefill_fraction": frac,
                    "num_prefill": meta.num_prefill_instances,
                    "num_decode": meta.num_decode_instances,
                    "ttft_p95": _compute_percentile(ttfts, 95),
                    "tpot_p95": _compute_percentile(tpots, 95),
                    "total_latency_p95": _compute_percentile(totals, 95),
                    "qps": meta.measured_qps,
                }
            )

        # Sort by prefill fraction for interpolation
        self._points.sort(key=lambda p: p["prefill_fraction"])

        # Deduplicate by fraction (take last if duplicates)
        seen: dict[float, int] = {}
        for i, pt in enumerate(self._points):
            seen[pt["prefill_fraction"]] = i
        self._points = [self._points[i] for i in sorted(seen.values())]

        if len(self._points) < 2:
            raise ValueError(
                "Need at least 2 datasets at different P:D ratios after deduplication."
            )

        self._fitted = True

    def predict(
        self,
        target_ratios: list[tuple[int, int]],
        method: InterpolationMethod = InterpolationMethod.LINEAR,
    ) -> InterpolationResult:
        """Predict performance at target P:D ratios.

        Args:
            target_ratios: List of (num_prefill, num_decode) tuples to predict.
            method: Interpolation method (linear or spline).

        Returns:
            InterpolationResult with predictions for each requested ratio.

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If target ratios have inconsistent total instances.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        assert self._total_instances is not None

        # Validate target ratios
        for p, d in target_ratios:
            if p + d != self._total_instances:
                raise ValueError(
                    f"Ratio ({p}, {d}) has total {p + d}, "
                    f"but fitted data has total_instances={self._total_instances}."
                )

        fracs = np.array([pt["prefill_fraction"] for pt in self._points])
        frac_min, frac_max = float(fracs.min()), float(fracs.max())

        metrics = ["ttft_p95", "tpot_p95", "total_latency_p95", "qps"]
        metric_arrays = {m: np.array([pt[m] for pt in self._points]) for m in metrics}

        # Build measured fraction set for is_measured check
        measured_fracs = {pt["prefill_fraction"] for pt in self._points}

        predictions: list[PredictedPerformance] = []
        notes: list[str] = []

        if method == InterpolationMethod.SPLINE and len(self._points) < 4:
            notes.append(
                "Spline requires ≥4 data points; falling back to linear interpolation."
            )
            method = InterpolationMethod.LINEAR

        for num_p, num_d in target_ratios:
            target_frac = num_p / self._total_instances
            confidence = _classify_confidence(target_frac, frac_min, frac_max)
            is_measured = target_frac in measured_fracs

            predicted: dict[str, float] = {}
            for metric_name in metrics:
                y = metric_arrays[metric_name]
                predicted[metric_name] = float(
                    self._interpolate_1d(fracs, y, target_frac, method)
                )

            # Clamp latencies to >= 0
            for key in ["ttft_p95", "tpot_p95", "total_latency_p95", "qps"]:
                if predicted[key] < 0:
                    predicted[key] = 0.0

            predictions.append(
                PredictedPerformance(
                    num_prefill=num_p,
                    num_decode=num_d,
                    total_instances=self._total_instances,
                    prefill_fraction=target_frac,
                    ttft_p95_ms=predicted["ttft_p95"],
                    tpot_p95_ms=predicted["tpot_p95"],
                    total_latency_p95_ms=predicted["total_latency_p95"],
                    throughput_qps=predicted["qps"],
                    confidence=confidence,
                    is_measured=is_measured,
                )
            )

        # Find best throughput
        best = None
        if predictions:
            best = max(predictions, key=lambda p: p.throughput_qps)

        return InterpolationResult(
            method=method,
            measured_points=len(self._points),
            total_instances=self._total_instances,
            predictions=predictions,
            best_predicted=best,
            notes=notes,
        )

    @staticmethod
    def _interpolate_1d(
        x: np.ndarray,
        y: np.ndarray,
        target: float,
        method: InterpolationMethod,
    ) -> float:
        """Interpolate/extrapolate a single value.

        Args:
            x: Known x values (sorted).
            y: Known y values.
            target: Target x to predict.
            method: Interpolation method.

        Returns:
            Predicted y value.
        """
        if method == InterpolationMethod.SPLINE and len(x) >= 4:
            from scipy.interpolate import CubicSpline

            cs = CubicSpline(x, y, extrapolate=True)
            return float(cs(target))
        else:
            # Linear interpolation / extrapolation
            return float(np.interp(target, x, y))


def interpolate_performance(
    datasets: list[BenchmarkData],
    target_ratios: list[tuple[int, int]] | None = None,
    method: InterpolationMethod = InterpolationMethod.LINEAR,
) -> InterpolationResult:
    """Programmatic API for performance interpolation.

    Args:
        datasets: Benchmark datasets at different P:D ratios (same total instances).
        target_ratios: P:D ratios to predict. If None, predicts all integer ratios.
        method: Interpolation method.

    Returns:
        InterpolationResult with predictions.
    """
    interpolator = PerformanceInterpolator()
    interpolator.fit(datasets)

    total = datasets[0].metadata.total_instances
    if target_ratios is None:
        # Generate all possible integer P:D ratios
        target_ratios = [(p, total - p) for p in range(1, total)]

    return interpolator.predict(target_ratios, method=method)

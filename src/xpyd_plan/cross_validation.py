"""Cross-validation for interpolation model accuracy assessment.

Given multiple benchmark files (different P:D ratios), perform leave-one-out
cross-validation: for each file, train the interpolator on the remaining files,
predict the held-out ratio, and measure prediction error against actual data.

This validates whether the interpolation model generalizes or overfits.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.interpolator import (
    InterpolationMethod,
    PerformanceInterpolator,
    PredictedPerformance,
)


class ErrorMetric(BaseModel):
    """Prediction error for a single held-out ratio."""

    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    actual_ttft_p95_ms: float = Field(..., ge=0)
    predicted_ttft_p95_ms: float = Field(..., ge=0)
    ttft_error_pct: float = Field(..., description="Relative error (%) for TTFT P95")
    actual_tpot_p95_ms: float = Field(..., ge=0)
    predicted_tpot_p95_ms: float = Field(..., ge=0)
    tpot_error_pct: float = Field(..., description="Relative error (%) for TPOT P95")
    actual_total_p95_ms: float = Field(..., ge=0)
    predicted_total_p95_ms: float = Field(..., ge=0)
    total_error_pct: float = Field(..., description="Relative error (%) for total latency P95")
    actual_qps: float = Field(..., ge=0)
    predicted_qps: float = Field(..., ge=0)
    qps_error_pct: float = Field(..., description="Relative error (%) for QPS")


class ModelAccuracy(str, Enum):
    """Overall model accuracy classification."""

    EXCELLENT = "excellent"  # Mean error < 5%
    GOOD = "good"  # Mean error 5-15%
    FAIR = "fair"  # Mean error 15-30%
    POOR = "poor"  # Mean error > 30%


class CrossValidationReport(BaseModel):
    """Result of leave-one-out cross-validation."""

    method: InterpolationMethod
    num_folds: int = Field(..., ge=2, description="Number of LOO folds (= number of benchmarks)")
    fold_errors: list[ErrorMetric] = Field(default_factory=list)
    mean_ttft_error_pct: float = Field(..., description="Mean absolute TTFT error (%)")
    mean_tpot_error_pct: float = Field(..., description="Mean absolute TPOT error (%)")
    mean_total_error_pct: float = Field(..., description="Mean absolute total latency error (%)")
    mean_qps_error_pct: float = Field(..., description="Mean absolute QPS error (%)")
    overall_mean_error_pct: float = Field(
        ..., description="Grand mean error across all metrics (%)"
    )
    worst_fold_error_pct: float = Field(
        ..., description="Worst single-fold mean error (%)"
    )
    accuracy: ModelAccuracy
    recommendations: list[str] = Field(default_factory=list)


def _relative_error(actual: float, predicted: float) -> float:
    """Compute relative error in percent. Returns 0 if actual is 0."""
    if actual == 0:
        return 0.0
    return abs(predicted - actual) / actual * 100.0


def _compute_p95(values: list[float]) -> float:
    """Compute P95 of a list of values."""
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def _classify_accuracy(mean_error: float) -> ModelAccuracy:
    """Classify accuracy based on mean error percentage."""
    if mean_error < 5.0:
        return ModelAccuracy.EXCELLENT
    if mean_error < 15.0:
        return ModelAccuracy.GOOD
    if mean_error < 30.0:
        return ModelAccuracy.FAIR
    return ModelAccuracy.POOR


class CrossValidator:
    """Leave-one-out cross-validation for interpolation models."""

    def __init__(
        self,
        benchmark_data: list[BenchmarkData],
        method: InterpolationMethod = InterpolationMethod.LINEAR,
    ) -> None:
        if len(benchmark_data) < 3:
            msg = "Cross-validation requires at least 3 benchmark files"
            raise ValueError(msg)
        self._data = benchmark_data
        self._method = method

    def validate(self) -> CrossValidationReport:
        """Run leave-one-out cross-validation."""
        fold_errors: list[ErrorMetric] = []

        for i in range(len(self._data)):
            held_out = self._data[i]
            training = [d for j, d in enumerate(self._data) if j != i]

            # Get actual values from held-out benchmark
            requests = held_out.requests
            if not requests:
                continue

            actual_ttft = _compute_p95([r.ttft_ms for r in requests])
            actual_tpot = _compute_p95([r.tpot_ms for r in requests])
            actual_total = _compute_p95([r.total_latency_ms for r in requests])
            actual_qps = held_out.metadata.measured_qps

            held_prefill = held_out.metadata.num_prefill_instances
            held_decode = held_out.metadata.num_decode_instances

            # Train interpolator on remaining data
            interpolator = PerformanceInterpolator()
            try:
                interpolator.fit(training)
            except (ValueError, RuntimeError):
                continue

            target_ratio = [(held_prefill, held_decode)]
            try:
                result = interpolator.predict(target_ratio, method=self._method)
            except (ValueError, RuntimeError):
                continue

            # Find prediction for the held-out ratio
            prediction = self._find_prediction(result.predictions, held_prefill, held_decode)

            if prediction is None:
                # Ratio was outside interpolator's range or couldn't be predicted
                continue

            error = ErrorMetric(
                num_prefill=held_prefill,
                num_decode=held_decode,
                actual_ttft_p95_ms=actual_ttft,
                predicted_ttft_p95_ms=prediction.ttft_p95_ms,
                ttft_error_pct=_relative_error(actual_ttft, prediction.ttft_p95_ms),
                actual_tpot_p95_ms=actual_tpot,
                predicted_tpot_p95_ms=prediction.tpot_p95_ms,
                tpot_error_pct=_relative_error(actual_tpot, prediction.tpot_p95_ms),
                actual_total_p95_ms=actual_total,
                predicted_total_p95_ms=prediction.total_latency_p95_ms,
                total_error_pct=_relative_error(actual_total, prediction.total_latency_p95_ms),
                actual_qps=actual_qps,
                predicted_qps=prediction.throughput_qps,
                qps_error_pct=_relative_error(actual_qps, prediction.throughput_qps),
            )
            fold_errors.append(error)

        if not fold_errors:
            msg = "No valid folds produced — check benchmark data compatibility"
            raise ValueError(msg)

        # Aggregate errors
        mean_ttft = float(np.mean([e.ttft_error_pct for e in fold_errors]))
        mean_tpot = float(np.mean([e.tpot_error_pct for e in fold_errors]))
        mean_total = float(np.mean([e.total_error_pct for e in fold_errors]))
        mean_qps = float(np.mean([e.qps_error_pct for e in fold_errors]))
        overall_mean = float(np.mean([mean_ttft, mean_tpot, mean_total, mean_qps]))

        # Worst fold: highest mean error across all metrics for a single fold
        worst_fold = max(
            float(np.mean([e.ttft_error_pct, e.tpot_error_pct, e.total_error_pct, e.qps_error_pct]))
            for e in fold_errors
        )

        accuracy = _classify_accuracy(overall_mean)

        recommendations = self._generate_recommendations(
            accuracy, fold_errors, mean_ttft, mean_tpot, mean_total, mean_qps
        )

        return CrossValidationReport(
            method=self._method,
            num_folds=len(fold_errors),
            fold_errors=fold_errors,
            mean_ttft_error_pct=round(mean_ttft, 2),
            mean_tpot_error_pct=round(mean_tpot, 2),
            mean_total_error_pct=round(mean_total, 2),
            mean_qps_error_pct=round(mean_qps, 2),
            overall_mean_error_pct=round(overall_mean, 2),
            worst_fold_error_pct=round(worst_fold, 2),
            accuracy=accuracy,
            recommendations=recommendations,
        )

    @staticmethod
    def _find_prediction(
        predictions: list[PredictedPerformance],
        num_prefill: int,
        num_decode: int,
    ) -> PredictedPerformance | None:
        """Find the prediction matching the held-out ratio."""
        for p in predictions:
            if p.num_prefill == num_prefill and p.num_decode == num_decode:
                return p
        return None

    @staticmethod
    def _generate_recommendations(
        accuracy: ModelAccuracy,
        fold_errors: list[ErrorMetric],
        mean_ttft: float,
        mean_tpot: float,
        mean_total: float,
        mean_qps: float,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs: list[str] = []

        if accuracy == ModelAccuracy.POOR:
            recs.append(
                "Model accuracy is poor. Consider adding more benchmark data points "
                "or switching interpolation method."
            )
        elif accuracy == ModelAccuracy.FAIR:
            recs.append(
                "Model accuracy is fair. Additional benchmark points at intermediate "
                "P:D ratios would likely improve predictions."
            )

        # Check if specific metrics are much worse
        metrics = {
            "TTFT": mean_ttft, "TPOT": mean_tpot,
            "total_latency": mean_total, "QPS": mean_qps,
        }
        worst_metric = max(metrics, key=metrics.get)  # type: ignore[arg-type]
        best_metric = min(metrics, key=metrics.get)  # type: ignore[arg-type]

        if metrics[worst_metric] > 2 * metrics[best_metric] and metrics[best_metric] > 0:
            recs.append(
                f"{worst_metric} has significantly higher prediction error than {best_metric}. "
                f"The interpolation model may not capture {worst_metric} behavior well."
            )

        if len(fold_errors) < 5:
            recs.append(
                f"Only {len(fold_errors)} data points available. More benchmarks would "
                f"yield more reliable cross-validation results."
            )

        if not recs:
            recs.append("Model predictions are reliable for the tested P:D ratio range.")

        return recs


def cross_validate(
    benchmark_paths: list[str],
    method: str = "linear",
) -> dict:
    """Programmatic API for cross-validation.

    Args:
        benchmark_paths: Paths to benchmark JSON files (at least 3).
        method: Interpolation method ("linear" or "spline").

    Returns:
        Dictionary representation of CrossValidationReport.
    """
    data = [load_benchmark_auto(p) for p in benchmark_paths]
    interp_method = InterpolationMethod(method)
    validator = CrossValidator(data, method=interp_method)
    report = validator.validate()
    return report.model_dump()

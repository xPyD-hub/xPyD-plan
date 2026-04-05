"""QPS-Latency Curve Fitting — fit QPS→latency curves and predict latency at untested QPS levels.

Given multiple benchmark files at different QPS levels, fit regression curves
(linear, polynomial, exponential) to model the relationship between offered load
and latency, then predict latency at arbitrary QPS values and find maximum
sustainable QPS for given SLA constraints.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class FitMethod(str, Enum):
    """Curve fitting method."""

    LINEAR = "linear"
    POLY = "poly"
    EXP = "exp"
    AUTO = "auto"


class Confidence(str, Enum):
    """Prediction confidence level."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FitResult(BaseModel):
    """Result of fitting a QPS→latency curve for one metric."""

    metric: str = Field(..., description="Metric name (e.g., ttft_p95_ms)")
    method: str = Field(..., description="Fit method used (linear, poly, exp)")
    coefficients: list[float] = Field(..., description="Fit coefficients")
    residual_error: float = Field(..., ge=0, description="Mean squared error of the fit")
    r_squared: float = Field(..., description="Coefficient of determination")
    qps_range: tuple[float, float] = Field(..., description="(min_qps, max_qps) in training data")


class QPSPrediction(BaseModel):
    """Predicted latency at a given QPS."""

    qps: float = Field(..., gt=0)
    metric: str
    predicted_latency_ms: float = Field(..., ge=0)
    confidence: Confidence
    fit_method: str


class MaxSustainableQPS(BaseModel):
    """Maximum QPS that stays within SLA for a given metric."""

    metric: str
    max_qps: float = Field(..., gt=0, description="Maximum sustainable QPS")
    sla_threshold_ms: float = Field(..., gt=0)
    predicted_latency_at_max_ms: float = Field(..., ge=0)
    confidence: Confidence


class QPSCurveReport(BaseModel):
    """Complete QPS-latency curve fitting report."""

    fits: list[FitResult] = Field(default_factory=list)
    predictions: list[QPSPrediction] = Field(default_factory=list)
    max_sustainable: list[MaxSustainableQPS] = Field(default_factory=list)
    recommendation: str = Field(..., description="Human-readable recommendation")


class QPSCurveFitter:
    """Fit QPS→latency curves from benchmark data at multiple QPS levels.

    Supports linear, polynomial (degree 2), and exponential fits.
    When method is AUTO, selects the best fit by lowest residual error.
    """

    def __init__(
        self,
        method: FitMethod = FitMethod.AUTO,
        percentile: float = 95.0,
    ) -> None:
        if not 1.0 <= percentile <= 100.0:
            msg = f"percentile must be between 1 and 100, got {percentile}"
            raise ValueError(msg)
        self.method = method
        self.percentile = percentile

    def fit(
        self,
        benchmarks: list[BenchmarkData],
        predict_qps: list[float] | None = None,
        sla_ttft_ms: float | None = None,
        sla_tpot_ms: float | None = None,
        sla_total_ms: float | None = None,
    ) -> QPSCurveReport:
        """Fit QPS→latency curves and optionally predict/find max sustainable QPS.

        Args:
            benchmarks: At least 2 benchmarks at different QPS levels.
            predict_qps: QPS values to predict latency for.
            sla_ttft_ms: TTFT SLA threshold in ms.
            sla_tpot_ms: TPOT SLA threshold in ms.
            sla_total_ms: Total latency SLA threshold in ms.

        Returns:
            QPSCurveReport with fits, predictions, and max sustainable QPS.
        """
        if len(benchmarks) < 2:
            msg = "Need at least 2 benchmark datasets"
            raise ValueError(msg)

        # Extract QPS and latency percentiles
        data_points = self._extract_data_points(benchmarks)
        qps_values = np.array([d[0] for d in data_points])

        if len(set(qps_values)) < 2:
            msg = "Need benchmarks with at least 2 distinct measured_qps values"
            raise ValueError(msg)

        metrics = ["ttft", "tpot", "total_latency"]
        metric_idx = {"ttft": 1, "tpot": 2, "total_latency": 3}

        fits: list[FitResult] = []
        predictions: list[QPSPrediction] = []
        max_sustainable: list[MaxSustainableQPS] = []

        sla_map = {
            "ttft": sla_ttft_ms,
            "tpot": sla_tpot_ms,
            "total_latency": sla_total_ms,
        }

        qps_min, qps_max = float(qps_values.min()), float(qps_values.max())

        for metric_name in metrics:
            latency_values = np.array([d[metric_idx[metric_name]] for d in data_points])
            metric_label = f"{metric_name}_p{int(self.percentile)}_ms"

            # Fit curve
            fit_result = self._fit_curve(qps_values, latency_values, metric_label)
            fits.append(fit_result)

            # Predictions
            if predict_qps:
                for qps in predict_qps:
                    pred_val = self._predict(fit_result, qps)
                    confidence = self._classify_confidence(qps, qps_min, qps_max)
                    predictions.append(
                        QPSPrediction(
                            qps=qps,
                            metric=metric_label,
                            predicted_latency_ms=round(max(0.0, pred_val), 2),
                            confidence=confidence,
                            fit_method=fit_result.method,
                        )
                    )

            # Max sustainable QPS
            sla_threshold = sla_map.get(metric_name)
            if sla_threshold is not None:
                max_qps_result = self._find_max_sustainable(
                    fit_result, sla_threshold, qps_min, qps_max
                )
                if max_qps_result is not None:
                    max_sustainable.append(max_qps_result)

        recommendation = self._build_recommendation(fits, max_sustainable)

        return QPSCurveReport(
            fits=fits,
            predictions=predictions,
            max_sustainable=max_sustainable,
            recommendation=recommendation,
        )

    def _extract_data_points(
        self, benchmarks: list[BenchmarkData]
    ) -> list[tuple[float, float, float, float]]:
        """Extract (qps, ttft_pN, tpot_pN, total_pN) from each benchmark."""
        points = []
        sorted_benchmarks = sorted(benchmarks, key=lambda b: b.metadata.measured_qps)

        for bench in sorted_benchmarks:
            qps = bench.metadata.measured_qps
            ttfts = [r.ttft_ms for r in bench.requests]
            tpots = [r.tpot_ms for r in bench.requests]
            totals = [r.total_latency_ms for r in bench.requests]

            points.append((
                qps,
                float(np.percentile(ttfts, self.percentile)),
                float(np.percentile(tpots, self.percentile)),
                float(np.percentile(totals, self.percentile)),
            ))

        return points

    def _fit_curve(
        self, qps: np.ndarray, latency: np.ndarray, metric: str
    ) -> FitResult:
        """Fit a curve to QPS→latency data, selecting best method if AUTO."""
        candidates: list[FitMethod] = (
            [FitMethod.LINEAR, FitMethod.POLY, FitMethod.EXP]
            if self.method == FitMethod.AUTO
            else [self.method]
        )

        best_fit: FitResult | None = None

        for method in candidates:
            try:
                result = self._fit_single(qps, latency, metric, method)
                if best_fit is None or result.residual_error < best_fit.residual_error:
                    best_fit = result
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                continue

        if best_fit is None:
            # Fallback to linear
            best_fit = self._fit_single(qps, latency, metric, FitMethod.LINEAR)

        return best_fit

    def _fit_single(
        self, qps: np.ndarray, latency: np.ndarray, metric: str, method: FitMethod
    ) -> FitResult:
        """Fit a single method."""
        if method == FitMethod.LINEAR:
            coeffs = np.polyfit(qps, latency, 1)
            predicted = np.polyval(coeffs, qps)
        elif method == FitMethod.POLY:
            coeffs = np.polyfit(qps, latency, 2)
            predicted = np.polyval(coeffs, qps)
        elif method == FitMethod.EXP:
            # Fit log(latency) = a*qps + b → latency = exp(b) * exp(a*qps)
            log_latency = np.log(np.maximum(latency, 1e-10))
            coeffs_log = np.polyfit(qps, log_latency, 1)
            predicted = np.exp(np.polyval(coeffs_log, qps))
            coeffs = coeffs_log.tolist()
            # Store as [a, b] where latency = exp(b) * exp(a*qps)
            coeffs_list = [float(c) for c in coeffs]
        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

        if method != FitMethod.EXP:
            coeffs_list = [float(c) for c in coeffs]

        mse = float(np.mean((predicted - latency) ** 2))
        ss_res = float(np.sum((latency - predicted) ** 2))
        ss_tot = float(np.sum((latency - np.mean(latency)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return FitResult(
            metric=metric,
            method=method.value,
            coefficients=coeffs_list,
            residual_error=round(mse, 4),
            r_squared=round(r_squared, 6),
            qps_range=(float(qps.min()), float(qps.max())),
        )

    def _predict(self, fit: FitResult, qps: float) -> float:
        """Predict latency at a given QPS using a fitted model."""
        coeffs = fit.coefficients
        if fit.method == "exp":
            return float(np.exp(coeffs[1]) * np.exp(coeffs[0] * qps))
        else:
            return float(np.polyval(coeffs, qps))

    def _classify_confidence(
        self, qps: float, qps_min: float, qps_max: float
    ) -> Confidence:
        """Classify prediction confidence based on QPS range."""
        if qps_min <= qps <= qps_max:
            return Confidence.HIGH
        qps_range = qps_max - qps_min
        if qps_range <= 0:
            return Confidence.LOW
        if qps < qps_min:
            distance = (qps_min - qps) / qps_range
        else:
            distance = (qps - qps_max) / qps_range
        if distance <= 0.2:
            return Confidence.MEDIUM
        return Confidence.LOW

    def _find_max_sustainable(
        self,
        fit: FitResult,
        sla_threshold: float,
        qps_min: float,
        qps_max: float,
    ) -> MaxSustainableQPS | None:
        """Find maximum QPS where predicted latency <= SLA threshold.

        Searches from qps_max down to 0 in small steps.
        """
        # Search range: 0 to 2× max observed QPS
        search_max = qps_max * 2.0
        steps = 1000
        step_size = search_max / steps

        max_qps: float | None = None
        predicted_at_max: float = 0.0

        for i in range(steps, -1, -1):
            test_qps = step_size * i
            if test_qps <= 0:
                continue
            pred = self._predict(fit, test_qps)
            if pred <= sla_threshold:
                max_qps = test_qps
                predicted_at_max = pred
                break

        if max_qps is None:
            return None

        confidence = self._classify_confidence(max_qps, qps_min, qps_max)

        return MaxSustainableQPS(
            metric=fit.metric,
            max_qps=round(max_qps, 2),
            sla_threshold_ms=sla_threshold,
            predicted_latency_at_max_ms=round(max(0.0, predicted_at_max), 2),
            confidence=confidence,
        )

    def _build_recommendation(
        self,
        fits: list[FitResult],
        max_sustainable: list[MaxSustainableQPS],
    ) -> str:
        """Generate human-readable recommendation."""
        if not max_sustainable:
            best_r2 = max((f.r_squared for f in fits), default=0)
            return (
                f"Curve fitting complete (best R²={best_r2:.3f}). "
                f"No SLA thresholds provided — specify --sla-ttft, --sla-tpot, "
                f"or --sla-total to find maximum sustainable QPS."
            )

        min_max = min(max_sustainable, key=lambda m: m.max_qps)
        return (
            f"Maximum sustainable QPS: {min_max.max_qps:.1f} "
            f"(limited by {min_max.metric}, SLA={min_max.sla_threshold_ms:.0f}ms). "
            f"Confidence: {min_max.confidence.value}."
        )


def fit_qps_curve(
    benchmarks: list[BenchmarkData],
    method: str = "auto",
    percentile: float = 95.0,
    predict_qps: list[float] | None = None,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_ms: float | None = None,
) -> dict:
    """Programmatic API for QPS-latency curve fitting.

    Args:
        benchmarks: At least 2 benchmarks at different QPS levels.
        method: Fit method — "auto", "linear", "poly", "exp".
        percentile: Latency percentile to fit (default 95).
        predict_qps: QPS values to predict latency for.
        sla_ttft_ms: TTFT SLA threshold in ms.
        sla_tpot_ms: TPOT SLA threshold in ms.
        sla_total_ms: Total latency SLA threshold in ms.

    Returns:
        Dictionary representation of QPSCurveReport.
    """
    fitter = QPSCurveFitter(
        method=FitMethod(method),
        percentile=percentile,
    )
    report = fitter.fit(
        benchmarks,
        predict_qps=predict_qps,
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_ms=sla_total_ms,
    )
    return report.model_dump()

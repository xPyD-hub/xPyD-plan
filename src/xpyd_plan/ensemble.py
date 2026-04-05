"""Latency Prediction Ensemble.

Combine multiple prediction methods (interpolation, regression, QPS curve)
with confidence-weighted averaging and disagreement detection.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.interpolator import (
    InterpolationMethod,
    PerformanceInterpolator,
)
from xpyd_plan.qps_curve import FitMethod, QPSCurveFitter
from xpyd_plan.regression import RegressionAnalyzer


class DisagreementLevel(str, Enum):
    """Level of disagreement between prediction methods."""

    NONE = "none"  # CV < 10%
    LOW = "low"  # CV 10-25%
    MODERATE = "moderate"  # CV 25-50%
    HIGH = "high"  # CV > 50%


class MethodPrediction(BaseModel):
    """Prediction from a single method."""

    method: str = Field(..., description="Method name (interpolation, regression, qps_curve)")
    ttft_p95_ms: float | None = Field(None, description="Predicted TTFT P95 (ms)")
    tpot_p95_ms: float | None = Field(None, description="Predicted TPOT P95 (ms)")
    total_latency_p95_ms: float | None = Field(
        None, description="Predicted total latency P95 (ms)"
    )
    throughput_qps: float | None = Field(None, description="Predicted throughput (QPS)")
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence weight (0-1, based on R² or accuracy)"
    )
    available: bool = Field(True, description="Whether this method produced a prediction")
    error: str | None = Field(None, description="Error message if method failed")


class EnsemblePrediction(BaseModel):
    """Ensemble-averaged prediction."""

    ttft_p95_ms: float | None = Field(None, description="Ensemble TTFT P95 (ms)")
    tpot_p95_ms: float | None = Field(None, description="Ensemble TPOT P95 (ms)")
    total_latency_p95_ms: float | None = Field(
        None, description="Ensemble total latency P95 (ms)"
    )
    throughput_qps: float | None = Field(None, description="Ensemble throughput (QPS)")
    ttft_disagreement: DisagreementLevel = Field(DisagreementLevel.NONE)
    tpot_disagreement: DisagreementLevel = Field(DisagreementLevel.NONE)
    total_latency_disagreement: DisagreementLevel = Field(DisagreementLevel.NONE)
    qps_disagreement: DisagreementLevel = Field(DisagreementLevel.NONE)
    num_methods: int = Field(..., ge=0, description="Number of methods that contributed")


class EnsembleReport(BaseModel):
    """Complete ensemble prediction report."""

    method_predictions: list[MethodPrediction] = Field(default_factory=list)
    ensemble: EnsemblePrediction
    recommendation: str = Field(..., description="Human-readable recommendation")


def _classify_disagreement(cv: float) -> DisagreementLevel:
    """Classify disagreement level based on coefficient of variation."""
    if cv < 0.10:
        return DisagreementLevel.NONE
    if cv < 0.25:
        return DisagreementLevel.LOW
    if cv < 0.50:
        return DisagreementLevel.MODERATE
    return DisagreementLevel.HIGH


def _weighted_average(
    values: list[float], weights: list[float]
) -> float:
    """Compute weighted average."""
    total_weight = sum(weights)
    if total_weight == 0:
        return float(np.mean(values))
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def _cv(values: list[float]) -> float:
    """Coefficient of variation."""
    if len(values) < 2:
        return 0.0
    mean = np.mean(values)
    if mean == 0:
        return 0.0
    return float(np.std(values, ddof=0) / abs(mean))


class EnsemblePredictor:
    """Combine predictions from interpolation, regression, and QPS curve fitting.

    Each method's prediction is weighted by its confidence score (R² or accuracy).
    Disagreement between methods is detected and reported.
    """

    def __init__(self, percentile: float = 95.0) -> None:
        if not 1.0 <= percentile <= 100.0:
            msg = f"percentile must be between 1 and 100, got {percentile}"
            raise ValueError(msg)
        self.percentile = percentile

    def predict(
        self,
        benchmarks: list[BenchmarkData],
        predict_qps: float | None = None,
        predict_ratio: tuple[int, int] | None = None,
        sla_ttft_ms: float | None = None,
        sla_tpot_ms: float | None = None,
        sla_total_ms: float | None = None,
    ) -> EnsembleReport:
        """Generate ensemble prediction from multiple methods.

        Args:
            benchmarks: Benchmark datasets (at least 2).
            predict_qps: QPS level to predict latency for (used by QPS curve fitter).
            predict_ratio: (num_prefill, num_decode) to predict (used by interpolator).
            sla_ttft_ms: TTFT SLA threshold in ms.
            sla_tpot_ms: TPOT SLA threshold in ms.
            sla_total_ms: Total latency SLA threshold in ms.

        Returns:
            EnsembleReport with per-method and ensemble predictions.
        """
        if len(benchmarks) < 2:
            msg = "Need at least 2 benchmark datasets"
            raise ValueError(msg)

        method_preds: list[MethodPrediction] = []

        # 1. Interpolation (needs same total_instances, different P:D ratios)
        method_preds.append(
            self._try_interpolation(benchmarks, predict_ratio)
        )

        # 2. Regression (uses first benchmark for token-latency relationship)
        method_preds.append(self._try_regression(benchmarks))

        # 3. QPS curve fitting (needs different QPS levels)
        method_preds.append(
            self._try_qps_curve(benchmarks, predict_qps, sla_ttft_ms, sla_tpot_ms, sla_total_ms)
        )

        # Compute ensemble
        ensemble = self._compute_ensemble(method_preds)

        # Generate recommendation
        recommendation = self._generate_recommendation(method_preds, ensemble)

        return EnsembleReport(
            method_predictions=method_preds,
            ensemble=ensemble,
            recommendation=recommendation,
        )

    def _try_interpolation(
        self,
        benchmarks: list[BenchmarkData],
        predict_ratio: tuple[int, int] | None,
    ) -> MethodPrediction:
        """Try interpolation method."""
        try:
            interpolator = PerformanceInterpolator()
            interpolator.fit(benchmarks)

            if predict_ratio is None:
                # Use the midpoint ratio
                total = benchmarks[0].metadata.total_instances
                mid_p = total // 2
                mid_d = total - mid_p
                predict_ratio = (mid_p, mid_d)

            result = interpolator.predict(
                target_ratios=[predict_ratio],
                method=InterpolationMethod.LINEAR,
            )

            if not result.predictions:
                return MethodPrediction(
                    method="interpolation",
                    confidence=0.0,
                    available=False,
                    error="No predictions produced",
                )

            pred = result.predictions[0]
            # Confidence based on whether it's interpolation vs extrapolation
            conf_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
            confidence = conf_map.get(pred.confidence.value, 0.5)

            return MethodPrediction(
                method="interpolation",
                ttft_p95_ms=pred.ttft_p95_ms,
                tpot_p95_ms=pred.tpot_p95_ms,
                total_latency_p95_ms=pred.total_latency_p95_ms,
                throughput_qps=pred.throughput_qps,
                confidence=confidence,
            )
        except Exception as e:
            return MethodPrediction(
                method="interpolation",
                confidence=0.0,
                available=False,
                error=str(e),
            )

    def _try_regression(
        self,
        benchmarks: list[BenchmarkData],
    ) -> MethodPrediction:
        """Try regression method (token-latency relationship)."""
        try:
            # Use the benchmark with the most requests for best regression fit
            best_bm = max(benchmarks, key=lambda b: len(b.requests))
            analyzer = RegressionAnalyzer(best_bm)
            report = analyzer.analyze()

            if not report.fits:
                return MethodPrediction(
                    method="regression",
                    confidence=0.0,
                    available=False,
                    error="No regression fits produced",
                )

            # Use mean R² as confidence
            r2_values = [f.r_squared for f in report.fits]
            confidence = float(np.mean(r2_values))
            confidence = max(0.0, min(1.0, confidence))

            # Compute predicted P95 using the regression model
            # For each metric, use the P95 token count from the data
            requests = best_bm.requests
            prompt_tokens = sorted([r.prompt_tokens for r in requests])
            output_tokens = sorted([r.output_tokens for r in requests])
            p95_idx = int(len(requests) * 0.95)
            p95_prompt = prompt_tokens[min(p95_idx, len(prompt_tokens) - 1)]
            p95_output = output_tokens[min(p95_idx, len(output_tokens) - 1)]

            # Predict using fits
            ttft_fit = next((f for f in report.fits if f.target == "ttft_ms"), None)
            tpot_fit = next((f for f in report.fits if f.target == "tpot_ms"), None)
            total_fit = next(
                (f for f in report.fits if f.target == "total_latency_ms"), None
            )

            ttft_pred = None
            tpot_pred = None
            total_pred = None

            if ttft_fit:
                ttft_pred = ttft_fit.slope * p95_prompt + ttft_fit.intercept
                ttft_pred = max(0.0, ttft_pred)
            if tpot_fit:
                tpot_pred = tpot_fit.slope * p95_output + tpot_fit.intercept
                tpot_pred = max(0.0, tpot_pred)
            if total_fit:
                total_pred = total_fit.slope * (p95_prompt + p95_output) + total_fit.intercept
                total_pred = max(0.0, total_pred)

            return MethodPrediction(
                method="regression",
                ttft_p95_ms=ttft_pred,
                tpot_p95_ms=tpot_pred,
                total_latency_p95_ms=total_pred,
                throughput_qps=None,  # Regression doesn't predict QPS
                confidence=confidence,
            )
        except Exception as e:
            return MethodPrediction(
                method="regression",
                confidence=0.0,
                available=False,
                error=str(e),
            )

    def _try_qps_curve(
        self,
        benchmarks: list[BenchmarkData],
        predict_qps: float | None,
        sla_ttft_ms: float | None,
        sla_tpot_ms: float | None,
        sla_total_ms: float | None,
    ) -> MethodPrediction:
        """Try QPS curve fitting method."""
        try:
            fitter = QPSCurveFitter(method=FitMethod.AUTO, percentile=self.percentile)

            qps_list = [predict_qps] if predict_qps else None
            report = fitter.fit(
                benchmarks,
                predict_qps=qps_list,
                sla_ttft_ms=sla_ttft_ms,
                sla_tpot_ms=sla_tpot_ms,
                sla_total_ms=sla_total_ms,
            )

            if not report.fits:
                return MethodPrediction(
                    method="qps_curve",
                    confidence=0.0,
                    available=False,
                    error="No QPS curve fits produced",
                )

            r2_values = [f.r_squared for f in report.fits]
            confidence = float(np.mean(r2_values))
            confidence = max(0.0, min(1.0, confidence))

            ttft_pred = None
            tpot_pred = None
            total_pred = None

            if report.predictions:
                for pred in report.predictions:
                    metric = pred.metric
                    if "ttft" in metric:
                        ttft_pred = pred.predicted_latency_ms
                    elif "tpot" in metric:
                        tpot_pred = pred.predicted_latency_ms
                    elif "total" in metric:
                        total_pred = pred.predicted_latency_ms

            return MethodPrediction(
                method="qps_curve",
                ttft_p95_ms=ttft_pred,
                tpot_p95_ms=tpot_pred,
                total_latency_p95_ms=total_pred,
                throughput_qps=None,
                confidence=confidence,
            )
        except Exception as e:
            return MethodPrediction(
                method="qps_curve",
                confidence=0.0,
                available=False,
                error=str(e),
            )

    def _compute_ensemble(
        self, method_preds: list[MethodPrediction]
    ) -> EnsemblePrediction:
        """Compute confidence-weighted ensemble prediction."""
        available = [m for m in method_preds if m.available]

        def _agg(attr: str) -> tuple[float | None, DisagreementLevel]:
            vals = []
            weights = []
            for m in available:
                v = getattr(m, attr)
                if v is not None:
                    vals.append(v)
                    weights.append(m.confidence)
            if not vals:
                return None, DisagreementLevel.NONE
            avg = _weighted_average(vals, weights)
            disagreement = _classify_disagreement(_cv(vals))
            return avg, disagreement

        ttft, ttft_d = _agg("ttft_p95_ms")
        tpot, tpot_d = _agg("tpot_p95_ms")
        total, total_d = _agg("total_latency_p95_ms")
        qps, qps_d = _agg("throughput_qps")

        return EnsemblePrediction(
            ttft_p95_ms=ttft,
            tpot_p95_ms=tpot,
            total_latency_p95_ms=total,
            throughput_qps=qps,
            ttft_disagreement=ttft_d,
            tpot_disagreement=tpot_d,
            total_latency_disagreement=total_d,
            qps_disagreement=qps_d,
            num_methods=len(available),
        )

    def _generate_recommendation(
        self,
        method_preds: list[MethodPrediction],
        ensemble: EnsemblePrediction,
    ) -> str:
        """Generate human-readable recommendation."""
        available_count = sum(1 for m in method_preds if m.available)
        if available_count == 0:
            return "No prediction methods succeeded. Provide more benchmark data."

        parts = []
        parts.append(
            f"Ensemble prediction based on {available_count} method(s)."
        )

        # Check for high disagreement
        high_disagree = []
        for metric, level in [
            ("TTFT", ensemble.ttft_disagreement),
            ("TPOT", ensemble.tpot_disagreement),
            ("Total latency", ensemble.total_latency_disagreement),
        ]:
            if level in (DisagreementLevel.MODERATE, DisagreementLevel.HIGH):
                high_disagree.append(metric)

        if high_disagree:
            parts.append(
                f"WARNING: High disagreement between methods for: {', '.join(high_disagree)}. "
                "Consider collecting more benchmark data to improve prediction reliability."
            )
        elif available_count >= 2:
            parts.append("Methods show good agreement — prediction is reliable.")

        failed = [m for m in method_preds if not m.available]
        if failed:
            names = [m.method for m in failed]
            parts.append(f"Methods not available: {', '.join(names)}.")

        return " ".join(parts)


def ensemble_predict(
    benchmarks: list[BenchmarkData],
    predict_qps: float | None = None,
    predict_ratio: tuple[int, int] | None = None,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_ms: float | None = None,
    percentile: float = 95.0,
) -> dict:
    """Programmatic API for ensemble prediction.

    Returns:
        Dictionary with ensemble prediction results.
    """
    predictor = EnsemblePredictor(percentile=percentile)
    report = predictor.predict(
        benchmarks,
        predict_qps=predict_qps,
        predict_ratio=predict_ratio,
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_ms=sla_total_ms,
    )
    return report.model_dump()

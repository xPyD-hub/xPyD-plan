"""Latency-token regression analysis.

Fit linear regression models to understand how latency scales with token counts
and predict expected latency for given token counts.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class RegressionFit(BaseModel):
    """Result of a single linear regression fit."""

    predictor: str = Field(..., description="Predictor variable name")
    target: str = Field(..., description="Target latency metric name")
    slope: float = Field(..., description="Regression slope (ms per token)")
    intercept: float = Field(..., description="Regression intercept (ms)")
    r_squared: float = Field(..., ge=0, le=1, description="Coefficient of determination")
    n_samples: int = Field(..., ge=1, description="Number of data points")
    slope_std_err: float = Field(..., ge=0, description="Standard error of slope")


class PredictedLatency(BaseModel):
    """Predicted latency for given token counts."""

    metric: str = Field(..., description="Latency metric name")
    predicted_ms: float = Field(..., description="Predicted latency (ms)")
    ci_lower_ms: float = Field(..., description="95% CI lower bound (ms)")
    ci_upper_ms: float = Field(..., description="95% CI upper bound (ms)")
    token_count: int = Field(..., description="Input token count used for prediction")


class RegressionReport(BaseModel):
    """Complete regression analysis report."""

    fits: list[RegressionFit] = Field(..., description="Regression fits for each metric pair")
    predictions: list[PredictedLatency] = Field(
        default_factory=list, description="Predicted latencies (if requested)"
    )


def _linear_regression(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Compute linear regression: y = slope*x + intercept.

    Returns (slope, intercept, r_squared, slope_std_err, residual_std).
    """
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_yy = np.sum((y - y_mean) ** 2)

    if ss_xx == 0:
        return 0.0, float(y_mean), 0.0, 0.0, 0.0

    slope = float(ss_xy / ss_xx)
    intercept = float(y_mean - slope * x_mean)

    ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
    r_squared = 1 - ss_res / ss_yy if ss_yy > 0 else 0.0
    r_squared = max(0.0, min(1.0, r_squared))

    if n > 2:
        mse = ss_res / (n - 2)
        slope_std_err = math.sqrt(mse / ss_xx) if ss_xx > 0 else 0.0
        residual_std = math.sqrt(mse)
    else:
        slope_std_err = 0.0
        residual_std = 0.0

    return slope, intercept, r_squared, slope_std_err, residual_std


def _predict_with_ci(
    x_pred: float,
    slope: float,
    intercept: float,
    residual_std: float,
    x_arr: np.ndarray,
    n: int,
) -> tuple[float, float, float]:
    """Predict y at x_pred with 95% confidence interval."""
    y_pred = slope * x_pred + intercept

    if n <= 2 or residual_std == 0:
        return y_pred, y_pred, y_pred

    x_mean = float(np.mean(x_arr))
    ss_xx = float(np.sum((x_arr - x_mean) ** 2))

    # Use ~1.96 for 95% CI (normal approximation for large n)
    t_val = 1.96
    if ss_xx > 0:
        se_pred = residual_std * math.sqrt(1 + 1 / n + (x_pred - x_mean) ** 2 / ss_xx)
    else:
        se_pred = residual_std

    ci_lower = y_pred - t_val * se_pred
    ci_upper = y_pred + t_val * se_pred
    return y_pred, ci_lower, ci_upper


class RegressionAnalyzer:
    """Analyze latency-token regression relationships."""

    def __init__(self, data: BenchmarkData) -> None:
        self._data = data

    def analyze(
        self,
        predict_prompt: int | None = None,
        predict_output: int | None = None,
    ) -> RegressionReport:
        """Run regression analysis and optional predictions."""
        requests = self._data.requests
        prompt_arr = np.array([r.prompt_tokens for r in requests], dtype=float)
        output_arr = np.array([r.output_tokens for r in requests], dtype=float)
        ttft_arr = np.array([r.ttft_ms for r in requests], dtype=float)
        tpot_arr = np.array([r.tpot_ms for r in requests], dtype=float)
        total_arr = np.array([r.total_latency_ms for r in requests], dtype=float)
        total_tokens = prompt_arr + output_arr

        n = len(requests)

        # Define regression pairs: (predictor_name, predictor_arr, target_name, target_arr)
        pairs = [
            ("prompt_tokens", prompt_arr, "ttft_ms", ttft_arr),
            ("output_tokens", output_arr, "tpot_ms", tpot_arr),
            ("total_tokens", total_tokens, "total_latency_ms", total_arr),
        ]

        fits: list[RegressionFit] = []
        residual_stds: dict[str, tuple[float, np.ndarray]] = {}

        for pred_name, x, target_name, y in pairs:
            slope, intercept, r_sq, slope_se, res_std = _linear_regression(x, y)
            fits.append(
                RegressionFit(
                    predictor=pred_name,
                    target=target_name,
                    slope=slope,
                    intercept=intercept,
                    r_squared=r_sq,
                    n_samples=n,
                    slope_std_err=slope_se,
                )
            )
            residual_stds[target_name] = (res_std, x)

        predictions: list[PredictedLatency] = []

        if predict_prompt is not None:
            # TTFT prediction
            fit_ttft = fits[0]
            res_std, x_arr = residual_stds["ttft_ms"]
            y_pred, ci_lo, ci_hi = _predict_with_ci(
                float(predict_prompt),
                fit_ttft.slope,
                fit_ttft.intercept,
                res_std,
                x_arr,
                n,
            )
            predictions.append(
                PredictedLatency(
                    metric="ttft_ms",
                    predicted_ms=y_pred,
                    ci_lower_ms=ci_lo,
                    ci_upper_ms=ci_hi,
                    token_count=predict_prompt,
                )
            )

        if predict_output is not None:
            # TPOT prediction
            fit_tpot = fits[1]
            res_std, x_arr = residual_stds["tpot_ms"]
            y_pred, ci_lo, ci_hi = _predict_with_ci(
                float(predict_output),
                fit_tpot.slope,
                fit_tpot.intercept,
                res_std,
                x_arr,
                n,
            )
            predictions.append(
                PredictedLatency(
                    metric="tpot_ms",
                    predicted_ms=y_pred,
                    ci_lower_ms=ci_lo,
                    ci_upper_ms=ci_hi,
                    token_count=predict_output,
                )
            )

        if predict_prompt is not None and predict_output is not None:
            # Total latency prediction
            fit_total = fits[2]
            total_tok = predict_prompt + predict_output
            res_std, x_arr = residual_stds["total_latency_ms"]
            y_pred, ci_lo, ci_hi = _predict_with_ci(
                float(total_tok),
                fit_total.slope,
                fit_total.intercept,
                res_std,
                x_arr,
                n,
            )
            predictions.append(
                PredictedLatency(
                    metric="total_latency_ms",
                    predicted_ms=y_pred,
                    ci_lower_ms=ci_lo,
                    ci_upper_ms=ci_hi,
                    token_count=total_tok,
                )
            )

        return RegressionReport(fits=fits, predictions=predictions)


def analyze_regression(
    data: BenchmarkData,
    predict_prompt: int | None = None,
    predict_output: int | None = None,
) -> RegressionReport:
    """Programmatic API for regression analysis."""
    analyzer = RegressionAnalyzer(data)
    return analyzer.analyze(
        predict_prompt=predict_prompt, predict_output=predict_output
    )

"""Token budget estimator.

Given SLA latency thresholds, invert regression fits to estimate the maximum
prompt and output token counts that stay within SLA constraints.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.regression import RegressionAnalyzer, RegressionFit


class TokenLimit(BaseModel):
    """Estimated maximum token count for a single metric."""

    metric: str = Field(..., description="Latency metric (ttft_ms, tpot_ms, total_latency_ms)")
    sla_threshold_ms: float = Field(..., description="SLA threshold used")
    max_tokens: int = Field(..., description="Estimated max tokens within SLA (floored)")
    predictor: str = Field(
        ..., description="Token type (prompt_tokens, output_tokens, total_tokens)"
    )
    r_squared: float = Field(..., ge=0, le=1, description="R² of underlying regression fit")
    confidence: str = Field(
        ..., description="Confidence level: HIGH (R²≥0.7), MEDIUM (0.4-0.7), LOW (<0.4)"
    )


class TokenBudgetReport(BaseModel):
    """Complete token budget estimation report."""

    limits: list[TokenLimit] = Field(..., description="Per-metric token limits")
    effective_max_prompt: int | None = Field(
        None, description="Conservative max prompt tokens (min across relevant limits)"
    )
    effective_max_output: int | None = Field(
        None, description="Conservative max output tokens (min across relevant limits)"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Warnings about estimation quality"
    )


def _confidence_label(r_squared: float) -> str:
    if r_squared >= 0.7:
        return "HIGH"
    elif r_squared >= 0.4:
        return "MEDIUM"
    return "LOW"


def _invert_regression(fit: RegressionFit, sla_ms: float) -> int:
    """Invert y = slope*x + intercept to find x given y = sla_ms."""
    if fit.slope <= 0:
        # Flat or negative slope — tokens don't increase latency, no meaningful limit
        return -1  # sentinel for "unlimited"
    max_tokens = (sla_ms - fit.intercept) / fit.slope
    return max(0, math.floor(max_tokens))


class TokenBudgetEstimator:
    """Estimate maximum token counts that satisfy SLA thresholds."""

    def __init__(self, data: BenchmarkData) -> None:
        self._data = data

    def estimate(
        self,
        sla_ttft_ms: float | None = None,
        sla_tpot_ms: float | None = None,
        sla_total_ms: float | None = None,
    ) -> TokenBudgetReport:
        """Estimate token budgets from regression fits and SLA thresholds.

        At least one SLA threshold must be provided.
        """
        if sla_ttft_ms is None and sla_tpot_ms is None and sla_total_ms is None:
            raise ValueError("At least one SLA threshold must be provided")

        analyzer = RegressionAnalyzer(self._data)
        report = analyzer.analyze()
        fits_by_target: dict[str, RegressionFit] = {f.target: f for f in report.fits}

        limits: list[TokenLimit] = []
        warnings: list[str] = []
        prompt_limits: list[int] = []
        output_limits: list[int] = []

        # TTFT → prompt_tokens
        if sla_ttft_ms is not None:
            fit = fits_by_target["ttft_ms"]
            conf = _confidence_label(fit.r_squared)
            max_tok = _invert_regression(fit, sla_ttft_ms)
            if max_tok < 0:
                warnings.append(
                    f"TTFT regression has non-positive slope ({fit.slope:.4f}); "
                    "cannot estimate prompt token limit"
                )
            else:
                limits.append(
                    TokenLimit(
                        metric="ttft_ms",
                        sla_threshold_ms=sla_ttft_ms,
                        max_tokens=max_tok,
                        predictor="prompt_tokens",
                        r_squared=fit.r_squared,
                        confidence=conf,
                    )
                )
                prompt_limits.append(max_tok)
            if conf == "LOW":
                warnings.append(
                    f"TTFT regression R²={fit.r_squared:.3f} is low; "
                    "prompt token estimate may be unreliable"
                )

        # TPOT → output_tokens
        if sla_tpot_ms is not None:
            fit = fits_by_target["tpot_ms"]
            conf = _confidence_label(fit.r_squared)
            max_tok = _invert_regression(fit, sla_tpot_ms)
            if max_tok < 0:
                warnings.append(
                    f"TPOT regression has non-positive slope ({fit.slope:.4f}); "
                    "cannot estimate output token limit"
                )
            else:
                limits.append(
                    TokenLimit(
                        metric="tpot_ms",
                        sla_threshold_ms=sla_tpot_ms,
                        max_tokens=max_tok,
                        predictor="output_tokens",
                        r_squared=fit.r_squared,
                        confidence=conf,
                    )
                )
                output_limits.append(max_tok)
            if conf == "LOW":
                warnings.append(
                    f"TPOT regression R²={fit.r_squared:.3f} is low; "
                    "output token estimate may be unreliable"
                )

        # total_latency → total_tokens (contributes to both prompt and output limits indirectly)
        if sla_total_ms is not None:
            fit = fits_by_target["total_latency_ms"]
            conf = _confidence_label(fit.r_squared)
            max_tok = _invert_regression(fit, sla_total_ms)
            if max_tok < 0:
                warnings.append(
                    f"Total latency regression has non-positive slope ({fit.slope:.4f}); "
                    "cannot estimate total token limit"
                )
            else:
                limits.append(
                    TokenLimit(
                        metric="total_latency_ms",
                        sla_threshold_ms=sla_total_ms,
                        max_tokens=max_tok,
                        predictor="total_tokens",
                        r_squared=fit.r_squared,
                        confidence=conf,
                    )
                )
            if conf == "LOW":
                warnings.append(
                    f"Total latency regression R²={fit.r_squared:.3f} "
                    "is low; total token estimate may be unreliable"
                )

        effective_prompt = min(prompt_limits) if prompt_limits else None
        effective_output = min(output_limits) if output_limits else None

        return TokenBudgetReport(
            limits=limits,
            effective_max_prompt=effective_prompt,
            effective_max_output=effective_output,
            warnings=warnings,
        )


def estimate_token_budget(
    data: BenchmarkData,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_ms: float | None = None,
) -> TokenBudgetReport:
    """Programmatic API for token budget estimation."""
    estimator = TokenBudgetEstimator(data)
    return estimator.estimate(
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_ms=sla_total_ms,
    )

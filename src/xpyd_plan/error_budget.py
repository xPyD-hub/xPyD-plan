"""SLO error budget burn rate analysis.

Inspired by SRE practices, this module tracks how fast the SLA error budget
is being consumed across time windows. An error budget is the allowed fraction
of requests that may violate SLA (e.g., 0.1% for a 99.9% SLO). Burn rate
measures how many multiples of the ideal consumption rate the budget is
actually being spent at.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData, BenchmarkRequest


class BurnRateLevel(str, Enum):
    """Burn rate severity classification."""

    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EXHAUSTED = "EXHAUSTED"


class ErrorBudgetConfig(BaseModel):
    """Configuration for error budget analysis."""

    slo_target: float = Field(
        0.999, gt=0, le=1, description="SLO target (e.g., 0.999 for 99.9%)"
    )
    sla_ttft_ms: float | None = Field(None, description="TTFT SLA threshold in ms")
    sla_tpot_ms: float | None = Field(None, description="TPOT SLA threshold in ms")
    sla_total_latency_ms: float | None = Field(
        None, description="Total latency SLA threshold in ms"
    )
    window_size: float = Field(
        10.0, gt=0, description="Time window size in seconds"
    )
    warning_burn_rate: float = Field(
        2.0, gt=0, description="Burn rate threshold for WARNING"
    )
    critical_burn_rate: float = Field(
        10.0, gt=0, description="Burn rate threshold for CRITICAL"
    )


class BurnRateWindow(BaseModel):
    """Burn rate for a single time window."""

    window_start: float = Field(..., description="Window start (epoch seconds)")
    window_end: float = Field(..., description="Window end (epoch seconds)")
    total_requests: int = Field(..., ge=0)
    failing_requests: int = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=1)
    burn_rate: float = Field(
        ..., ge=0, description="Multiples of ideal error consumption rate"
    )
    level: BurnRateLevel = Field(..., description="Severity classification")
    budget_consumed_fraction: float = Field(
        ..., ge=0, description="Fraction of total budget consumed in this window"
    )


class BudgetStatus(BaseModel):
    """Overall error budget status."""

    total_budget: float = Field(
        ..., ge=0, le=1, description="Allowed error fraction (1 - SLO target)"
    )
    consumed: float = Field(
        ..., ge=0, description="Fraction of budget consumed so far"
    )
    remaining: float = Field(
        ..., description="Fraction of budget remaining (can be negative)"
    )
    is_exhausted: bool = Field(..., description="Whether budget is fully consumed")


class ErrorBudgetReport(BaseModel):
    """Complete error budget burn rate report."""

    config: ErrorBudgetConfig
    total_requests: int = Field(..., ge=0)
    total_failures: int = Field(..., ge=0)
    overall_error_rate: float = Field(..., ge=0, le=1)
    overall_burn_rate: float = Field(..., ge=0)
    overall_level: BurnRateLevel
    budget_status: BudgetStatus
    windows: list[BurnRateWindow] = Field(default_factory=list)
    worst_window_burn_rate: float = Field(
        ..., ge=0, description="Peak burn rate across all windows"
    )
    safe_window_fraction: float = Field(
        ..., ge=0, le=1, description="Fraction of windows at SAFE level"
    )
    recommendation: str = Field(..., description="Actionable recommendation")


def _request_passes_sla(
    req: BenchmarkRequest,
    sla_ttft_ms: float | None,
    sla_tpot_ms: float | None,
    sla_total_latency_ms: float | None,
) -> bool:
    """Check if a single request passes all configured SLA thresholds."""
    if sla_ttft_ms is not None and req.ttft_ms > sla_ttft_ms:
        return False
    if sla_tpot_ms is not None and req.tpot_ms > sla_tpot_ms:
        return False
    if sla_total_latency_ms is not None and req.total_latency_ms > sla_total_latency_ms:
        return False
    return True


def _classify_burn_rate(
    burn_rate: float,
    warning_threshold: float,
    critical_threshold: float,
    budget_remaining: float,
) -> BurnRateLevel:
    """Classify burn rate into severity level."""
    if budget_remaining <= 0:
        return BurnRateLevel.EXHAUSTED
    if burn_rate >= critical_threshold:
        return BurnRateLevel.CRITICAL
    if burn_rate >= warning_threshold:
        return BurnRateLevel.WARNING
    return BurnRateLevel.SAFE


class ErrorBudgetAnalyzer:
    """Analyze SLO error budget burn rate from benchmark data."""

    def __init__(
        self,
        slo_target: float = 0.999,
        sla_ttft_ms: float | None = None,
        sla_tpot_ms: float | None = None,
        sla_total_latency_ms: float | None = None,
        window_size: float = 10.0,
        warning_burn_rate: float = 2.0,
        critical_burn_rate: float = 10.0,
    ) -> None:
        self._config = ErrorBudgetConfig(
            slo_target=slo_target,
            sla_ttft_ms=sla_ttft_ms,
            sla_tpot_ms=sla_tpot_ms,
            sla_total_latency_ms=sla_total_latency_ms,
            window_size=window_size,
            warning_burn_rate=warning_burn_rate,
            critical_burn_rate=critical_burn_rate,
        )

    def analyze(self, data: BenchmarkData) -> ErrorBudgetReport:
        """Analyze error budget burn rate."""
        requests = data.requests
        if not requests:
            return self._empty_report()

        error_budget = 1.0 - self._config.slo_target

        # Classify each request
        passes = [
            _request_passes_sla(
                r,
                self._config.sla_ttft_ms,
                self._config.sla_tpot_ms,
                self._config.sla_total_latency_ms,
            )
            for r in requests
        ]
        total = len(requests)
        failures = sum(1 for p in passes if not p)
        overall_error_rate = failures / total if total > 0 else 0.0

        # Overall burn rate: actual error rate / allowed error rate
        overall_burn_rate = (
            overall_error_rate / error_budget if error_budget > 0 else float("inf")
        )

        budget_consumed = overall_error_rate / error_budget if error_budget > 0 else float("inf")
        budget_remaining = 1.0 - budget_consumed

        budget_status = BudgetStatus(
            total_budget=error_budget,
            consumed=min(budget_consumed, 999.0),
            remaining=budget_remaining,
            is_exhausted=budget_remaining <= 0,
        )

        overall_level = _classify_burn_rate(
            overall_burn_rate,
            self._config.warning_burn_rate,
            self._config.critical_burn_rate,
            budget_remaining,
        )

        # Time-windowed analysis
        timestamps = np.array([r.timestamp for r in requests])
        if timestamps.min() > 0:
            t_start = timestamps.min()
            t_end = timestamps.max()
        else:
            t_start = 0.0
            t_end = float(total)

        windows: list[BurnRateWindow] = []
        ws = self._config.window_size
        current = t_start

        while current < t_end:
            w_end = current + ws
            # Find requests in this window
            mask = (timestamps >= current) & (timestamps < w_end)
            indices = np.where(mask)[0]
            w_total = len(indices)
            w_failures = sum(1 for i in indices if not passes[i])

            if w_total > 0:
                w_error_rate = w_failures / w_total
                w_burn_rate = (
                    w_error_rate / error_budget if error_budget > 0 else float("inf")
                )
                w_budget_frac = w_error_rate / error_budget if error_budget > 0 else 0.0
            else:
                w_error_rate = 0.0
                w_burn_rate = 0.0
                w_budget_frac = 0.0

            w_level = _classify_burn_rate(
                w_burn_rate,
                self._config.warning_burn_rate,
                self._config.critical_burn_rate,
                budget_remaining,
            )

            windows.append(
                BurnRateWindow(
                    window_start=current,
                    window_end=w_end,
                    total_requests=w_total,
                    failing_requests=w_failures,
                    error_rate=w_error_rate,
                    burn_rate=w_burn_rate,
                    level=w_level,
                    budget_consumed_fraction=min(w_budget_frac, 999.0),
                )
            )
            current = w_end

        worst_burn = max((w.burn_rate for w in windows), default=0.0)
        safe_count = sum(1 for w in windows if w.level == BurnRateLevel.SAFE)
        safe_frac = safe_count / len(windows) if windows else 1.0

        recommendation = self._recommend(overall_level, overall_burn_rate, budget_status)

        return ErrorBudgetReport(
            config=self._config,
            total_requests=total,
            total_failures=failures,
            overall_error_rate=overall_error_rate,
            overall_burn_rate=overall_burn_rate,
            overall_level=overall_level,
            budget_status=budget_status,
            windows=windows,
            worst_window_burn_rate=worst_burn,
            safe_window_fraction=safe_frac,
            recommendation=recommendation,
        )

    def _empty_report(self) -> ErrorBudgetReport:
        error_budget = 1.0 - self._config.slo_target
        return ErrorBudgetReport(
            config=self._config,
            total_requests=0,
            total_failures=0,
            overall_error_rate=0.0,
            overall_burn_rate=0.0,
            overall_level=BurnRateLevel.SAFE,
            budget_status=BudgetStatus(
                total_budget=error_budget,
                consumed=0.0,
                remaining=1.0,
                is_exhausted=False,
            ),
            windows=[],
            worst_window_burn_rate=0.0,
            safe_window_fraction=1.0,
            recommendation="No requests to analyze.",
        )

    @staticmethod
    def _recommend(
        level: BurnRateLevel,
        burn_rate: float,
        status: BudgetStatus,
    ) -> str:
        if level == BurnRateLevel.EXHAUSTED:
            return (
                f"Error budget exhausted ({status.consumed:.1%} consumed). "
                "Immediate action required: scale instances or relax SLA thresholds."
            )
        if level == BurnRateLevel.CRITICAL:
            return (
                f"Critical burn rate ({burn_rate:.1f}x). "
                "Budget will exhaust soon. Investigate root cause and consider scaling."
            )
        if level == BurnRateLevel.WARNING:
            return (
                f"Elevated burn rate ({burn_rate:.1f}x). "
                "Monitor closely. Budget remaining: {:.1%}.".format(status.remaining)
            )
        return (
            f"Burn rate healthy ({burn_rate:.1f}x). "
            "Budget remaining: {:.1%}.".format(status.remaining)
        )


def analyze_error_budget(
    benchmark_path: str,
    slo_target: float = 0.999,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_latency_ms: float | None = None,
    window_size: float = 10.0,
    warning_burn_rate: float = 2.0,
    critical_burn_rate: float = 10.0,
) -> dict:
    """Programmatic API for error budget burn rate analysis.

    Args:
        benchmark_path: Path to benchmark JSON file.
        slo_target: SLO target (e.g., 0.999 for 99.9%).
        sla_ttft_ms: TTFT SLA threshold in ms.
        sla_tpot_ms: TPOT SLA threshold in ms.
        sla_total_latency_ms: Total latency SLA threshold in ms.
        window_size: Time window size in seconds.
        warning_burn_rate: Burn rate threshold for WARNING level.
        critical_burn_rate: Burn rate threshold for CRITICAL level.

    Returns:
        dict: ErrorBudgetReport as a dictionary.
    """
    from .bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    analyzer = ErrorBudgetAnalyzer(
        slo_target=slo_target,
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_latency_ms=sla_total_latency_ms,
        window_size=window_size,
        warning_burn_rate=warning_burn_rate,
        critical_burn_rate=critical_burn_rate,
    )
    report = analyzer.analyze(data)
    return report.model_dump()

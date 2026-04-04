"""Capacity forecasting from historical trend data.

Project future latency and QPS trajectories using historical benchmark
trends.  Estimate time-to-SLA-breach so users can proactively scale
before degradation occurs.
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.trend import TrendTracker


class ForecastMethod(str, Enum):
    """Extrapolation method."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class ForecastPoint(BaseModel):
    """A single projected data point."""

    days_from_now: float = Field(..., description="Days into the future")
    ttft_p95_ms: float = Field(..., ge=0)
    tpot_p95_ms: float = Field(..., ge=0)
    total_latency_p95_ms: float = Field(..., ge=0)


class CapacityExhaustion(BaseModel):
    """When a specific SLA metric is projected to breach its threshold."""

    metric: str = Field(..., description="Metric name (e.g. ttft_p95_ms)")
    current_value_ms: float = Field(..., ge=0)
    threshold_ms: float = Field(..., ge=0)
    days_to_breach: float | None = Field(
        None, description="Days until breach; None if no breach within horizon"
    )
    breaches_within_horizon: bool = Field(False)


class ForecastReport(BaseModel):
    """Complete forecast report."""

    method: ForecastMethod
    horizon_days: int = Field(..., ge=1)
    num_historical_entries: int = Field(..., ge=0)
    projections: list[ForecastPoint] = Field(default_factory=list)
    exhaustions: list[CapacityExhaustion] = Field(default_factory=list)
    has_breach: bool = Field(False, description="True if any metric breaches within horizon")
    earliest_breach_days: float | None = Field(
        None, description="Days to earliest breach; None if no breach"
    )


class CapacityForecaster:
    """Forecast capacity from historical trend data."""

    METRICS = ["ttft_p95_ms", "tpot_p95_ms", "total_latency_p95_ms"]

    def __init__(self, tracker: TrendTracker) -> None:
        self._tracker = tracker

    def forecast(
        self,
        horizon_days: int = 30,
        method: ForecastMethod = ForecastMethod.LINEAR,
        sla_ttft_ms: float | None = None,
        sla_tpot_ms: float | None = None,
        sla_total_ms: float | None = None,
        num_points: int = 10,
    ) -> ForecastReport:
        """Run capacity forecast.

        Parameters
        ----------
        horizon_days:
            How many days to project into the future.
        method:
            Extrapolation method (linear or exponential).
        sla_ttft_ms, sla_tpot_ms, sla_total_ms:
            SLA thresholds; if provided, compute time-to-breach.
        num_points:
            Number of projection points to generate.
        """
        entries = self._tracker.list_entries()

        if len(entries) < 2:
            return ForecastReport(
                method=method,
                horizon_days=horizon_days,
                num_historical_entries=len(entries),
            )

        # Build time series: days relative to first entry
        timestamps = np.array([e.timestamp for e in entries])
        days = (timestamps - timestamps[0]) / 86400.0

        metric_series: dict[str, np.ndarray] = {}
        for m in self.METRICS:
            metric_series[m] = np.array([getattr(e, m) for e in entries])

        # Current values (last entry)
        current = {m: float(metric_series[m][-1]) for m in self.METRICS}
        last_day = float(days[-1])

        # Fit models
        models: dict[str, tuple] = {}
        for m in self.METRICS:
            models[m] = self._fit(days, metric_series[m], method)

        # Generate projection points
        step = horizon_days / num_points
        projections: list[ForecastPoint] = []
        for i in range(1, num_points + 1):
            d = step * i
            future_day = last_day + d
            values = {}
            for m in self.METRICS:
                values[m] = max(0.0, self._predict(models[m], future_day, method))
            projections.append(
                ForecastPoint(
                    days_from_now=round(d, 2),
                    ttft_p95_ms=round(values["ttft_p95_ms"], 2),
                    tpot_p95_ms=round(values["tpot_p95_ms"], 2),
                    total_latency_p95_ms=round(values["total_latency_p95_ms"], 2),
                )
            )

        # Compute exhaustions
        thresholds = {
            "ttft_p95_ms": sla_ttft_ms,
            "tpot_p95_ms": sla_tpot_ms,
            "total_latency_p95_ms": sla_total_ms,
        }

        exhaustions: list[CapacityExhaustion] = []
        for m in self.METRICS:
            thresh = thresholds[m]
            if thresh is None:
                continue

            dtb = self._days_to_breach(
                models[m], method, last_day, thresh, horizon_days
            )
            exhaustions.append(
                CapacityExhaustion(
                    metric=m,
                    current_value_ms=round(current[m], 2),
                    threshold_ms=thresh,
                    days_to_breach=round(dtb, 2) if dtb is not None else None,
                    breaches_within_horizon=dtb is not None,
                )
            )

        has_breach = any(e.breaches_within_horizon for e in exhaustions)
        breach_days = [
            e.days_to_breach for e in exhaustions if e.days_to_breach is not None
        ]
        earliest = min(breach_days) if breach_days else None

        return ForecastReport(
            method=method,
            horizon_days=horizon_days,
            num_historical_entries=len(entries),
            projections=projections,
            exhaustions=exhaustions,
            has_breach=has_breach,
            earliest_breach_days=earliest,
        )

    @staticmethod
    def _fit(
        days: np.ndarray, values: np.ndarray, method: ForecastMethod
    ) -> tuple:
        """Fit a model and return parameters."""
        if method == ForecastMethod.LINEAR:
            # y = a*x + b
            coeffs = np.polyfit(days, values, 1)
            return tuple(coeffs)  # (slope, intercept)
        else:
            # Exponential: y = b * exp(a*x)
            # Fit in log space: ln(y) = a*x + ln(b)
            safe_values = np.clip(values, 1e-9, None)
            log_v = np.log(safe_values)
            coeffs = np.polyfit(days, log_v, 1)
            return (coeffs[0], math.exp(coeffs[1]))  # (rate, base)

    @staticmethod
    def _predict(params: tuple, day: float, method: ForecastMethod) -> float:
        if method == ForecastMethod.LINEAR:
            slope, intercept = params
            return slope * day + intercept
        else:
            rate, base = params
            return base * math.exp(rate * day)

    def _days_to_breach(
        self,
        params: tuple,
        method: ForecastMethod,
        last_day: float,
        threshold: float,
        horizon_days: int,
    ) -> float | None:
        """Compute days from now until predicted value exceeds threshold."""
        # Check if already breached
        current_val = self._predict(params, last_day, method)
        if current_val >= threshold:
            return 0.0

        if method == ForecastMethod.LINEAR:
            slope, intercept = params
            if slope <= 0:
                return None  # improving, won't breach
            # threshold = slope * (last_day + d) + intercept
            breach_day = (threshold - intercept) / slope
            d = breach_day - last_day
            if d < 0 or d > horizon_days:
                return None
            return d

        else:
            rate, base = params
            if rate <= 0:
                return None  # improving
            if base <= 0:
                return None
            # threshold = base * exp(rate * (last_day + d))
            try:
                breach_day = math.log(threshold / base) / rate
            except (ValueError, ZeroDivisionError):
                return None
            d = breach_day - last_day
            if d < 0 or d > horizon_days:
                return None
            return d


def forecast_capacity(
    trend_db: str | Path,
    horizon_days: int = 30,
    method: str = "linear",
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_ms: float | None = None,
) -> dict:
    """Programmatic API for capacity forecasting.

    Parameters
    ----------
    trend_db:
        Path to TrendTracker SQLite database.
    horizon_days:
        Planning horizon in days.
    method:
        "linear" or "exponential".
    sla_ttft_ms, sla_tpot_ms, sla_total_ms:
        Optional SLA thresholds for breach estimation.

    Returns
    -------
    dict with forecast report data.
    """
    tracker = TrendTracker(db_path=trend_db)
    try:
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(
            horizon_days=horizon_days,
            method=ForecastMethod(method),
            sla_ttft_ms=sla_ttft_ms,
            sla_tpot_ms=sla_tpot_ms,
            sla_total_ms=sla_total_ms,
        )
        return report.model_dump()
    finally:
        tracker.close()

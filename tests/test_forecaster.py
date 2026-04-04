"""Tests for capacity forecasting (M47)."""

from __future__ import annotations

import pytest

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.forecaster import (
    CapacityForecaster,
    ForecastMethod,
    ForecastReport,
    forecast_capacity,
)
from xpyd_plan.trend import TrendTracker


def _make_benchmark(
    qps: float,
    n_prefill: int,
    n_decode: int,
    ttft_base: float,
    tpot_base: float,
    n_requests: int = 100,
) -> BenchmarkData:
    """Create a synthetic BenchmarkData with controllable latencies."""
    reqs = []
    for i in range(n_requests):
        reqs.append(
            BenchmarkRequest(
                request_id=f"r{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=ttft_base + (i % 10),
                tpot_ms=tpot_base + (i % 5),
                total_latency_ms=ttft_base + tpot_base * 50 + (i % 15),
                timestamp=1000.0 + i,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=n_prefill,
            num_decode_instances=n_decode,
            total_instances=n_prefill + n_decode,
            measured_qps=qps,
        ),
        requests=reqs,
    )


def _build_tracker_with_trend(
    num_entries: int = 5,
    ttft_start: float = 50.0,
    ttft_increment: float = 5.0,
    tpot_start: float = 20.0,
    tpot_increment: float = 2.0,
    day_gap: float = 1.0,
) -> TrendTracker:
    """Build an in-memory TrendTracker with a known upward trend."""
    tracker = TrendTracker(db_path=":memory:")
    base_ts = 1_700_000_000.0
    for i in range(num_entries):
        bm = _make_benchmark(
            qps=100.0,
            n_prefill=2,
            n_decode=6,
            ttft_base=ttft_start + ttft_increment * i,
            tpot_base=tpot_start + tpot_increment * i,
        )
        tracker.add(bm, label=f"run-{i}", timestamp=base_ts + i * day_gap * 86400)
    return tracker


# --- Basic construction ---


class TestCapacityForecasterBasic:
    def test_empty_tracker(self):
        tracker = TrendTracker(db_path=":memory:")
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast()
        assert report.num_historical_entries == 0
        assert report.projections == []
        assert not report.has_breach

    def test_single_entry(self):
        tracker = TrendTracker(db_path=":memory:")
        bm = _make_benchmark(100, 2, 6, 50, 20)
        tracker.add(bm, label="only")
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast()
        assert report.num_historical_entries == 1
        assert report.projections == []

    def test_two_entries_produce_projections(self):
        tracker = _build_tracker_with_trend(num_entries=2)
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(horizon_days=10, num_points=5)
        assert report.num_historical_entries == 2
        assert len(report.projections) == 5

    def test_projection_days_increase(self):
        tracker = _build_tracker_with_trend()
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(horizon_days=30, num_points=10)
        days = [p.days_from_now for p in report.projections]
        assert days == sorted(days)
        assert days[-1] == pytest.approx(30.0, abs=0.1)


# --- Linear forecasting ---


class TestLinearForecast:
    def test_increasing_trend_projects_higher(self):
        tracker = _build_tracker_with_trend(ttft_increment=10.0)
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(method=ForecastMethod.LINEAR, num_points=3)
        # Last entry TTFT base is 50 + 10*4 = 90; projections should be higher
        last_proj = report.projections[-1]
        assert last_proj.ttft_p95_ms > 90.0

    def test_breach_detected(self):
        tracker = _build_tracker_with_trend(
            ttft_start=50.0, ttft_increment=10.0, num_entries=5
        )
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(
            horizon_days=30,
            sla_ttft_ms=120.0,
            method=ForecastMethod.LINEAR,
        )
        assert report.has_breach
        assert report.earliest_breach_days is not None
        assert report.earliest_breach_days > 0

    def test_no_breach_when_threshold_high(self):
        tracker = _build_tracker_with_trend(
            ttft_start=50.0, ttft_increment=1.0, num_entries=5
        )
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(
            horizon_days=10,
            sla_ttft_ms=999.0,
            method=ForecastMethod.LINEAR,
        )
        ttft_exhaustion = [e for e in report.exhaustions if e.metric == "ttft_p95_ms"]
        assert len(ttft_exhaustion) == 1
        assert not ttft_exhaustion[0].breaches_within_horizon

    def test_already_breached(self):
        # Start high, so current already exceeds threshold
        tracker = _build_tracker_with_trend(ttft_start=200.0, ttft_increment=10.0)
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(sla_ttft_ms=100.0)
        ttft_ex = [e for e in report.exhaustions if e.metric == "ttft_p95_ms"][0]
        assert ttft_ex.breaches_within_horizon
        assert ttft_ex.days_to_breach == 0.0


# --- Exponential forecasting ---


class TestExponentialForecast:
    def test_exponential_produces_projections(self):
        tracker = _build_tracker_with_trend()
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(method=ForecastMethod.EXPONENTIAL, num_points=5)
        assert len(report.projections) == 5
        # Values should be positive
        for p in report.projections:
            assert p.ttft_p95_ms >= 0
            assert p.tpot_p95_ms >= 0

    def test_exponential_breach(self):
        tracker = _build_tracker_with_trend(
            ttft_start=50, ttft_increment=15, num_entries=5
        )
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(
            method=ForecastMethod.EXPONENTIAL,
            sla_ttft_ms=200.0,
            horizon_days=60,
        )
        assert any(e.metric == "ttft_p95_ms" for e in report.exhaustions)


# --- Multiple SLA metrics ---


class TestMultipleSLAMetrics:
    def test_all_three_thresholds(self):
        tracker = _build_tracker_with_trend()
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(
            sla_ttft_ms=200.0,
            sla_tpot_ms=100.0,
            sla_total_ms=5000.0,
        )
        assert len(report.exhaustions) == 3
        metrics = {e.metric for e in report.exhaustions}
        assert metrics == {"ttft_p95_ms", "tpot_p95_ms", "total_latency_p95_ms"}

    def test_partial_thresholds(self):
        tracker = _build_tracker_with_trend()
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(sla_ttft_ms=200.0)
        assert len(report.exhaustions) == 1
        assert report.exhaustions[0].metric == "ttft_p95_ms"


# --- ForecastReport model ---


class TestForecastReportModel:
    def test_serialization(self):
        tracker = _build_tracker_with_trend()
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(sla_ttft_ms=200.0, num_points=3)
        d = report.model_dump()
        assert "method" in d
        assert "projections" in d
        assert "exhaustions" in d
        assert "has_breach" in d

    def test_method_enum(self):
        report = ForecastReport(
            method=ForecastMethod.LINEAR,
            horizon_days=30,
            num_historical_entries=0,
        )
        assert report.method == ForecastMethod.LINEAR


# --- Programmatic API ---


class TestForecastCapacityAPI:
    def test_api_with_file(self, tmp_path):
        db_path = tmp_path / "trend.db"
        tracker = TrendTracker(db_path=str(db_path))
        base_ts = 1_700_000_000.0
        for i in range(3):
            bm = _make_benchmark(100, 2, 6, 50 + i * 10, 20 + i * 3)
            tracker.add(bm, label=f"r{i}", timestamp=base_ts + i * 86400)
        tracker.close()

        result = forecast_capacity(
            trend_db=str(db_path),
            horizon_days=14,
            method="linear",
            sla_ttft_ms=150.0,
        )
        assert isinstance(result, dict)
        assert result["num_historical_entries"] == 3
        assert "projections" in result
        assert "exhaustions" in result

    def test_api_exponential(self, tmp_path):
        db_path = tmp_path / "trend.db"
        tracker = TrendTracker(db_path=str(db_path))
        base_ts = 1_700_000_000.0
        for i in range(4):
            bm = _make_benchmark(100, 2, 6, 50 + i * 5, 20 + i * 2)
            tracker.add(bm, label=f"r{i}", timestamp=base_ts + i * 86400)
        tracker.close()

        result = forecast_capacity(
            trend_db=str(db_path),
            method="exponential",
        )
        assert result["method"] == "exponential"


# --- Edge cases ---


class TestEdgeCases:
    def test_flat_trend_no_breach(self):
        """No degradation → no breach."""
        tracker = _build_tracker_with_trend(ttft_increment=0.0, tpot_increment=0.0)
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(sla_ttft_ms=200.0, sla_tpot_ms=200.0)
        assert not report.has_breach

    def test_improving_trend_no_breach(self):
        """Negative trend (improving) → no breach."""
        tracker = _build_tracker_with_trend(
            ttft_start=100.0, ttft_increment=-5.0, tpot_start=50.0, tpot_increment=-2.0
        )
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(sla_ttft_ms=200.0)
        assert not report.has_breach

    def test_projection_values_nonnegative(self):
        """Even with negative trend, projections should be >= 0."""
        tracker = _build_tracker_with_trend(
            ttft_start=100.0, ttft_increment=-20.0, num_entries=5
        )
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(horizon_days=100, num_points=20)
        for p in report.projections:
            assert p.ttft_p95_ms >= 0
            assert p.tpot_p95_ms >= 0
            assert p.total_latency_p95_ms >= 0

    def test_horizon_one_day(self):
        tracker = _build_tracker_with_trend()
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(horizon_days=1, num_points=2)
        assert len(report.projections) == 2
        assert report.projections[-1].days_from_now == pytest.approx(1.0, abs=0.1)

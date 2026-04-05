"""Tests for QPS-latency curve fitting."""

from __future__ import annotations

import random

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.qps_curve import (
    Confidence,
    FitMethod,
    QPSCurveFitter,
    QPSCurveReport,
    fit_qps_curve,
)


def _make_benchmark(
    qps: float,
    num_prefill: int = 2,
    num_decode: int = 2,
    num_requests: int = 100,
    base_ttft: float = 50.0,
    base_tpot: float = 10.0,
    base_total: float = 200.0,
) -> BenchmarkData:
    """Create a synthetic benchmark with latency that increases with QPS."""
    rng = random.Random(int(qps * 100))
    # Simulate latency increasing with QPS (roughly linear)
    scale = 1.0 + (qps - 10.0) * 0.05  # at qps=10 → scale=1, qps=20 → scale=1.5
    requests = []
    for i in range(num_requests):
        noise = rng.uniform(0.8, 1.2)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{qps}-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=base_ttft * scale * noise,
                tpot_ms=base_tpot * scale * noise,
                total_latency_ms=base_total * scale * noise,
                timestamp=1000.0 + i * (1.0 / qps),
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=qps,
        ),
        requests=requests,
    )


def _make_benchmarks(qps_levels: list[float] | None = None) -> list[BenchmarkData]:
    if qps_levels is None:
        qps_levels = [10.0, 20.0, 30.0, 40.0, 50.0]
    return [_make_benchmark(q) for q in qps_levels]


class TestQPSCurveFitter:
    """Tests for QPSCurveFitter."""

    def test_init_default(self) -> None:
        fitter = QPSCurveFitter()
        assert fitter.method == FitMethod.AUTO
        assert fitter.percentile == 95.0

    def test_init_custom(self) -> None:
        fitter = QPSCurveFitter(method=FitMethod.LINEAR, percentile=99.0)
        assert fitter.method == FitMethod.LINEAR
        assert fitter.percentile == 99.0

    def test_init_invalid_percentile(self) -> None:
        with pytest.raises(ValueError, match="percentile"):
            QPSCurveFitter(percentile=0.5)
        with pytest.raises(ValueError, match="percentile"):
            QPSCurveFitter(percentile=101.0)

    def test_fit_requires_two_benchmarks(self) -> None:
        fitter = QPSCurveFitter()
        with pytest.raises(ValueError, match="at least 2"):
            fitter.fit([_make_benchmark(10.0)])

    def test_fit_requires_distinct_qps(self) -> None:
        fitter = QPSCurveFitter()
        with pytest.raises(ValueError, match="distinct"):
            fitter.fit([_make_benchmark(10.0), _make_benchmark(10.0)])

    def test_fit_auto_returns_report(self) -> None:
        benchmarks = _make_benchmarks()
        fitter = QPSCurveFitter(method=FitMethod.AUTO)
        report = fitter.fit(benchmarks)
        assert isinstance(report, QPSCurveReport)
        assert len(report.fits) == 3  # ttft, tpot, total_latency
        for fit in report.fits:
            assert fit.r_squared >= 0
            assert fit.residual_error >= 0
            assert fit.method in ("linear", "poly", "exp")

    def test_fit_linear(self) -> None:
        benchmarks = _make_benchmarks()
        fitter = QPSCurveFitter(method=FitMethod.LINEAR)
        report = fitter.fit(benchmarks)
        assert all(f.method == "linear" for f in report.fits)

    def test_fit_poly(self) -> None:
        benchmarks = _make_benchmarks()
        fitter = QPSCurveFitter(method=FitMethod.POLY)
        report = fitter.fit(benchmarks)
        assert all(f.method == "poly" for f in report.fits)

    def test_fit_exp(self) -> None:
        benchmarks = _make_benchmarks()
        fitter = QPSCurveFitter(method=FitMethod.EXP)
        report = fitter.fit(benchmarks)
        assert all(f.method == "exp" for f in report.fits)

    def test_predictions_interpolation(self) -> None:
        benchmarks = _make_benchmarks()
        fitter = QPSCurveFitter()
        report = fitter.fit(benchmarks, predict_qps=[25.0])
        assert len(report.predictions) == 3  # one per metric
        for pred in report.predictions:
            assert pred.qps == 25.0
            assert pred.predicted_latency_ms >= 0
            assert pred.confidence == Confidence.HIGH

    def test_predictions_extrapolation_medium(self) -> None:
        benchmarks = _make_benchmarks()  # QPS 10-50
        fitter = QPSCurveFitter()
        # 55 is within 20% of range (40) beyond max (50)
        report = fitter.fit(benchmarks, predict_qps=[55.0])
        for pred in report.predictions:
            assert pred.confidence == Confidence.MEDIUM

    def test_predictions_extrapolation_low(self) -> None:
        benchmarks = _make_benchmarks()  # QPS 10-50, range=40
        fitter = QPSCurveFitter()
        # 70 is 50% beyond max → LOW
        report = fitter.fit(benchmarks, predict_qps=[70.0])
        for pred in report.predictions:
            assert pred.confidence == Confidence.LOW

    def test_max_sustainable_qps(self) -> None:
        benchmarks = _make_benchmarks()
        fitter = QPSCurveFitter()
        report = fitter.fit(benchmarks, sla_ttft_ms=200.0)
        assert len(report.max_sustainable) >= 1
        for m in report.max_sustainable:
            assert m.max_qps > 0
            assert m.predicted_latency_at_max_ms <= m.sla_threshold_ms

    def test_max_sustainable_no_sla(self) -> None:
        benchmarks = _make_benchmarks()
        fitter = QPSCurveFitter()
        report = fitter.fit(benchmarks)
        assert len(report.max_sustainable) == 0
        assert "No SLA thresholds" in report.recommendation

    def test_max_sustainable_multiple_slas(self) -> None:
        benchmarks = _make_benchmarks()
        fitter = QPSCurveFitter()
        report = fitter.fit(benchmarks, sla_ttft_ms=200.0, sla_tpot_ms=50.0, sla_total_ms=800.0)
        assert len(report.max_sustainable) >= 1

    def test_qps_range_in_fits(self) -> None:
        benchmarks = _make_benchmarks([10.0, 30.0, 50.0])
        fitter = QPSCurveFitter()
        report = fitter.fit(benchmarks)
        for fit in report.fits:
            assert fit.qps_range[0] == 10.0
            assert fit.qps_range[1] == 50.0

    def test_two_benchmarks_minimum(self) -> None:
        benchmarks = _make_benchmarks([10.0, 50.0])
        fitter = QPSCurveFitter()
        report = fitter.fit(benchmarks)
        assert len(report.fits) == 3

    def test_percentile_99(self) -> None:
        benchmarks = _make_benchmarks()
        fitter = QPSCurveFitter(percentile=99.0)
        report = fitter.fit(benchmarks)
        for fit in report.fits:
            assert "p99" in fit.metric

    def test_recommendation_with_sla(self) -> None:
        benchmarks = _make_benchmarks()
        fitter = QPSCurveFitter()
        report = fitter.fit(benchmarks, sla_ttft_ms=200.0)
        assert "Maximum sustainable QPS" in report.recommendation

    def test_report_serializable(self) -> None:
        benchmarks = _make_benchmarks()
        fitter = QPSCurveFitter()
        report = fitter.fit(benchmarks, predict_qps=[25.0], sla_ttft_ms=200.0)
        d = report.model_dump()
        assert "fits" in d
        assert "predictions" in d
        assert "max_sustainable" in d
        assert "recommendation" in d


class TestFitQpsCurveAPI:
    """Tests for the programmatic API."""

    def test_basic(self) -> None:
        benchmarks = _make_benchmarks()
        result = fit_qps_curve(benchmarks)
        assert isinstance(result, dict)
        assert "fits" in result
        assert len(result["fits"]) == 3

    def test_with_predictions(self) -> None:
        benchmarks = _make_benchmarks()
        result = fit_qps_curve(benchmarks, predict_qps=[25.0, 35.0])
        assert len(result["predictions"]) == 6  # 2 QPS × 3 metrics

    def test_with_sla(self) -> None:
        benchmarks = _make_benchmarks()
        result = fit_qps_curve(benchmarks, sla_ttft_ms=200.0)
        assert len(result["max_sustainable"]) >= 1

    def test_method_linear(self) -> None:
        benchmarks = _make_benchmarks()
        result = fit_qps_curve(benchmarks, method="linear")
        assert all(f["method"] == "linear" for f in result["fits"])

    def test_method_poly(self) -> None:
        benchmarks = _make_benchmarks()
        result = fit_qps_curve(benchmarks, method="poly")
        assert all(f["method"] == "poly" for f in result["fits"])

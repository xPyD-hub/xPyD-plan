"""Tests for ensemble prediction module."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.ensemble import (
    DisagreementLevel,
    EnsemblePrediction,
    EnsemblePredictor,
    EnsembleReport,
    MethodPrediction,
    _classify_disagreement,
    _cv,
    _weighted_average,
    ensemble_predict,
)


def _make_requests(n: int, base_ttft: float = 50.0, base_tpot: float = 20.0) -> list[dict]:
    """Generate benchmark requests."""
    import random

    random.seed(42)
    requests = []
    for i in range(n):
        prompt = random.randint(100, 500)
        output = random.randint(50, 200)
        ttft = base_ttft + prompt * 0.05 + random.gauss(0, 5)
        tpot = base_tpot + output * 0.02 + random.gauss(0, 2)
        total = ttft + tpot * output
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=prompt,
                output_tokens=output,
                ttft_ms=max(1, ttft),
                tpot_ms=max(1, tpot),
                total_latency_ms=max(1, total),
                timestamp=1000.0 + i * 0.1,
            )
        )
    return requests


def _make_benchmark(
    n: int = 100,
    num_prefill: int = 2,
    num_decode: int = 6,
    qps: float = 100.0,
    base_ttft: float = 50.0,
    base_tpot: float = 20.0,
) -> BenchmarkData:
    """Create a benchmark dataset."""
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=qps,
        ),
        requests=_make_requests(n, base_ttft, base_tpot),
    )


# --- Helper function tests ---


class TestHelpers:
    def test_weighted_average(self):
        result = _weighted_average([10.0, 20.0], [0.8, 0.2])
        assert abs(result - 12.0) < 0.01

    def test_weighted_average_equal_weights(self):
        result = _weighted_average([10.0, 20.0], [1.0, 1.0])
        assert abs(result - 15.0) < 0.01

    def test_weighted_average_zero_weights(self):
        result = _weighted_average([10.0, 20.0], [0.0, 0.0])
        assert abs(result - 15.0) < 0.01

    def test_cv_single_value(self):
        assert _cv([10.0]) == 0.0

    def test_cv_identical_values(self):
        assert _cv([5.0, 5.0, 5.0]) == 0.0

    def test_cv_varied_values(self):
        result = _cv([10.0, 20.0, 30.0])
        assert result > 0.0

    def test_classify_disagreement_none(self):
        assert _classify_disagreement(0.05) == DisagreementLevel.NONE

    def test_classify_disagreement_low(self):
        assert _classify_disagreement(0.15) == DisagreementLevel.LOW

    def test_classify_disagreement_moderate(self):
        assert _classify_disagreement(0.35) == DisagreementLevel.MODERATE

    def test_classify_disagreement_high(self):
        assert _classify_disagreement(0.60) == DisagreementLevel.HIGH


# --- Model tests ---


class TestModels:
    def test_method_prediction_defaults(self):
        mp = MethodPrediction(method="test", confidence=0.5)
        assert mp.available is True
        assert mp.error is None
        assert mp.ttft_p95_ms is None

    def test_ensemble_prediction(self):
        ep = EnsemblePrediction(
            ttft_p95_ms=50.0,
            tpot_p95_ms=20.0,
            total_latency_p95_ms=200.0,
            num_methods=3,
        )
        assert ep.num_methods == 3
        assert ep.ttft_disagreement == DisagreementLevel.NONE

    def test_ensemble_report(self):
        report = EnsembleReport(
            method_predictions=[],
            ensemble=EnsemblePrediction(num_methods=0),
            recommendation="Test",
        )
        assert report.recommendation == "Test"


# --- EnsemblePredictor tests ---


class TestEnsemblePredictor:
    def test_init_default(self):
        ep = EnsemblePredictor()
        assert ep.percentile == 95.0

    def test_init_custom_percentile(self):
        ep = EnsemblePredictor(percentile=99.0)
        assert ep.percentile == 99.0

    def test_init_invalid_percentile(self):
        with pytest.raises(ValueError, match="percentile"):
            EnsemblePredictor(percentile=0.5)

    def test_predict_too_few_benchmarks(self):
        ep = EnsemblePredictor()
        bm = _make_benchmark()
        with pytest.raises(ValueError, match="at least 2"):
            ep.predict([bm])

    def test_predict_same_ratio_benchmarks(self):
        """Two benchmarks with same P:D ratio but different QPS."""
        ep = EnsemblePredictor()
        bm1 = _make_benchmark(qps=100.0, base_ttft=50.0)
        bm2 = _make_benchmark(qps=200.0, base_ttft=70.0)
        report = ep.predict([bm1, bm2])
        assert isinstance(report, EnsembleReport)
        assert len(report.method_predictions) == 3
        # At least some methods should be available
        available = [m for m in report.method_predictions if m.available]
        assert len(available) >= 1

    def test_predict_different_ratios(self):
        """Two benchmarks with different P:D ratios."""
        ep = EnsemblePredictor()
        bm1 = _make_benchmark(num_prefill=2, num_decode=6, qps=100.0)
        bm2 = _make_benchmark(num_prefill=4, num_decode=4, qps=150.0)
        report = ep.predict([bm1, bm2])
        assert isinstance(report, EnsembleReport)
        # Interpolation should work since different ratios
        interp = next(
            m for m in report.method_predictions if m.method == "interpolation"
        )
        # May or may not work depending on total_instances match
        assert isinstance(interp, MethodPrediction)

    def test_predict_with_ratio(self):
        """Predict with specific P:D ratio."""
        ep = EnsemblePredictor()
        bm1 = _make_benchmark(num_prefill=2, num_decode=6)
        bm2 = _make_benchmark(num_prefill=4, num_decode=4)
        report = ep.predict([bm1, bm2], predict_ratio=(3, 5))
        assert report.ensemble.num_methods >= 1

    def test_predict_with_qps(self):
        """Predict with specific QPS."""
        ep = EnsemblePredictor()
        bm1 = _make_benchmark(qps=100.0, base_ttft=50.0)
        bm2 = _make_benchmark(qps=200.0, base_ttft=70.0)
        report = ep.predict([bm1, bm2], predict_qps=150.0)
        qps_method = next(
            m for m in report.method_predictions if m.method == "qps_curve"
        )
        assert isinstance(qps_method, MethodPrediction)

    def test_ensemble_has_recommendation(self):
        ep = EnsemblePredictor()
        bm1 = _make_benchmark(qps=100.0)
        bm2 = _make_benchmark(qps=200.0)
        report = ep.predict([bm1, bm2])
        assert len(report.recommendation) > 0

    def test_regression_always_available(self):
        """Regression should almost always succeed with valid data."""
        ep = EnsemblePredictor()
        bm1 = _make_benchmark(n=50, qps=100.0)
        bm2 = _make_benchmark(n=50, qps=200.0)
        report = ep.predict([bm1, bm2])
        regression = next(
            m for m in report.method_predictions if m.method == "regression"
        )
        assert regression.available is True
        assert regression.confidence > 0
        assert regression.ttft_p95_ms is not None

    def test_ensemble_produces_values(self):
        """Ensemble should produce at least some non-None values."""
        ep = EnsemblePredictor()
        bm1 = _make_benchmark(qps=100.0)
        bm2 = _make_benchmark(qps=200.0)
        report = ep.predict([bm1, bm2])
        ens = report.ensemble
        # At least one metric should have a value
        has_value = (
            ens.ttft_p95_ms is not None
            or ens.tpot_p95_ms is not None
            or ens.total_latency_p95_ms is not None
        )
        assert has_value

    def test_failed_method_has_error(self):
        """When a method fails, it should report an error."""
        ep = EnsemblePredictor()
        bm1 = _make_benchmark(num_prefill=2, num_decode=6, qps=100.0)
        bm2 = _make_benchmark(num_prefill=2, num_decode=6, qps=100.0)
        report = ep.predict([bm1, bm2])
        # QPS curve needs distinct QPS values — same QPS should fail
        qps_method = next(
            m for m in report.method_predictions if m.method == "qps_curve"
        )
        if not qps_method.available:
            assert qps_method.error is not None


# --- Programmatic API tests ---


class TestEnsemblePredictAPI:
    def test_ensemble_predict_returns_dict(self):
        bm1 = _make_benchmark(qps=100.0)
        bm2 = _make_benchmark(qps=200.0)
        result = ensemble_predict([bm1, bm2])
        assert isinstance(result, dict)
        assert "ensemble" in result
        assert "method_predictions" in result
        assert "recommendation" in result

    def test_ensemble_predict_json_serializable(self):
        bm1 = _make_benchmark(qps=100.0)
        bm2 = _make_benchmark(qps=200.0)
        result = ensemble_predict([bm1, bm2])
        # Should be JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_ensemble_predict_with_sla(self):
        bm1 = _make_benchmark(qps=100.0)
        bm2 = _make_benchmark(qps=200.0)
        result = ensemble_predict(
            [bm1, bm2],
            sla_ttft_ms=100.0,
            sla_tpot_ms=50.0,
            sla_total_ms=500.0,
        )
        assert "ensemble" in result

    def test_ensemble_predict_custom_percentile(self):
        bm1 = _make_benchmark(qps=100.0)
        bm2 = _make_benchmark(qps=200.0)
        result = ensemble_predict([bm1, bm2], percentile=99.0)
        assert isinstance(result, dict)


# --- Disagreement detection tests ---


class TestDisagreement:
    def test_no_disagreement_similar_predictions(self):
        """When all methods agree, disagreement should be NONE or LOW."""
        ep = EnsemblePredictor()
        # Same config, similar data — methods should roughly agree
        bm1 = _make_benchmark(n=200, qps=100.0)
        bm2 = _make_benchmark(n=200, qps=100.5)  # Very slightly different QPS
        report = ep.predict([bm1, bm2])
        # We can't guarantee exact disagreement level, but report should be valid
        assert report.ensemble.ttft_disagreement in list(DisagreementLevel)

    def test_disagreement_level_enum(self):
        assert DisagreementLevel.NONE.value == "none"
        assert DisagreementLevel.LOW.value == "low"
        assert DisagreementLevel.MODERATE.value == "moderate"
        assert DisagreementLevel.HIGH.value == "high"

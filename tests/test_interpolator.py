"""Tests for the performance interpolation module."""

from __future__ import annotations

import numpy as np
import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.interpolator import (
    InterpolationConfidence,
    InterpolationMethod,
    InterpolationResult,
    PerformanceInterpolator,
    PredictedPerformance,
    _classify_confidence,
    interpolate_performance,
)


def _make_dataset(
    num_prefill: int,
    num_decode: int,
    qps: float,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    n_requests: int = 100,
) -> BenchmarkData:
    """Create a synthetic benchmark dataset."""
    total = num_prefill + num_decode
    rng = np.random.RandomState(num_prefill * 100 + num_decode)
    requests = []
    for i in range(n_requests):
        ttft = ttft_base + rng.exponential(ttft_base * 0.3)
        tpot = tpot_base + rng.exponential(tpot_base * 0.2)
        output_tokens = rng.randint(10, 200)
        total_lat = ttft + tpot * output_tokens
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=rng.randint(50, 500),
                output_tokens=output_tokens,
                ttft_ms=ttft,
                tpot_ms=tpot,
                total_latency_ms=total_lat,
                timestamp=1700000000.0 + i,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=total,
            measured_qps=qps,
        ),
        requests=requests,
    )


# --- PerformanceInterpolator tests ---


class TestPerformanceInterpolator:
    def test_fit_requires_at_least_2_datasets(self):
        interp = PerformanceInterpolator()
        ds = _make_dataset(2, 6, qps=100)
        with pytest.raises(ValueError, match="at least 2"):
            interp.fit([ds])

    def test_fit_requires_same_total_instances(self):
        interp = PerformanceInterpolator()
        ds1 = _make_dataset(2, 6, qps=100)  # total=8
        ds2 = _make_dataset(3, 7, qps=120)  # total=10
        with pytest.raises(ValueError, match="same total_instances"):
            interp.fit([ds1, ds2])

    def test_fit_and_predict_basic(self):
        interp = PerformanceInterpolator()
        ds1 = _make_dataset(2, 6, qps=100)
        ds2 = _make_dataset(4, 4, qps=80)
        interp.fit([ds1, ds2])
        assert interp.fitted

        result = interp.predict([(3, 5)], method=InterpolationMethod.LINEAR)
        assert isinstance(result, InterpolationResult)
        assert len(result.predictions) == 1
        pred = result.predictions[0]
        assert pred.num_prefill == 3
        assert pred.num_decode == 5
        assert pred.confidence == InterpolationConfidence.HIGH
        assert pred.ttft_p95_ms > 0
        assert pred.tpot_p95_ms > 0
        assert pred.throughput_qps > 0

    def test_predict_before_fit_raises(self):
        interp = PerformanceInterpolator()
        with pytest.raises(RuntimeError, match="Call fit"):
            interp.predict([(3, 5)])

    def test_predict_wrong_total_raises(self):
        interp = PerformanceInterpolator()
        interp.fit([_make_dataset(2, 6, qps=100), _make_dataset(4, 4, qps=80)])
        with pytest.raises(ValueError, match="total"):
            interp.predict([(3, 3)])  # total=6, expected 8

    def test_measured_point_is_marked(self):
        interp = PerformanceInterpolator()
        interp.fit([_make_dataset(2, 6, qps=100), _make_dataset(4, 4, qps=80)])
        result = interp.predict([(2, 6), (3, 5)])
        assert result.predictions[0].is_measured is True
        assert result.predictions[1].is_measured is False

    def test_extrapolation_confidence_medium(self):
        interp = PerformanceInterpolator()
        # fractions: 2/8=0.25, 4/8=0.5
        interp.fit([_make_dataset(2, 6, qps=100), _make_dataset(4, 4, qps=80)])
        # 5/8=0.625, range is 0.25..0.5, span=0.25, dist=0.125, ratio=0.5 => LOW
        # 1/8=0.125, dist from 0.25 = 0.125, ratio=0.5 => LOW
        # Actually for MEDIUM need ≤20%: dist/span ≤ 0.2
        # With range 0.25..0.5, span=0.25, need dist ≤ 0.05
        # That's fraction 0.20..0.25 or 0.50..0.55 — not integer ratios on 8 total
        # Let's use more data points to get a wider range
        pass  # Tested via _classify_confidence directly below

    def test_extrapolation_confidence_low(self):
        interp = PerformanceInterpolator()
        interp.fit([_make_dataset(3, 5, qps=100), _make_dataset(4, 4, qps=80)])
        # range: 3/8=0.375 .. 4/8=0.5, span=0.125
        # 1/8=0.125 => dist=0.25, ratio=2.0 => LOW
        result = interp.predict([(1, 7)])
        assert result.predictions[0].confidence == InterpolationConfidence.LOW

    def test_multiple_predictions(self):
        interp = PerformanceInterpolator()
        interp.fit([_make_dataset(1, 7, qps=120), _make_dataset(4, 4, qps=80)])
        result = interp.predict([(2, 6), (3, 5)])
        assert len(result.predictions) == 2
        assert result.best_predicted is not None

    def test_best_predicted_highest_throughput(self):
        interp = PerformanceInterpolator()
        interp.fit([_make_dataset(1, 7, qps=120), _make_dataset(4, 4, qps=80)])
        result = interp.predict([(1, 7), (2, 6), (3, 5), (4, 4)])
        # Best should be highest QPS
        qps_values = [p.throughput_qps for p in result.predictions]
        assert result.best_predicted is not None
        assert result.best_predicted.throughput_qps == max(qps_values)

    def test_spline_fallback_with_few_points(self):
        interp = PerformanceInterpolator()
        interp.fit([_make_dataset(2, 6, qps=100), _make_dataset(4, 4, qps=80)])
        result = interp.predict([(3, 5)], method=InterpolationMethod.SPLINE)
        # Should fall back to linear and note it
        assert result.method == InterpolationMethod.LINEAR
        assert any("fallback" in n.lower() or "spline" in n.lower() for n in result.notes)

    def test_spline_with_enough_points(self):
        interp = PerformanceInterpolator()
        datasets = [
            _make_dataset(1, 7, qps=120),
            _make_dataset(2, 6, qps=110),
            _make_dataset(3, 5, qps=95),
            _make_dataset(4, 4, qps=80),
        ]
        interp.fit(datasets)
        result = interp.predict([(1, 7), (2, 6)], method=InterpolationMethod.SPLINE)
        assert result.method == InterpolationMethod.SPLINE
        assert len(result.predictions) == 2

    def test_latencies_clamped_nonnegative(self):
        """Even with extrapolation, latencies should not go negative."""
        interp = PerformanceInterpolator()
        interp.fit([_make_dataset(3, 5, qps=100), _make_dataset(4, 4, qps=80)])
        # Extreme extrapolation
        result = interp.predict([(1, 7)])
        for pred in result.predictions:
            assert pred.ttft_p95_ms >= 0
            assert pred.tpot_p95_ms >= 0
            assert pred.total_latency_p95_ms >= 0
            assert pred.throughput_qps >= 0

    def test_duplicate_fraction_deduplication(self):
        """Two datasets with same P:D ratio should be deduplicated."""
        interp = PerformanceInterpolator()
        ds1 = _make_dataset(2, 6, qps=100)
        ds2 = _make_dataset(2, 6, qps=110)  # Same ratio, different QPS
        ds3 = _make_dataset(4, 4, qps=80)
        interp.fit([ds1, ds2, ds3])
        assert interp.fitted

    def test_predict_all_integer_ratios(self):
        interp = PerformanceInterpolator()
        interp.fit([_make_dataset(2, 6, qps=100), _make_dataset(6, 2, qps=80)])
        # Predict all 7 possible ratios for total=8
        ratios = [(p, 8 - p) for p in range(1, 8)]
        result = interp.predict(ratios)
        assert len(result.predictions) == 7


# --- _classify_confidence tests ---


class TestClassifyConfidence:
    def test_within_range(self):
        assert _classify_confidence(0.5, 0.25, 0.75) == InterpolationConfidence.HIGH

    def test_at_boundary(self):
        assert _classify_confidence(0.25, 0.25, 0.75) == InterpolationConfidence.HIGH
        assert _classify_confidence(0.75, 0.25, 0.75) == InterpolationConfidence.HIGH

    def test_near_extrapolation(self):
        # span=0.5, 20% of span=0.1
        assert _classify_confidence(0.15, 0.25, 0.75) == InterpolationConfidence.MEDIUM
        assert _classify_confidence(0.85, 0.25, 0.75) == InterpolationConfidence.MEDIUM

    def test_far_extrapolation(self):
        # span=0.5, >20% of span => >0.1 distance
        assert _classify_confidence(0.0, 0.25, 0.75) == InterpolationConfidence.LOW
        assert _classify_confidence(1.0, 0.25, 0.75) == InterpolationConfidence.LOW

    def test_zero_span(self):
        # All same x -> LOW for any different point
        assert _classify_confidence(0.5, 0.3, 0.3) == InterpolationConfidence.LOW


# --- interpolate_performance API tests ---


class TestInterpolatePerformance:
    def test_basic_api(self):
        datasets = [_make_dataset(2, 6, qps=100), _make_dataset(4, 4, qps=80)]
        result = interpolate_performance(datasets, target_ratios=[(3, 5)])
        assert isinstance(result, InterpolationResult)
        assert len(result.predictions) == 1
        assert result.measured_points == 2

    def test_auto_generate_all_ratios(self):
        datasets = [_make_dataset(2, 6, qps=100), _make_dataset(4, 4, qps=80)]
        result = interpolate_performance(datasets, target_ratios=None)
        # total=8, should generate 1:7 through 7:1 = 7 ratios
        assert len(result.predictions) == 7

    def test_with_spline(self):
        datasets = [
            _make_dataset(1, 7, qps=120),
            _make_dataset(2, 6, qps=110),
            _make_dataset(3, 5, qps=95),
            _make_dataset(4, 4, qps=80),
        ]
        result = interpolate_performance(
            datasets,
            target_ratios=[(2, 6), (3, 5)],
            method=InterpolationMethod.SPLINE,
        )
        assert result.method == InterpolationMethod.SPLINE
        assert len(result.predictions) == 2


# --- Pydantic model tests ---


class TestModels:
    def test_predicted_performance_serialization(self):
        pred = PredictedPerformance(
            num_prefill=2,
            num_decode=6,
            total_instances=8,
            prefill_fraction=0.25,
            ttft_p95_ms=65.0,
            tpot_p95_ms=12.0,
            total_latency_p95_ms=300.0,
            throughput_qps=100.0,
            confidence=InterpolationConfidence.HIGH,
            is_measured=True,
        )
        data = pred.model_dump()
        assert data["confidence"] == "high"
        assert data["is_measured"] is True

    def test_interpolation_result_serialization(self):
        result = InterpolationResult(
            method=InterpolationMethod.LINEAR,
            measured_points=3,
            total_instances=8,
            predictions=[],
            notes=["test note"],
        )
        data = result.model_dump()
        assert data["method"] == "linear"
        assert data["notes"] == ["test note"]

"""Tests for cross_validation module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.cross_validation import (
    CrossValidationReport,
    CrossValidator,
    ErrorMetric,
    ModelAccuracy,
    _classify_accuracy,
    _relative_error,
    cross_validate,
)
from xpyd_plan.interpolator import InterpolationMethod


def _make_benchmark(
    num_prefill: int,
    num_decode: int,
    qps: float,
    base_ttft: float,
    base_tpot: float,
    n_requests: int = 100,
) -> BenchmarkData:
    """Create a synthetic benchmark with predictable latency."""
    import numpy as np

    rng = np.random.default_rng(seed=num_prefill * 100 + num_decode)
    requests = []
    for i in range(n_requests):
        ttft = base_ttft + rng.normal(0, base_ttft * 0.05)
        tpot = base_tpot + rng.normal(0, base_tpot * 0.05)
        total = ttft + tpot * 50  # ~50 output tokens
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=rng.integers(100, 500),
                output_tokens=rng.integers(20, 100),
                ttft_ms=max(1.0, ttft),
                tpot_ms=max(0.1, tpot),
                total_latency_ms=max(1.0, total),
                timestamp=1700000000.0 + i * (1.0 / qps),
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


def _make_linear_benchmarks() -> list[BenchmarkData]:
    """Create benchmarks with roughly linear latency scaling."""
    # More prefill → lower TTFT, more decode → lower TPOT
    configs = [
        (1, 7, 100.0, 80.0, 20.0),  # P:D = 1:7
        (2, 6, 100.0, 50.0, 25.0),  # P:D = 2:6
        (3, 5, 100.0, 35.0, 32.0),  # P:D = 3:5
        (4, 4, 100.0, 28.0, 42.0),  # P:D = 4:4
        (5, 3, 100.0, 23.0, 58.0),  # P:D = 5:3
    ]
    return [_make_benchmark(p, d, q, t, o) for p, d, q, t, o in configs]


class TestRelativeError:
    """Tests for _relative_error helper."""

    def test_zero_actual(self) -> None:
        assert _relative_error(0.0, 10.0) == 0.0

    def test_exact_match(self) -> None:
        assert _relative_error(100.0, 100.0) == 0.0

    def test_positive_error(self) -> None:
        assert abs(_relative_error(100.0, 110.0) - 10.0) < 0.01

    def test_negative_error(self) -> None:
        assert abs(_relative_error(100.0, 90.0) - 10.0) < 0.01


class TestClassifyAccuracy:
    """Tests for _classify_accuracy."""

    def test_excellent(self) -> None:
        assert _classify_accuracy(3.0) == ModelAccuracy.EXCELLENT

    def test_good(self) -> None:
        assert _classify_accuracy(10.0) == ModelAccuracy.GOOD

    def test_fair(self) -> None:
        assert _classify_accuracy(20.0) == ModelAccuracy.FAIR

    def test_poor(self) -> None:
        assert _classify_accuracy(35.0) == ModelAccuracy.POOR


class TestCrossValidator:
    """Tests for CrossValidator class."""

    def test_requires_at_least_3_benchmarks(self) -> None:
        data = _make_linear_benchmarks()[:2]
        with pytest.raises(ValueError, match="at least 3"):
            CrossValidator(data)

    def test_basic_validation(self) -> None:
        data = _make_linear_benchmarks()
        cv = CrossValidator(data, method=InterpolationMethod.LINEAR)
        report = cv.validate()

        assert isinstance(report, CrossValidationReport)
        assert report.method == InterpolationMethod.LINEAR
        assert report.num_folds >= 3
        assert len(report.fold_errors) >= 3
        assert report.overall_mean_error_pct >= 0
        assert report.accuracy in list(ModelAccuracy)
        assert len(report.recommendations) > 0

    def test_fold_errors_have_correct_fields(self) -> None:
        data = _make_linear_benchmarks()
        cv = CrossValidator(data)
        report = cv.validate()

        for e in report.fold_errors:
            assert isinstance(e, ErrorMetric)
            assert e.num_prefill >= 1
            assert e.num_decode >= 1
            assert e.actual_ttft_p95_ms >= 0
            assert e.predicted_ttft_p95_ms >= 0
            assert e.ttft_error_pct >= 0
            assert e.tpot_error_pct >= 0
            assert e.total_error_pct >= 0
            assert e.qps_error_pct >= 0

    def test_worst_fold_gte_mean(self) -> None:
        data = _make_linear_benchmarks()
        report = CrossValidator(data).validate()
        # Worst fold should be >= overall mean (or close due to rounding)
        assert report.worst_fold_error_pct >= report.overall_mean_error_pct - 0.1

    def test_with_spline_method(self) -> None:
        data = _make_linear_benchmarks()
        report = CrossValidator(data, method=InterpolationMethod.SPLINE).validate()
        assert report.method == InterpolationMethod.SPLINE
        assert report.num_folds >= 3

    def test_three_benchmarks_minimum(self) -> None:
        data = _make_linear_benchmarks()[:3]
        report = CrossValidator(data).validate()
        assert report.num_folds >= 2  # some folds might not have predictions

    def test_recommendations_present(self) -> None:
        data = _make_linear_benchmarks()[:3]
        report = CrossValidator(data).validate()
        assert len(report.recommendations) > 0

    def test_report_serializable(self) -> None:
        data = _make_linear_benchmarks()
        report = CrossValidator(data).validate()
        d = report.model_dump()
        assert "method" in d
        assert "fold_errors" in d
        assert "accuracy" in d


class TestCrossValidateAPI:
    """Tests for the cross_validate() programmatic API."""

    def test_basic_api(self) -> None:
        data = _make_linear_benchmarks()
        # Write to temp files
        paths = []
        for i, d in enumerate(data):
            p = Path(tempfile.mktemp(suffix=".json"))
            p.write_text(json.dumps(d.model_dump(), default=str))
            paths.append(str(p))

        try:
            result = cross_validate(paths, method="linear")
            assert "accuracy" in result
            assert "fold_errors" in result
            assert result["num_folds"] >= 3
        finally:
            for p in paths:
                Path(p).unlink(missing_ok=True)

    def test_api_with_too_few_files(self) -> None:
        data = _make_linear_benchmarks()[:2]
        paths = []
        for d in data:
            p = Path(tempfile.mktemp(suffix=".json"))
            p.write_text(json.dumps(d.model_dump(), default=str))
            paths.append(str(p))

        try:
            with pytest.raises(ValueError, match="at least 3"):
                cross_validate(paths)
        finally:
            for p in paths:
                Path(p).unlink(missing_ok=True)


class TestModelAccuracy:
    """Tests for ModelAccuracy enum."""

    def test_values(self) -> None:
        assert ModelAccuracy.EXCELLENT.value == "excellent"
        assert ModelAccuracy.GOOD.value == "good"
        assert ModelAccuracy.FAIR.value == "fair"
        assert ModelAccuracy.POOR.value == "poor"

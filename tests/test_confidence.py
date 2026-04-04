"""Tests for bootstrap confidence interval analysis."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.confidence import (
    Adequacy,
    ConfidenceAnalyzer,
    ConfidenceReport,
    analyze_confidence,
)


def _make_requests(n: int, seed: int = 42) -> list[BenchmarkRequest]:
    """Generate n synthetic benchmark requests with deterministic values."""
    import numpy as np

    rng = np.random.default_rng(seed)
    requests = []
    for i in range(n):
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=float(rng.normal(40, 5)),
                tpot_ms=float(rng.normal(10, 2)),
                total_latency_ms=float(rng.normal(200, 30)),
                timestamp=1000.0 + i,
            )
        )
    return requests


def _make_benchmark(n: int = 100, seed: int = 42) -> BenchmarkData:
    """Create a BenchmarkData with n requests."""
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=4,
            total_instances=6,
            measured_qps=10.0,
        ),
        requests=_make_requests(n, seed=seed),
    )


class TestConfidenceAnalyzer:
    """Tests for ConfidenceAnalyzer."""

    def test_basic_analysis(self):
        data = _make_benchmark(100)
        analyzer = ConfidenceAnalyzer(seed=123)
        report = analyzer.analyze(data)

        assert isinstance(report, ConfidenceReport)
        assert len(report.metrics) == 3
        assert report.sample_size == 100
        assert report.confidence_level == 0.95
        assert report.percentile == 95.0

    def test_three_metrics(self):
        data = _make_benchmark(200)
        report = analyze_confidence(data, seed=42)
        metric_names = [m.metric for m in report.metrics]
        assert metric_names == ["ttft_ms", "tpot_ms", "total_latency_ms"]

    def test_ci_bounds_order(self):
        """CI lower should be <= point estimate <= CI upper."""
        data = _make_benchmark(500)
        report = analyze_confidence(data, seed=42)
        for mc in report.metrics:
            ci = mc.intervals[0]
            assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper

    def test_ci_width_positive(self):
        data = _make_benchmark(100)
        report = analyze_confidence(data, seed=42)
        for mc in report.metrics:
            ci = mc.intervals[0]
            assert ci.ci_width >= 0

    def test_larger_sample_tighter_ci(self):
        """More data should produce tighter confidence intervals."""
        small = analyze_confidence(_make_benchmark(30, seed=1), seed=42)
        large = analyze_confidence(_make_benchmark(1000, seed=1), seed=42)

        for s_mc, l_mc in zip(small.metrics, large.metrics):
            # Larger sample should generally have smaller relative CI width
            # Use a generous comparison since bootstrap has randomness
            assert l_mc.intervals[0].relative_ci_width <= s_mc.intervals[0].relative_ci_width + 0.1

    def test_single_request(self):
        """Single request should produce zero-width CI."""
        data = _make_benchmark(1)
        report = analyze_confidence(data, seed=42)
        for mc in report.metrics:
            ci = mc.intervals[0]
            assert ci.ci_width == 0.0
            assert ci.relative_ci_width == 0.0
            assert ci.ci_lower == ci.ci_upper == ci.point_estimate

    def test_small_sample_warning(self):
        data = _make_benchmark(5)
        report = analyze_confidence(data, seed=42)
        assert any("small sample" in w.lower() for w in report.warnings)

    def test_empty_data_raises(self):
        """BenchmarkData with empty requests is rejected by Pydantic validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BenchmarkData(
                metadata=BenchmarkMetadata(
                    num_prefill_instances=1,
                    num_decode_instances=1,
                    total_instances=2,
                    measured_qps=1.0,
                ),
                requests=[],
            )

    def test_custom_percentile(self):
        data = _make_benchmark(200)
        report = analyze_confidence(data, percentile=99.0, seed=42)
        assert report.percentile == 99.0
        for mc in report.metrics:
            assert mc.intervals[0].percentile == 99.0

    def test_custom_confidence_level(self):
        data = _make_benchmark(200)
        report = analyze_confidence(data, confidence_level=0.90, seed=42)
        assert report.confidence_level == 0.90

    def test_higher_confidence_wider_ci(self):
        """99% CI should be wider than 90% CI."""
        data = _make_benchmark(200)
        r90 = analyze_confidence(data, confidence_level=0.90, seed=42)
        r99 = analyze_confidence(data, confidence_level=0.99, seed=42)
        for m90, m99 in zip(r90.metrics, r99.metrics):
            assert m99.intervals[0].ci_width >= m90.intervals[0].ci_width

    def test_reproducibility_with_seed(self):
        data = _make_benchmark(100)
        r1 = analyze_confidence(data, seed=42)
        r2 = analyze_confidence(data, seed=42)
        for m1, m2 in zip(r1.metrics, r2.metrics):
            assert m1.intervals[0].ci_lower == m2.intervals[0].ci_lower
            assert m1.intervals[0].ci_upper == m2.intervals[0].ci_upper

    def test_different_seeds_different_results(self):
        data = _make_benchmark(100)
        r1 = analyze_confidence(data, seed=1)
        r2 = analyze_confidence(data, seed=2)
        # At least one metric should differ
        any_diff = any(
            m1.intervals[0].ci_lower != m2.intervals[0].ci_lower
            for m1, m2 in zip(r1.metrics, r2.metrics)
        )
        assert any_diff

    def test_adequacy_classification_sufficient(self):
        data = _make_benchmark(1000)
        report = analyze_confidence(data, seed=42)
        assert report.adequate is True
        for mc in report.metrics:
            assert mc.adequacy == Adequacy.SUFFICIENT

    def test_invalid_confidence_level(self):
        with pytest.raises(ValueError, match="confidence_level"):
            ConfidenceAnalyzer(confidence_level=0.0)
        with pytest.raises(ValueError, match="confidence_level"):
            ConfidenceAnalyzer(confidence_level=1.0)

    def test_invalid_iterations(self):
        with pytest.raises(ValueError, match="iterations"):
            ConfidenceAnalyzer(iterations=5)

    def test_invalid_percentile(self):
        data = _make_benchmark(100)
        analyzer = ConfidenceAnalyzer(seed=42)
        with pytest.raises(ValueError, match="percentile"):
            analyzer.analyze(data, percentile=0.0)
        with pytest.raises(ValueError, match="percentile"):
            analyzer.analyze(data, percentile=101.0)

    def test_model_serialization(self):
        data = _make_benchmark(100)
        report = analyze_confidence(data, seed=42)
        d = report.model_dump()
        assert "metrics" in d
        assert "adequate" in d
        # Should be JSON-serializable
        json.dumps(d)

    def test_iterations_parameter(self):
        data = _make_benchmark(100)
        report = analyze_confidence(data, iterations=50, seed=42)
        assert report.iterations == 50
        for mc in report.metrics:
            assert mc.intervals[0].iterations == 50

    def test_p50_percentile(self):
        data = _make_benchmark(200)
        report = analyze_confidence(data, percentile=50.0, seed=42)
        assert report.percentile == 50.0

    def test_convenience_function_matches_class(self):
        data = _make_benchmark(100)
        r1 = analyze_confidence(data, seed=42)
        analyzer = ConfidenceAnalyzer(seed=42)
        r2 = analyzer.analyze(data)
        for m1, m2 in zip(r1.metrics, r2.metrics):
            assert m1.intervals[0].point_estimate == m2.intervals[0].point_estimate
            assert m1.intervals[0].ci_lower == m2.intervals[0].ci_lower

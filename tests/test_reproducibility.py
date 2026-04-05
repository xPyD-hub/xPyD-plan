"""Tests for benchmark reproducibility analysis."""

from __future__ import annotations

import random

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.reproducibility import (
    ReproducibilityAnalyzer,
    ReproducibilityGrade,
    analyze_reproducibility,
)


def _make_data(
    n: int = 100,
    seed: int = 42,
    total_instances: int = 4,
    prefill: int = 2,
    decode: int = 2,
    qps: float = 50.0,
    latency_scale: float = 1.0,
) -> BenchmarkData:
    """Create benchmark data with controllable noise."""
    rng = random.Random(seed)
    requests = []
    base_ts = 1700000000.0
    for i in range(n):
        pt = rng.randint(50, 2000)
        ot = rng.randint(10, 500)
        ttft = (pt * 0.05 + rng.uniform(5, 30)) * latency_scale
        tpot = (ot * 0.03 + rng.uniform(2, 15)) * latency_scale
        total = (ttft + tpot * ot + rng.uniform(10, 200)) * latency_scale
        ts = base_ts + i / qps
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=pt,
                output_tokens=ot,
                ttft_ms=ttft,
                tpot_ms=tpot,
                total_latency_ms=total,
                timestamp=ts,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=prefill,
            num_decode_instances=decode,
            total_instances=total_instances,
            measured_qps=qps,
        ),
        requests=requests,
    )


class TestReproducibilityAnalyzer:
    def test_basic_analysis(self) -> None:
        d1 = _make_data(100, seed=1)
        d2 = _make_data(100, seed=2)
        report = analyze_reproducibility([d1, d2])
        assert report.num_runs == 2
        assert report.total_requests == 200
        assert 0 <= report.composite_score <= 100
        assert isinstance(report.grade, ReproducibilityGrade)

    def test_identical_runs_high_score(self) -> None:
        d1 = _make_data(100, seed=42)
        d2 = _make_data(100, seed=42)
        report = analyze_reproducibility([d1, d2])
        assert report.composite_score >= 90
        assert report.grade == ReproducibilityGrade.EXCELLENT

    def test_very_different_runs_low_score(self) -> None:
        d1 = _make_data(100, seed=1, latency_scale=1.0)
        d2 = _make_data(100, seed=2, latency_scale=5.0)
        report = analyze_reproducibility([d1, d2])
        assert report.composite_score < 75

    def test_three_runs(self) -> None:
        runs = [_make_data(50, seed=i) for i in range(3)]
        report = analyze_reproducibility(runs)
        assert report.num_runs == 3
        assert len(report.pair_tests) > 0

    def test_fewer_than_two_raises(self) -> None:
        with pytest.raises(ValueError, match="At least 2"):
            analyze_reproducibility([_make_data(50)])

    def test_metrics_count(self) -> None:
        report = analyze_reproducibility([_make_data(50, seed=1), _make_data(50, seed=2)])
        assert len(report.metrics) == 10  # 9 latency percentiles + QPS

    def test_metric_fields(self) -> None:
        report = analyze_reproducibility([_make_data(50, seed=1), _make_data(50, seed=2)])
        for m in report.metrics:
            assert len(m.values) == 2
            assert m.mean > 0
            assert m.std >= 0
            assert m.cv >= 0

    def test_pair_tests_count_two_runs(self) -> None:
        report = analyze_reproducibility([_make_data(50, seed=1), _make_data(50, seed=2)])
        # 1 pair × 3 metrics = 3
        assert len(report.pair_tests) == 3

    def test_pair_tests_count_three_runs(self) -> None:
        runs = [_make_data(50, seed=i) for i in range(3)]
        report = analyze_reproducibility(runs)
        # 3 pairs × 3 metrics = 9
        assert len(report.pair_tests) == 9

    def test_ks_test_identical_consistent(self) -> None:
        d = _make_data(100, seed=42)
        report = analyze_reproducibility([d, d])
        for t in report.pair_tests:
            assert t.distributions_consistent is True
            assert t.p_value >= 0.05

    def test_unreliable_metrics(self) -> None:
        d1 = _make_data(100, seed=1, latency_scale=1.0)
        d2 = _make_data(100, seed=2, latency_scale=3.0)
        report = analyze_reproducibility([d1, d2], cv_threshold=0.05)
        assert len(report.unreliable_metrics) > 0

    def test_custom_cv_threshold(self) -> None:
        d1 = _make_data(100, seed=1)
        d2 = _make_data(100, seed=2)
        strict = analyze_reproducibility([d1, d2], cv_threshold=0.001)
        lenient = analyze_reproducibility([d1, d2], cv_threshold=1.0)
        assert len(strict.unreliable_metrics) >= len(lenient.unreliable_metrics)

    def test_invalid_cv_threshold(self) -> None:
        with pytest.raises(ValueError, match="cv_threshold must be > 0"):
            ReproducibilityAnalyzer(cv_threshold=0)

    def test_recommended_min_runs(self) -> None:
        report = analyze_reproducibility([_make_data(50, seed=1), _make_data(50, seed=2)])
        assert report.recommended_min_runs >= 3

    def test_grade_excellent(self) -> None:
        d = _make_data(100, seed=42)
        report = analyze_reproducibility([d, d])
        assert report.grade == ReproducibilityGrade.EXCELLENT

    def test_json_serialization(self) -> None:
        report = analyze_reproducibility([_make_data(50, seed=1), _make_data(50, seed=2)])
        d = report.model_dump()
        assert "metrics" in d
        assert "pair_tests" in d
        assert "composite_score" in d
        assert "grade" in d

    def test_programmatic_api(self) -> None:
        report = analyze_reproducibility(
            [_make_data(50, seed=1), _make_data(50, seed=2)],
            cv_threshold=0.15,
        )
        assert isinstance(report.num_runs, int)

    def test_five_runs(self) -> None:
        runs = [_make_data(30, seed=i) for i in range(5)]
        report = analyze_reproducibility(runs)
        assert report.num_runs == 5
        # 10 pairs × 3 metrics = 30
        assert len(report.pair_tests) == 30

    def test_pair_test_fields(self) -> None:
        report = analyze_reproducibility([_make_data(50, seed=1), _make_data(50, seed=2)])
        for t in report.pair_tests:
            assert t.run_a < t.run_b
            assert 0 <= t.ks_statistic <= 1
            assert 0 <= t.p_value <= 1
            assert t.metric in ("ttft_ms", "tpot_ms", "total_latency_ms")

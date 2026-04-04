"""Tests for multi-benchmark statistical summary."""

from __future__ import annotations

import json
import random

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.stat_summary import (
    AggregatedStats,
    LatencyAggStats,
    RunStability,
    StatSummaryAnalyzer,
    StatSummaryReport,
    summarize_stats,
)


def _make_benchmark(
    n_requests: int = 100,
    qps: float = 50.0,
    ttft_base: float = 20.0,
    tpot_base: float = 5.0,
    total_base: float = 100.0,
    noise: float = 5.0,
    seed: int = 42,
    prefill: int = 2,
    decode: int = 6,
) -> BenchmarkData:
    """Create a synthetic benchmark dataset."""
    rng = random.Random(seed)
    requests = []
    for i in range(n_requests):
        requests.append(BenchmarkRequest(
            request_id=f"req-{i}",
            prompt_tokens=rng.randint(50, 500),
            output_tokens=rng.randint(10, 200),
            ttft_ms=max(1.0, ttft_base + rng.gauss(0, noise)),
            tpot_ms=max(0.1, tpot_base + rng.gauss(0, noise * 0.5)),
            total_latency_ms=max(1.0, total_base + rng.gauss(0, noise * 2)),
            timestamp=1000.0 + i * (1.0 / qps),
        ))
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=prefill,
            num_decode_instances=decode,
            total_instances=prefill + decode,
            measured_qps=qps,
        ),
        requests=requests,
    )


class TestStatSummaryAnalyzer:
    """Test StatSummaryAnalyzer."""

    def test_basic_two_runs(self):
        d1 = _make_benchmark(seed=1)
        d2 = _make_benchmark(seed=2)
        analyzer = StatSummaryAnalyzer([d1, d2])
        report = analyzer.summarize()

        assert isinstance(report, StatSummaryReport)
        assert len(report.runs) == 2
        assert report.aggregated.num_runs == 2

    def test_requires_at_least_two(self):
        d1 = _make_benchmark(seed=1)
        with pytest.raises(ValueError, match="At least 2"):
            StatSummaryAnalyzer([d1])

    def test_run_summaries_populated(self):
        datasets = [_make_benchmark(seed=i) for i in range(4)]
        report = StatSummaryAnalyzer(datasets).summarize()

        assert len(report.runs) == 4
        for i, run in enumerate(report.runs):
            assert run.index == i
            assert run.request_count == 100
            assert run.measured_qps == 50.0
            assert run.ttft_p95_ms > 0
            assert run.tpot_p95_ms > 0
            assert run.total_latency_p95_ms > 0

    def test_aggregated_stats_structure(self):
        datasets = [_make_benchmark(seed=i) for i in range(3)]
        report = StatSummaryAnalyzer(datasets).summarize()
        agg = report.aggregated

        assert isinstance(agg, AggregatedStats)
        assert agg.num_runs == 3
        assert agg.total_requests == 300
        assert agg.mean_qps == 50.0
        assert isinstance(agg.ttft, LatencyAggStats)
        assert isinstance(agg.tpot, LatencyAggStats)
        assert isinstance(agg.total_latency, LatencyAggStats)

    def test_latency_agg_stats_values(self):
        datasets = [_make_benchmark(seed=i) for i in range(5)]
        report = StatSummaryAnalyzer(datasets).summarize()
        ttft = report.aggregated.ttft

        assert ttft.mean_of_means_ms > 0
        assert ttft.mean_of_p50s_ms > 0
        assert ttft.mean_of_p95s_ms > 0
        assert ttft.mean_of_p99s_ms > 0
        assert ttft.cv_of_p95s >= 0

    def test_cv_qps_zero_for_identical_runs(self):
        datasets = [_make_benchmark(seed=i) for i in range(3)]
        report = StatSummaryAnalyzer(datasets).summarize()
        # All have qps=50.0 so std=0, cv=0
        assert report.aggregated.cv_qps == 0.0

    def test_cv_qps_positive_for_varying_runs(self):
        d1 = _make_benchmark(qps=40.0, seed=1)
        d2 = _make_benchmark(qps=60.0, seed=2)
        d3 = _make_benchmark(qps=50.0, seed=3)
        report = StatSummaryAnalyzer([d1, d2, d3]).summarize()
        assert report.aggregated.cv_qps > 0

    def test_stability_classification(self):
        # Make one run with high noise (unstable) and others low noise
        d_stable1 = _make_benchmark(noise=1.0, seed=1)
        d_stable2 = _make_benchmark(noise=1.0, seed=2)
        d_unstable = _make_benchmark(noise=50.0, seed=3)
        report = StatSummaryAnalyzer([d_stable1, d_stable2, d_unstable]).summarize()

        stabilities = [r.stability for r in report.runs]
        assert RunStability.MOST_STABLE in stabilities
        assert RunStability.LEAST_STABLE in stabilities

    def test_most_stable_least_stable_indices(self):
        datasets = [_make_benchmark(seed=i) for i in range(4)]
        report = StatSummaryAnalyzer(datasets).summarize()
        assert 0 <= report.most_stable_run < 4
        assert 0 <= report.least_stable_run < 4
        assert report.runs[report.most_stable_run].stability == RunStability.MOST_STABLE
        assert report.runs[report.least_stable_run].stability == RunStability.LEAST_STABLE

    def test_repeatability_cv(self):
        datasets = [_make_benchmark(seed=i) for i in range(3)]
        report = StatSummaryAnalyzer(datasets).summarize()
        assert report.aggregated.repeatability_cv >= 0

    def test_total_requests(self):
        d1 = _make_benchmark(n_requests=50, seed=1)
        d2 = _make_benchmark(n_requests=150, seed=2)
        report = StatSummaryAnalyzer([d1, d2]).summarize()
        assert report.aggregated.total_requests == 200

    def test_many_runs(self):
        datasets = [_make_benchmark(seed=i) for i in range(10)]
        report = StatSummaryAnalyzer(datasets).summarize()
        assert report.aggregated.num_runs == 10
        assert len(report.runs) == 10

    def test_model_dump_serializable(self):
        datasets = [_make_benchmark(seed=i) for i in range(3)]
        report = StatSummaryAnalyzer(datasets).summarize()
        d = report.model_dump()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0


class TestSummarizeStatsAPI:
    """Test the programmatic API."""

    def test_returns_dict(self):
        datasets = [_make_benchmark(seed=i) for i in range(3)]
        result = summarize_stats(datasets)
        assert isinstance(result, dict)
        assert "runs" in result
        assert "aggregated" in result
        assert "most_stable_run" in result
        assert "least_stable_run" in result

    def test_raises_on_single_dataset(self):
        with pytest.raises(ValueError):
            summarize_stats([_make_benchmark(seed=1)])

    def test_aggregated_keys(self):
        datasets = [_make_benchmark(seed=i) for i in range(2)]
        result = summarize_stats(datasets)
        agg = result["aggregated"]
        assert "num_runs" in agg
        assert "mean_qps" in agg
        assert "repeatability_cv" in agg
        assert "ttft" in agg
        assert "tpot" in agg
        assert "total_latency" in agg

    def test_run_summaries_in_result(self):
        datasets = [_make_benchmark(seed=i) for i in range(4)]
        result = summarize_stats(datasets)
        assert len(result["runs"]) == 4
        for run in result["runs"]:
            assert "index" in run
            assert "request_count" in run
            assert "stability" in run


class TestEdgeCases:
    """Test edge cases."""

    def test_different_request_counts(self):
        d1 = _make_benchmark(n_requests=10, seed=1)
        d2 = _make_benchmark(n_requests=1000, seed=2)
        report = StatSummaryAnalyzer([d1, d2]).summarize()
        assert report.runs[0].request_count == 10
        assert report.runs[1].request_count == 1000

    def test_different_qps(self):
        d1 = _make_benchmark(qps=10.0, seed=1)
        d2 = _make_benchmark(qps=100.0, seed=2)
        report = StatSummaryAnalyzer([d1, d2]).summarize()
        assert report.aggregated.mean_qps == pytest.approx(55.0)

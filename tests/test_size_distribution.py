"""Tests for request size distribution analysis."""

from __future__ import annotations

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.size_distribution import (
    DistributionShape,
    SizeDistributionAnalyzer,
    analyze_size_distribution,
)


def _make_data(n: int = 200, seed: int = 42) -> BenchmarkData:
    """Create benchmark data with n requests."""
    import random as _random

    rng = _random.Random(seed)
    requests = []
    for i in range(n):
        pt = rng.randint(50, 2000)
        ot = rng.randint(10, 500)
        # Make latency correlate with token counts for testability
        ttft = pt * 0.1 + rng.uniform(0, 20)
        tpot = ot * 0.05 + rng.uniform(0, 10)
        total = ttft + tpot * ot + rng.uniform(0, 100)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=pt,
                output_tokens=ot,
                ttft_ms=ttft,
                tpot_ms=tpot,
                total_latency_ms=total,
                timestamp=1700000000 + i * 0.5,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=100.0,
        ),
        requests=requests,
    )


class TestSizeDistributionAnalyzer:
    def test_basic_analyze(self) -> None:
        data = _make_data(200)
        analyzer = SizeDistributionAnalyzer(bins=5)
        report = analyzer.analyze(data)
        assert report.total_requests == 200
        assert len(report.prompt_histogram.bins) == 5
        assert len(report.output_histogram.bins) == 5
        assert report.prompt_histogram.dimension == "prompt"
        assert report.output_histogram.dimension == "output"

    def test_bin_counts_sum_to_total(self) -> None:
        data = _make_data(100)
        report = analyze_size_distribution(data, bins=8)
        prompt_total = sum(b.count for b in report.prompt_histogram.bins)
        output_total = sum(b.count for b in report.output_histogram.bins)
        assert prompt_total == 100
        assert output_total == 100

    def test_fractions_sum_to_one(self) -> None:
        data = _make_data(100)
        report = analyze_size_distribution(data, bins=5)
        prompt_frac = sum(b.fraction for b in report.prompt_histogram.bins)
        output_frac = sum(b.fraction for b in report.output_histogram.bins)
        assert prompt_frac == pytest.approx(1.0, abs=0.01)
        assert output_frac == pytest.approx(1.0, abs=0.01)

    def test_latency_stats_present_for_nonempty_bins(self) -> None:
        data = _make_data(200)
        report = analyze_size_distribution(data, bins=5)
        for b in report.prompt_histogram.bins:
            if b.count > 0:
                assert b.ttft_p50_ms is not None
                assert b.ttft_p95_ms is not None
                assert b.tpot_p50_ms is not None
                assert b.tpot_p95_ms is not None
                assert b.total_latency_p50_ms is not None
                assert b.total_latency_p95_ms is not None

    def test_latency_stats_none_for_empty_bins(self) -> None:
        # With 2 requests and 10 bins, most bins will be empty
        data = _make_data(2)
        report = analyze_size_distribution(data, bins=10)
        empty_bins = [b for b in report.prompt_histogram.bins if b.count == 0]
        for b in empty_bins:
            assert b.ttft_p50_ms is None
            assert b.total_latency_p50_ms is None

    def test_correlations_present(self) -> None:
        data = _make_data(200)
        report = analyze_size_distribution(data)
        assert len(report.correlations) == 2
        dims = {c.dimension for c in report.correlations}
        assert dims == {"prompt", "output"}

    def test_prompt_ttft_correlation_positive(self) -> None:
        """Since we made ttft ~ prompt_tokens * 0.1, correlation should be positive."""
        data = _make_data(500, seed=1)
        report = analyze_size_distribution(data)
        prompt_corr = next(c for c in report.correlations if c.dimension == "prompt")
        assert prompt_corr.ttft_pearson_r > 0.5

    def test_histogram_stats(self) -> None:
        data = _make_data(200)
        report = analyze_size_distribution(data, bins=5)
        h = report.prompt_histogram
        assert h.min_tokens <= h.mean_tokens <= h.max_tokens
        assert h.std_tokens >= 0

    def test_shape_classification_exists(self) -> None:
        data = _make_data(200)
        report = analyze_size_distribution(data, bins=5)
        assert report.prompt_histogram.shape in DistributionShape
        assert report.output_histogram.shape in DistributionShape

    def test_bins_must_be_at_least_2(self) -> None:
        with pytest.raises(ValueError, match="bins must be >= 2"):
            SizeDistributionAnalyzer(bins=1)

    def test_empty_data_raises(self) -> None:
        """BenchmarkData enforces min 1 request, so we test via direct call."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BenchmarkData(
                metadata=BenchmarkMetadata(
                    num_prefill_instances=1,
                    num_decode_instances=1,
                    total_instances=2,
                    measured_qps=10.0,
                ),
                requests=[],
            )

    def test_default_bins_is_10(self) -> None:
        data = _make_data(200)
        report = analyze_size_distribution(data)
        assert len(report.prompt_histogram.bins) == 10

    def test_single_request(self) -> None:
        data = _make_data(1)
        report = analyze_size_distribution(data, bins=3)
        assert report.total_requests == 1
        nonempty = [b for b in report.prompt_histogram.bins if b.count > 0]
        assert len(nonempty) >= 1
        total = sum(b.count for b in report.prompt_histogram.bins)
        assert total == 1

    def test_programmatic_api(self) -> None:
        data = _make_data(50)
        report = analyze_size_distribution(data, bins=4)
        assert report.total_requests == 50
        assert len(report.prompt_histogram.bins) == 4

    def test_model_dump_serializable(self) -> None:
        """Ensure report can be serialized to dict/JSON."""
        import json

        data = _make_data(50)
        report = analyze_size_distribution(data, bins=4)
        d = report.model_dump()
        assert isinstance(d, dict)
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_large_bins(self) -> None:
        data = _make_data(20)
        report = analyze_size_distribution(data, bins=20)
        total = sum(b.count for b in report.prompt_histogram.bins)
        assert total == 20

    def test_p95_ge_p50(self) -> None:
        data = _make_data(200)
        report = analyze_size_distribution(data, bins=5)
        for b in report.prompt_histogram.bins:
            if b.count > 1:
                assert b.ttft_p95_ms >= b.ttft_p50_ms
                assert b.tpot_p95_ms >= b.tpot_p50_ms
                assert b.total_latency_p95_ms >= b.total_latency_p50_ms

    def test_correlation_range(self) -> None:
        data = _make_data(200)
        report = analyze_size_distribution(data)
        for c in report.correlations:
            assert -1.0 <= c.ttft_pearson_r <= 1.0
            assert -1.0 <= c.tpot_pearson_r <= 1.0
            assert -1.0 <= c.total_latency_pearson_r <= 1.0

    def test_constant_tokens_correlation_zero(self) -> None:
        """When all prompt tokens are the same, correlation should be 0."""
        requests = [
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=100,
                ttft_ms=float(i + 10),
                tpot_ms=float(i + 5),
                total_latency_ms=float(i * 10 + 100),
                timestamp=1700000000 + i,
            )
            for i in range(50)
        ]
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=10.0,
            ),
            requests=requests,
        )
        report = analyze_size_distribution(data, bins=3)
        prompt_corr = next(c for c in report.correlations if c.dimension == "prompt")
        assert prompt_corr.ttft_pearson_r == pytest.approx(0.0, abs=0.01)

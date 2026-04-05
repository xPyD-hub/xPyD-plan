"""Tests for latency percentile comparison across P:D ratios."""

from __future__ import annotations

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.ratio_compare import (
    PercentileStats,
    RatioComparer,
    RatioComparison,
    compare_ratios,
)


def _make_request(
    request_id: str = "r1",
    prompt_tokens: int = 100,
    output_tokens: int = 50,
    ttft_ms: float = 20.0,
    tpot_ms: float = 10.0,
    total_latency_ms: float = 520.0,
    timestamp: float = 1000.0,
) -> BenchmarkRequest:
    return BenchmarkRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_latency_ms=total_latency_ms,
        timestamp=timestamp,
    )


def _make_data(
    requests: list[BenchmarkRequest],
    num_prefill: int = 2,
    num_decode: int = 2,
    qps: float = 10.0,
) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=qps,
        ),
        requests=requests,
    )


def _make_dataset(
    num_prefill: int,
    num_decode: int,
    ttft_base: float = 20.0,
    tpot_base: float = 10.0,
    total_base: float = 500.0,
    n: int = 100,
    qps: float = 10.0,
) -> BenchmarkData:
    """Create a dataset with some variance around base values."""
    requests = []
    for i in range(n):
        # Add some variance
        factor = 1.0 + (i % 10) * 0.1
        requests.append(
            _make_request(
                request_id=f"r{i}",
                ttft_ms=ttft_base * factor,
                tpot_ms=tpot_base * factor,
                total_latency_ms=total_base * factor,
                timestamp=1000.0 + i,
            )
        )
    return _make_data(requests, num_prefill=num_prefill, num_decode=num_decode, qps=qps)


class TestRatioComparerBasic:
    """Basic comparison tests."""

    def test_two_ratios(self):
        d1 = _make_dataset(3, 1, ttft_base=30.0)
        d2 = _make_dataset(1, 3, ttft_base=20.0)
        report = RatioComparer().compare([d1, d2])
        assert len(report.ratios) == 2
        assert report.ratios[0].ratio_label == "3:1"
        assert report.ratios[1].ratio_label == "1:3"

    def test_three_ratios(self):
        d1 = _make_dataset(3, 1)
        d2 = _make_dataset(2, 2)
        d3 = _make_dataset(1, 3)
        report = RatioComparer().compare([d1, d2, d3])
        assert len(report.ratios) == 3

    def test_requires_at_least_two(self):
        d1 = _make_dataset(2, 2)
        with pytest.raises(ValueError, match="At least 2"):
            RatioComparer().compare([d1])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            RatioComparer().compare([])


class TestRatioMetrics:
    """Test that per-ratio metrics are computed correctly."""

    def test_ratio_label(self):
        d = _make_dataset(4, 2)
        d2 = _make_dataset(2, 4)
        report = RatioComparer().compare([d, d2])
        assert report.ratios[0].ratio_label == "4:2"
        assert report.ratios[0].num_prefill == 4
        assert report.ratios[0].num_decode == 2

    def test_request_count(self):
        d1 = _make_dataset(2, 2, n=50)
        d2 = _make_dataset(3, 1, n=75)
        report = RatioComparer().compare([d1, d2])
        assert report.ratios[0].request_count == 50
        assert report.ratios[1].request_count == 75

    def test_measured_qps(self):
        d1 = _make_dataset(2, 2, qps=15.0)
        d2 = _make_dataset(3, 1, qps=20.0)
        report = RatioComparer().compare([d1, d2])
        assert report.ratios[0].measured_qps == 15.0
        assert report.ratios[1].measured_qps == 20.0

    def test_percentile_ordering(self):
        d1 = _make_dataset(2, 2)
        d2 = _make_dataset(3, 1)
        report = RatioComparer().compare([d1, d2])
        for r in report.ratios:
            assert r.ttft.p50 <= r.ttft.p95 <= r.ttft.p99
            assert r.tpot.p50 <= r.tpot.p95 <= r.tpot.p99
            assert r.total_latency.p50 <= r.total_latency.p95 <= r.total_latency.p99


class TestBestRatioDetection:
    """Test best ratio identification."""

    def test_lower_latency_wins(self):
        d1 = _make_dataset(3, 1, ttft_base=30.0, tpot_base=15.0, total_base=600.0)
        d2 = _make_dataset(1, 3, ttft_base=20.0, tpot_base=10.0, total_base=400.0)
        report = RatioComparer().compare([d1, d2])
        # d2 has lower everything, should win all
        for br in report.best_ratios:
            assert br.ratio_label == "1:3"

    def test_best_ratios_cover_all_combinations(self):
        d1 = _make_dataset(2, 2)
        d2 = _make_dataset(3, 1)
        report = RatioComparer().compare([d1, d2])
        # 3 metrics × 3 percentiles = 9
        assert len(report.best_ratios) == 9
        metrics = {(br.metric, br.percentile) for br in report.best_ratios}
        for m in ("ttft", "tpot", "total_latency"):
            for p in ("p50", "p95", "p99"):
                assert (m, p) in metrics

    def test_mixed_winners(self):
        # d1 better TTFT, d2 better TPOT
        d1 = _make_dataset(3, 1, ttft_base=10.0, tpot_base=20.0, total_base=500.0)
        d2 = _make_dataset(1, 3, ttft_base=20.0, tpot_base=10.0, total_base=500.0)
        report = RatioComparer().compare([d1, d2])
        ttft_winners = {br.ratio_label for br in report.best_ratios if br.metric == "ttft"}
        tpot_winners = {br.ratio_label for br in report.best_ratios if br.metric == "tpot"}
        assert "3:1" in ttft_winners
        assert "1:3" in tpot_winners


class TestOverallBest:
    """Test overall best ratio selection."""

    def test_clear_winner(self):
        d1 = _make_dataset(3, 1, ttft_base=30.0, tpot_base=30.0, total_base=600.0)
        d2 = _make_dataset(1, 3, ttft_base=10.0, tpot_base=10.0, total_base=200.0)
        report = RatioComparer().compare([d1, d2])
        assert report.overall_best == "1:3"

    def test_overall_best_is_string(self):
        d1 = _make_dataset(2, 2)
        d2 = _make_dataset(3, 1)
        report = RatioComparer().compare([d1, d2])
        assert isinstance(report.overall_best, str)
        assert ":" in report.overall_best


class TestProgrammaticAPI:
    """Test the compare_ratios() convenience function."""

    def test_basic(self):
        d1 = _make_dataset(2, 2)
        d2 = _make_dataset(3, 1)
        report = compare_ratios([d1, d2])
        assert isinstance(report, RatioComparison)
        assert len(report.ratios) == 2

    def test_raises_on_single(self):
        with pytest.raises(ValueError):
            compare_ratios([_make_dataset(2, 2)])


class TestReportSerialization:
    """Test that reports serialize correctly."""

    def test_model_dump(self):
        d1 = _make_dataset(2, 2)
        d2 = _make_dataset(3, 1)
        report = RatioComparer().compare([d1, d2])
        dumped = report.model_dump()
        assert isinstance(dumped, dict)
        assert len(dumped["ratios"]) == 2
        assert len(dumped["best_ratios"]) == 9
        assert isinstance(dumped["overall_best"], str)

    def test_model_dump_json(self):
        d1 = _make_dataset(2, 2)
        d2 = _make_dataset(3, 1)
        report = RatioComparer().compare([d1, d2])
        json_str = report.model_dump_json()
        assert isinstance(json_str, str)
        assert "ratio_label" in json_str


class TestPercentileStats:
    """Test PercentileStats model."""

    def test_valid(self):
        stats = PercentileStats(p50=10.0, p95=20.0, p99=30.0)
        assert stats.p50 == 10.0

    def test_serialization(self):
        stats = PercentileStats(p50=10.0, p95=20.0, p99=30.0)
        d = stats.model_dump()
        assert d == {"p50": 10.0, "p95": 20.0, "p99": 30.0}

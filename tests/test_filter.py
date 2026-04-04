"""Tests for benchmark filtering and slicing."""

from __future__ import annotations

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.filter import BenchmarkFilter, BenchmarkFilterResult, FilterConfig, filter_benchmark


def _make_requests(n: int = 20, base_ts: float = 1000.0) -> list[BenchmarkRequest]:
    """Generate n benchmark requests with varying properties."""
    requests = []
    for i in range(n):
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i:03d}",
                prompt_tokens=50 + i * 10,  # 50..240
                output_tokens=10 + i * 5,  # 10..105
                ttft_ms=10.0 + i * 2.0,  # 10..48
                tpot_ms=1.0 + i * 0.5,  # 1..10.5
                total_latency_ms=50.0 + i * 10.0,  # 50..240
                timestamp=base_ts + i * 1.0,  # 1000..1019
            )
        )
    return requests


def _make_data(n: int = 20) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=4,
            total_instances=6,
            measured_qps=100.0,
        ),
        requests=_make_requests(n),
    )


class TestFilterConfig:
    """Test FilterConfig validation."""

    def test_empty_config(self) -> None:
        config = FilterConfig()
        assert config.min_prompt_tokens is None
        assert config.sample_count is None

    def test_valid_range(self) -> None:
        config = FilterConfig(min_prompt_tokens=10, max_prompt_tokens=100)
        assert config.min_prompt_tokens == 10

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError, match="must be <="):
            FilterConfig(min_prompt_tokens=200, max_prompt_tokens=100)

    def test_invalid_latency_range(self) -> None:
        with pytest.raises(ValueError, match="must be <="):
            FilterConfig(min_ttft_ms=50.0, max_ttft_ms=10.0)

    def test_invalid_time_range(self) -> None:
        with pytest.raises(ValueError, match="must be <="):
            FilterConfig(time_start=2000.0, time_end=1000.0)

    def test_sample_count_and_fraction_exclusive(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both"):
            FilterConfig(sample_count=5, sample_fraction=0.5)

    def test_sample_count_only(self) -> None:
        config = FilterConfig(sample_count=10)
        assert config.sample_count == 10
        assert config.sample_fraction is None

    def test_sample_fraction_only(self) -> None:
        config = FilterConfig(sample_fraction=0.5)
        assert config.sample_fraction == 0.5
        assert config.sample_count is None


class TestBenchmarkFilter:
    """Test BenchmarkFilter.apply()."""

    def test_no_filters_returns_all(self) -> None:
        data = _make_data(20)
        config = FilterConfig()
        result = BenchmarkFilter(config).apply(data)
        assert result.filtered_count == 20
        assert result.removed_count == 0
        assert result.retention_rate == 1.0
        assert result.filters_applied == []

    def test_min_prompt_tokens(self) -> None:
        data = _make_data(20)
        config = FilterConfig(min_prompt_tokens=150)
        result = BenchmarkFilter(config).apply(data)
        # prompt_tokens: 50,60,...240 — >=150 means i>=10 → 10 requests
        assert result.filtered_count == 10
        for r in result.data.requests:
            assert r.prompt_tokens >= 150

    def test_max_prompt_tokens(self) -> None:
        data = _make_data(20)
        config = FilterConfig(max_prompt_tokens=100)
        result = BenchmarkFilter(config).apply(data)
        # prompt_tokens: 50,60,70,80,90,100 → 6 requests
        assert result.filtered_count == 6
        for r in result.data.requests:
            assert r.prompt_tokens <= 100

    def test_prompt_token_range(self) -> None:
        data = _make_data(20)
        config = FilterConfig(min_prompt_tokens=100, max_prompt_tokens=150)
        result = BenchmarkFilter(config).apply(data)
        # 100,110,120,130,140,150 → 6 requests
        assert result.filtered_count == 6

    def test_output_token_filter(self) -> None:
        data = _make_data(20)
        config = FilterConfig(min_output_tokens=50, max_output_tokens=80)
        result = BenchmarkFilter(config).apply(data)
        for r in result.data.requests:
            assert 50 <= r.output_tokens <= 80

    def test_ttft_filter(self) -> None:
        data = _make_data(20)
        config = FilterConfig(max_ttft_ms=20.0)
        result = BenchmarkFilter(config).apply(data)
        # ttft: 10,12,14,16,18,20 → 6 requests
        assert result.filtered_count == 6

    def test_tpot_filter(self) -> None:
        data = _make_data(20)
        config = FilterConfig(min_tpot_ms=5.0)
        result = BenchmarkFilter(config).apply(data)
        for r in result.data.requests:
            assert r.tpot_ms >= 5.0

    def test_total_latency_filter(self) -> None:
        data = _make_data(20)
        config = FilterConfig(min_total_latency_ms=100.0, max_total_latency_ms=200.0)
        result = BenchmarkFilter(config).apply(data)
        for r in result.data.requests:
            assert 100.0 <= r.total_latency_ms <= 200.0

    def test_time_window(self) -> None:
        data = _make_data(20)
        config = FilterConfig(time_start=1005.0, time_end=1010.0)
        result = BenchmarkFilter(config).apply(data)
        # timestamps 1005,1006,1007,1008,1009,1010 → 6 requests
        assert result.filtered_count == 6

    def test_sample_count(self) -> None:
        data = _make_data(20)
        config = FilterConfig(sample_count=5, seed=42)
        result = BenchmarkFilter(config).apply(data)
        assert result.filtered_count == 5

    def test_sample_count_larger_than_data(self) -> None:
        data = _make_data(5)
        config = FilterConfig(sample_count=100, seed=42)
        result = BenchmarkFilter(config).apply(data)
        assert result.filtered_count == 5

    def test_sample_fraction(self) -> None:
        data = _make_data(20)
        config = FilterConfig(sample_fraction=0.25, seed=42)
        result = BenchmarkFilter(config).apply(data)
        assert result.filtered_count == 5

    def test_sample_reproducible_with_seed(self) -> None:
        data = _make_data(20)
        config = FilterConfig(sample_count=5, seed=42)
        r1 = BenchmarkFilter(config).apply(data)
        r2 = BenchmarkFilter(config).apply(data)
        ids1 = [r.request_id for r in r1.data.requests]
        ids2 = [r.request_id for r in r2.data.requests]
        assert ids1 == ids2

    def test_combined_filters(self) -> None:
        data = _make_data(20)
        config = FilterConfig(min_prompt_tokens=100, max_ttft_ms=30.0)
        result = BenchmarkFilter(config).apply(data)
        for r in result.data.requests:
            assert r.prompt_tokens >= 100
            assert r.ttft_ms <= 30.0

    def test_qps_adjusted(self) -> None:
        data = _make_data(20)
        config = FilterConfig(min_prompt_tokens=150)
        result = BenchmarkFilter(config).apply(data)
        # 10/20 retained → QPS should be 50.0
        assert result.data.metadata.measured_qps == pytest.approx(50.0)

    def test_metadata_preserved(self) -> None:
        data = _make_data(20)
        config = FilterConfig(min_prompt_tokens=150)
        result = BenchmarkFilter(config).apply(data)
        assert result.data.metadata.num_prefill_instances == 2
        assert result.data.metadata.num_decode_instances == 4
        assert result.data.metadata.total_instances == 6

    def test_empty_result_raises(self) -> None:
        data = _make_data(20)
        config = FilterConfig(min_prompt_tokens=9999)
        with pytest.raises(ValueError, match="no data remaining"):
            BenchmarkFilter(config).apply(data)

    def test_filters_applied_list(self) -> None:
        data = _make_data(20)
        config = FilterConfig(min_prompt_tokens=100, sample_count=3, seed=1)
        result = BenchmarkFilter(config).apply(data)
        assert len(result.filters_applied) == 2
        assert "prompt_tokens>=100" in result.filters_applied[0]
        assert "sample_count=3" in result.filters_applied[1]

    def test_result_model(self) -> None:
        data = _make_data(20)
        config = FilterConfig(max_prompt_tokens=100)
        result = BenchmarkFilter(config).apply(data)
        assert isinstance(result, BenchmarkFilterResult)
        assert result.original_count == 20
        assert result.removed_count == result.original_count - result.filtered_count


class TestFilterBenchmarkAPI:
    """Test the programmatic filter_benchmark() API."""

    def test_api(self) -> None:
        data = _make_data(20)
        config = FilterConfig(min_prompt_tokens=100, max_prompt_tokens=200)
        result = filter_benchmark(data, config)
        assert isinstance(result, BenchmarkFilterResult)
        assert result.filtered_count > 0
        for r in result.data.requests:
            assert 100 <= r.prompt_tokens <= 200

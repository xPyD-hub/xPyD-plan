"""Tests for benchmark merge and aggregation."""

from __future__ import annotations

import json
import tempfile

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.merger import (
    BenchmarkMerger,
    MergeConfig,
    MergeStrategy,
    merge_benchmarks,
)


def _make_request(rid: str, ts: float = 1000.0) -> BenchmarkRequest:
    return BenchmarkRequest(
        request_id=rid,
        prompt_tokens=100,
        output_tokens=50,
        ttft_ms=20.0,
        tpot_ms=5.0,
        total_latency_ms=270.0,
        timestamp=ts,
    )


def _make_benchmark(
    request_ids: list[str],
    prefill: int = 2,
    decode: int = 2,
    qps: float = 10.0,
    base_ts: float = 1000.0,
) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=prefill,
            num_decode_instances=decode,
            total_instances=prefill + decode,
            measured_qps=qps,
        ),
        requests=[_make_request(rid, base_ts + i) for i, rid in enumerate(request_ids)],
    )


class TestBenchmarkMerger:
    """Tests for BenchmarkMerger class."""

    def test_merge_union_no_overlap(self):
        ds1 = _make_benchmark(["a1", "a2", "a3"], qps=10.0)
        ds2 = _make_benchmark(["b1", "b2", "b3"], qps=12.0, base_ts=2000.0)
        merger = BenchmarkMerger()
        result = merger.merge([ds1, ds2])

        assert result.source_count == 2
        assert result.total_requests_before == 6
        assert result.total_requests_after == 6
        assert result.duplicates_removed == 0
        assert result.strategy_used == MergeStrategy.UNION
        assert len(result.merged.requests) == 6

    def test_merge_union_with_duplicates(self):
        ds1 = _make_benchmark(["a1", "a2", "shared1"], qps=10.0)
        ds2 = _make_benchmark(["shared1", "b1", "b2"], qps=12.0, base_ts=2000.0)
        merger = BenchmarkMerger()
        result = merger.merge([ds1, ds2])

        assert result.total_requests_before == 6
        assert result.total_requests_after == 5
        assert result.duplicates_removed == 1
        # First occurrence wins — shared1 should have ds1 timestamp
        shared = [r for r in result.merged.requests if r.request_id == "shared1"]
        assert len(shared) == 1
        assert shared[0].timestamp == 1002.0  # 3rd request in ds1

    def test_merge_intersection(self):
        ds1 = _make_benchmark(["a1", "shared1", "shared2"], qps=10.0)
        ds2 = _make_benchmark(["shared1", "shared2", "b1"], qps=12.0, base_ts=2000.0)
        config = MergeConfig(strategy=MergeStrategy.INTERSECTION)
        merger = BenchmarkMerger(config)
        result = merger.merge([ds1, ds2])

        assert result.total_requests_after == 2
        ids = {r.request_id for r in result.merged.requests}
        assert ids == {"shared1", "shared2"}

    def test_merge_intersection_empty_raises(self):
        ds1 = _make_benchmark(["a1", "a2"], qps=10.0)
        ds2 = _make_benchmark(["b1", "b2"], qps=12.0, base_ts=2000.0)
        config = MergeConfig(strategy=MergeStrategy.INTERSECTION)
        merger = BenchmarkMerger(config)
        with pytest.raises(ValueError, match="empty dataset"):
            merger.merge([ds1, ds2])

    def test_merge_less_than_two_raises(self):
        ds1 = _make_benchmark(["a1", "a2"], qps=10.0)
        merger = BenchmarkMerger()
        with pytest.raises(ValueError, match="At least 2"):
            merger.merge([ds1])

    def test_merge_incompatible_config_raises(self):
        ds1 = _make_benchmark(["a1", "a2"], prefill=2, decode=2, qps=10.0)
        ds2 = _make_benchmark(["b1", "b2"], prefill=3, decode=1, qps=10.0, base_ts=2000.0)
        merger = BenchmarkMerger()
        with pytest.raises(ValueError, match="Incompatible cluster config"):
            merger.merge([ds1, ds2])

    def test_merge_incompatible_config_allowed(self):
        ds1 = _make_benchmark(["a1", "a2"], prefill=2, decode=2, qps=10.0)
        ds2 = _make_benchmark(["b1", "b2"], prefill=3, decode=1, qps=10.0, base_ts=2000.0)
        config = MergeConfig(require_same_config=False)
        merger = BenchmarkMerger(config)
        result = merger.merge([ds1, ds2])
        assert result.total_requests_after == 4

    def test_merge_aggregated_metadata(self):
        ds1 = _make_benchmark(["a1", "a2"], qps=10.0)
        ds2 = _make_benchmark(["b1", "b2"], qps=15.0, base_ts=2000.0)
        merger = BenchmarkMerger()
        result = merger.merge([ds1, ds2])

        assert result.merged.metadata.measured_qps == 25.0
        assert result.merged.metadata.num_prefill_instances == 2
        assert result.merged.metadata.num_decode_instances == 2

    def test_merge_three_datasets(self):
        ds1 = _make_benchmark(["a1", "a2"], qps=10.0)
        ds2 = _make_benchmark(["b1", "b2"], qps=12.0, base_ts=2000.0)
        ds3 = _make_benchmark(["c1", "c2"], qps=8.0, base_ts=3000.0)
        merger = BenchmarkMerger()
        result = merger.merge([ds1, ds2, ds3])

        assert result.source_count == 3
        assert result.total_requests_after == 6
        assert result.merged.metadata.measured_qps == 30.0

    def test_merge_sorted_by_timestamp(self):
        ds1 = _make_benchmark(["a1", "a2"], qps=10.0, base_ts=2000.0)
        ds2 = _make_benchmark(["b1", "b2"], qps=10.0, base_ts=1000.0)
        merger = BenchmarkMerger()
        result = merger.merge([ds1, ds2])

        timestamps = [r.timestamp for r in result.merged.requests]
        assert timestamps == sorted(timestamps)

    def test_merge_intersection_three_datasets(self):
        ds1 = _make_benchmark(["s1", "s2", "a1"], qps=10.0)
        ds2 = _make_benchmark(["s1", "s2", "b1"], qps=10.0, base_ts=2000.0)
        ds3 = _make_benchmark(["s1", "s2", "c1"], qps=10.0, base_ts=3000.0)
        config = MergeConfig(strategy=MergeStrategy.INTERSECTION)
        merger = BenchmarkMerger(config)
        result = merger.merge([ds1, ds2, ds3])

        assert result.total_requests_after == 2
        ids = {r.request_id for r in result.merged.requests}
        assert ids == {"s1", "s2"}

    def test_default_config(self):
        config = MergeConfig()
        assert config.strategy == MergeStrategy.UNION
        assert config.require_same_config is True


class TestMergeResult:
    """Tests for MergeResult model."""

    def test_model_serialization(self):
        ds1 = _make_benchmark(["a1", "a2"], qps=10.0)
        ds2 = _make_benchmark(["b1", "b2"], qps=10.0, base_ts=2000.0)
        merger = BenchmarkMerger()
        result = merger.merge([ds1, ds2])

        data = result.model_dump()
        assert data["source_count"] == 2
        assert data["strategy_used"] == "union"


class TestMergeBenchmarksAPI:
    """Tests for the programmatic merge_benchmarks() API."""

    def test_api_union(self):
        ds1 = _make_benchmark(["a1", "a2"], qps=10.0)
        ds2 = _make_benchmark(["b1", "b2"], qps=10.0, base_ts=2000.0)
        result = merge_benchmarks([ds1, ds2])

        assert result["source_count"] == 2
        assert result["total_requests_after"] == 4
        assert result["strategy_used"] == "union"

    def test_api_intersection(self):
        ds1 = _make_benchmark(["s1", "a1"], qps=10.0)
        ds2 = _make_benchmark(["s1", "b1"], qps=10.0, base_ts=2000.0)
        result = merge_benchmarks([ds1, ds2], strategy="intersection")

        assert result["total_requests_after"] == 1
        assert result["strategy_used"] == "intersection"

    def test_api_no_config_check(self):
        ds1 = _make_benchmark(["a1"], prefill=2, decode=2, qps=10.0)
        ds2 = _make_benchmark(["b1"], prefill=3, decode=1, qps=10.0, base_ts=2000.0)
        result = merge_benchmarks([ds1, ds2], require_same_config=False)
        assert result["total_requests_after"] == 2

    def test_api_invalid_strategy_raises(self):
        ds1 = _make_benchmark(["a1"], qps=10.0)
        ds2 = _make_benchmark(["b1"], qps=10.0, base_ts=2000.0)
        with pytest.raises(ValueError):
            merge_benchmarks([ds1, ds2], strategy="invalid")


class TestMergeCLI:
    """Tests for CLI merge subcommand integration."""

    def test_merge_subcommand_help(self):
        """Verify the merge subcommand is registered in CLI."""
        from xpyd_plan.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["merge", "--help"])
        assert exc_info.value.code == 0

    def test_merge_subcommand_runs(self):
        """End-to-end CLI merge test."""
        from xpyd_plan.cli import main

        ds1 = _make_benchmark(["a1", "a2", "a3"], qps=10.0)
        ds2 = _make_benchmark(["b1", "b2", "b3"], qps=12.0, base_ts=2000.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1:
            f1.write(ds1.model_dump_json())
            f1_path = f1.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            f2.write(ds2.model_dump_json())
            f2_path = f2.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
            out_path = out.name

        with pytest.raises(SystemExit) as exc_info:
            main(["merge", "--benchmark", f1_path, "--benchmark", f2_path, "--output", out_path])
        assert exc_info.value.code == 0

        with open(out_path) as f:
            merged = json.load(f)
        assert len(merged["requests"]) == 6

    def test_merge_json_output(self):
        """CLI merge with JSON output format."""
        from xpyd_plan.cli import main

        ds1 = _make_benchmark(["a1", "a2"], qps=10.0)
        ds2 = _make_benchmark(["b1", "b2"], qps=12.0, base_ts=2000.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1:
            f1.write(ds1.model_dump_json())
            f1_path = f1.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            f2.write(ds2.model_dump_json())
            f2_path = f2.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
            out_path = out.name

        with pytest.raises(SystemExit) as exc_info:
            main([
                "merge",
                "--benchmark", f1_path,
                "--benchmark", f2_path,
                "--output", out_path,
                "--output-format", "json",
            ])
        assert exc_info.value.code == 0

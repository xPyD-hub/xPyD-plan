"""Tests for the heatmap module."""

from __future__ import annotations

import json
from pathlib import Path

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.heatmap import (
    AggregationMetric,
    HeatmapConfig,
    HeatmapGenerator,
    LatencyField,
    _aggregate,
    _percentile,
    generate_heatmap,
)


def _make_benchmark(requests: list[dict]) -> BenchmarkData:
    """Create a BenchmarkData with the given request dicts."""
    reqs = [
        BenchmarkRequest(
            request_id=f"req-{i}",
            prompt_tokens=r["prompt_tokens"],
            output_tokens=r["output_tokens"],
            ttft_ms=r.get("ttft_ms", 10.0),
            tpot_ms=r.get("tpot_ms", 5.0),
            total_latency_ms=r.get("total_latency_ms", 100.0),
            timestamp=1000000.0 + i,
        )
        for i, r in enumerate(requests)
    ]
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=1,
            num_decode_instances=1,
            total_instances=2,
            measured_qps=10.0,
        ),
        requests=reqs,
    )


def _make_benchmark_file(requests: list[dict], tmp_path: Path) -> Path:
    """Write a benchmark JSON file and return its path."""
    data = _make_benchmark(requests)
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(data.model_dump()))
    return p


class TestPercentile:
    """Tests for the _percentile helper."""

    def test_single_value(self):
        assert _percentile([5.0], 50) == 5.0

    def test_p50(self):
        result = _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50)
        assert result == 3.0

    def test_p95_interpolates(self):
        values = list(range(1, 101))
        result = _percentile([float(v) for v in values], 95)
        assert 94.0 <= result <= 96.0

    def test_empty(self):
        assert _percentile([], 50) == 0.0


class TestAggregate:
    """Tests for the _aggregate helper."""

    def test_mean(self):
        assert _aggregate([2.0, 4.0, 6.0], AggregationMetric.MEAN) == 4.0

    def test_p50(self):
        result = _aggregate([1.0, 2.0, 3.0], AggregationMetric.P50)
        assert result == 2.0

    def test_p95(self):
        result = _aggregate([float(i) for i in range(100)], AggregationMetric.P95)
        assert result > 90

    def test_p99(self):
        result = _aggregate([float(i) for i in range(100)], AggregationMetric.P99)
        assert result > 95

    def test_empty(self):
        assert _aggregate([], AggregationMetric.MEAN) == 0.0


class TestHeatmapGenerator:
    """Tests for HeatmapGenerator."""

    def test_empty_data(self):
        data = BenchmarkData.model_construct(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=10.0,
            ),
            requests=[],
        )
        gen = HeatmapGenerator()
        report = gen.generate(data)
        assert report.total_requests == 0
        assert report.grid.cells == []
        assert report.recommendation == "No requests in benchmark data."

    def test_basic_generation(self):
        requests = [
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 100.0},
            {"prompt_tokens": 200, "output_tokens": 100, "total_latency_ms": 200.0},
            {"prompt_tokens": 300, "output_tokens": 150, "total_latency_ms": 300.0},
        ]
        data = _make_benchmark(requests)
        gen = HeatmapGenerator()
        config = HeatmapConfig(prompt_bins=3, output_bins=3)
        report = gen.generate(data, config)

        assert report.total_requests == 3
        assert report.grid.rows == 3
        assert report.grid.cols == 3
        assert len(report.grid.cells) == 9

    def test_hotspot_detection(self):
        requests = [
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 50.0},
            {"prompt_tokens": 200, "output_tokens": 100, "total_latency_ms": 500.0},
        ]
        data = _make_benchmark(requests)
        gen = HeatmapGenerator()
        config = HeatmapConfig(
            prompt_bins=2,
            output_bins=2,
            sla_threshold_ms=200.0,
        )
        report = gen.generate(data, config)

        assert report.hotspot_count >= 1
        hotspots = [c for c in report.grid.cells if c.is_hotspot]
        assert len(hotspots) >= 1
        assert any(c.value > 200.0 for c in hotspots)

    def test_no_hotspot_without_threshold(self):
        requests = [
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 999.0},
        ]
        data = _make_benchmark(requests)
        gen = HeatmapGenerator()
        config = HeatmapConfig(prompt_bins=1, output_bins=1)
        report = gen.generate(data, config)

        assert report.hotspot_count == 0

    def test_different_metrics(self):
        requests = [
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 100.0},
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 200.0},
        ]
        data = _make_benchmark(requests)
        gen = HeatmapGenerator()

        mean_config = HeatmapConfig(
            prompt_bins=1, output_bins=1, metric=AggregationMetric.MEAN,
        )
        p95_config = HeatmapConfig(
            prompt_bins=1, output_bins=1, metric=AggregationMetric.P95,
        )
        mean_report = gen.generate(data, mean_config)
        p95_report = gen.generate(data, p95_config)

        mean_cell = [c for c in mean_report.grid.cells if c.count > 0][0]
        p95_cell = [c for c in p95_report.grid.cells if c.count > 0][0]
        assert mean_cell.value == 150.0
        assert p95_cell.value >= 190.0

    def test_different_fields(self):
        requests = [
            {
                "prompt_tokens": 100, "output_tokens": 50,
                "ttft_ms": 10.0, "tpot_ms": 5.0, "total_latency_ms": 100.0,
            },
        ]
        data = _make_benchmark(requests)
        gen = HeatmapGenerator()

        ttft_report = gen.generate(
            data,
            HeatmapConfig(
                prompt_bins=1, output_bins=1,
                field=LatencyField.TTFT, metric=AggregationMetric.MEAN,
            ),
        )
        tpot_report = gen.generate(
            data,
            HeatmapConfig(
                prompt_bins=1, output_bins=1,
                field=LatencyField.TPOT, metric=AggregationMetric.MEAN,
            ),
        )

        ttft_cell = [c for c in ttft_report.grid.cells if c.count > 0][0]
        tpot_cell = [c for c in tpot_report.grid.cells if c.count > 0][0]
        assert ttft_cell.value == 10.0
        assert tpot_cell.value == 5.0

    def test_single_value_range(self):
        """All requests have same token counts."""
        requests = [
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 100.0},
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 200.0},
        ]
        data = _make_benchmark(requests)
        gen = HeatmapGenerator()
        config = HeatmapConfig(prompt_bins=2, output_bins=2, metric=AggregationMetric.MEAN)
        report = gen.generate(data, config)

        # Should still work with degenerate ranges
        assert report.total_requests == 2
        non_empty = [c for c in report.grid.cells if c.count > 0]
        assert len(non_empty) >= 1
        total_count = sum(c.count for c in report.grid.cells)
        assert total_count == 2

    def test_min_max_values(self):
        requests = [
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 50.0},
            {"prompt_tokens": 200, "output_tokens": 100, "total_latency_ms": 500.0},
        ]
        data = _make_benchmark(requests)
        gen = HeatmapGenerator()
        config = HeatmapConfig(prompt_bins=2, output_bins=2, metric=AggregationMetric.MEAN)
        report = gen.generate(data, config)

        assert report.min_value is not None
        assert report.max_value is not None
        assert report.min_value <= report.max_value

    def test_recommendation_no_hotspots(self):
        requests = [
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 50.0},
        ]
        data = _make_benchmark(requests)
        gen = HeatmapGenerator()
        config = HeatmapConfig(prompt_bins=1, output_bins=1, sla_threshold_ms=1000.0)
        report = gen.generate(data, config)

        assert "No hotspots" in report.recommendation

    def test_recommendation_with_hotspots(self):
        requests = [
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 500.0},
        ]
        data = _make_benchmark(requests)
        gen = HeatmapGenerator()
        config = HeatmapConfig(prompt_bins=1, output_bins=1, sla_threshold_ms=100.0)
        report = gen.generate(data, config)

        assert "hotspot" in report.recommendation.lower()

    def test_many_requests_all_bins_covered(self):
        """With enough spread requests, most bins should have data."""
        requests = [
            {
                "prompt_tokens": 50 + i * 10,
                "output_tokens": 20 + i * 5,
                "total_latency_ms": 50.0 + i * 10,
            }
            for i in range(100)
        ]
        data = _make_benchmark(requests)
        gen = HeatmapGenerator()
        config = HeatmapConfig(prompt_bins=5, output_bins=5)
        report = gen.generate(data, config)

        assert report.total_requests == 100
        non_empty = [c for c in report.grid.cells if c.count > 0]
        assert len(non_empty) >= 5  # at least some bins filled

    def test_default_config(self):
        requests = [
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 100.0},
        ]
        data = _make_benchmark(requests)
        gen = HeatmapGenerator()
        report = gen.generate(data)  # no config = defaults

        assert report.config.prompt_bins == 10
        assert report.config.output_bins == 10
        assert report.config.metric == AggregationMetric.P95
        assert report.config.field == LatencyField.TOTAL

    def test_model_serialization(self):
        requests = [
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 100.0},
        ]
        data = _make_benchmark(requests)
        gen = HeatmapGenerator()
        report = gen.generate(data, HeatmapConfig(prompt_bins=2, output_bins=2))

        d = report.model_dump()
        assert "config" in d
        assert "grid" in d
        assert "cells" in d["grid"]
        assert "total_requests" in d


class TestProgrammaticAPI:
    """Tests for generate_heatmap() function."""

    def test_basic(self, tmp_path):
        requests = [
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 100.0},
            {"prompt_tokens": 200, "output_tokens": 100, "total_latency_ms": 200.0},
        ]
        bench_file = _make_benchmark_file(requests, tmp_path)
        result = generate_heatmap(str(bench_file), prompt_bins=2, output_bins=2)

        assert isinstance(result, dict)
        assert result["total_requests"] == 2
        assert "grid" in result

    def test_with_sla_threshold(self, tmp_path):
        requests = [
            {"prompt_tokens": 100, "output_tokens": 50, "total_latency_ms": 500.0},
        ]
        bench_file = _make_benchmark_file(requests, tmp_path)
        result = generate_heatmap(
            str(bench_file),
            prompt_bins=1,
            output_bins=1,
            sla_threshold_ms=100.0,
        )

        assert result["hotspot_count"] >= 1

    def test_custom_field(self, tmp_path):
        requests = [
            {
                "prompt_tokens": 100, "output_tokens": 50,
                "ttft_ms": 25.0, "tpot_ms": 5.0, "total_latency_ms": 100.0,
            },
        ]
        bench_file = _make_benchmark_file(requests, tmp_path)
        result = generate_heatmap(
            str(bench_file),
            prompt_bins=1,
            output_bins=1,
            field="ttft_ms",
            metric="mean",
        )

        cells = result["grid"]["cells"]
        non_empty = [c for c in cells if c["count"] > 0]
        assert len(non_empty) == 1
        assert non_empty[0]["value"] == 25.0

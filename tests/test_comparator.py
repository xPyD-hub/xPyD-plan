"""Tests for benchmark comparison and regression detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.comparator import (
    BenchmarkComparator,
    ComparisonResult,
    MetricDelta,
    compare_benchmarks,
)


def _make_requests(
    n: int = 50,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    total_base: float = 200.0,
    spread: float = 0.2,
) -> list[BenchmarkRequest]:
    """Generate benchmark requests with controlled latency distribution."""
    requests = []
    for i in range(n):
        factor = 1.0 + spread * (i / max(n - 1, 1))
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i:04d}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=ttft_base * factor,
                tpot_ms=tpot_base * factor,
                total_latency_ms=total_base * factor,
                timestamp=1700000000.0 + i,
            )
        )
    return requests


def _make_benchmark(
    qps: float = 10.0,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    total_base: float = 200.0,
    n: int = 50,
    spread: float = 0.2,
) -> BenchmarkData:
    """Create a BenchmarkData object."""
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=qps,
        ),
        requests=_make_requests(n, ttft_base, tpot_base, total_base, spread),
    )


class TestBenchmarkComparator:
    """Tests for BenchmarkComparator."""

    def test_no_regression_identical(self):
        """Identical benchmarks should show no regression."""
        data = _make_benchmark()
        comparator = BenchmarkComparator(threshold=0.1)
        result = comparator.compare(data, data)
        assert not result.has_regression
        assert result.regression_count == 0
        for delta in result.latency_deltas:
            assert delta.absolute_delta == 0.0
            assert delta.relative_delta == 0.0

    def test_regression_detected_latency(self):
        """Higher latency beyond threshold should be flagged."""
        baseline = _make_benchmark(ttft_base=50.0)
        current = _make_benchmark(ttft_base=60.0)  # 20% worse
        comparator = BenchmarkComparator(threshold=0.1)
        result = comparator.compare(baseline, current)
        assert result.has_regression
        ttft_deltas = [d for d in result.latency_deltas if d.metric.startswith("ttft")]
        assert any(d.is_regression for d in ttft_deltas)

    def test_regression_detected_qps_drop(self):
        """QPS drop beyond threshold should be flagged."""
        baseline = _make_benchmark(qps=100.0)
        current = _make_benchmark(qps=80.0)  # 20% drop
        comparator = BenchmarkComparator(threshold=0.1)
        result = comparator.compare(baseline, current)
        assert result.has_regression
        assert result.qps_delta.is_regression

    def test_no_regression_qps_increase(self):
        """QPS increase should not be a regression."""
        baseline = _make_benchmark(qps=100.0)
        current = _make_benchmark(qps=120.0)  # 20% better
        comparator = BenchmarkComparator(threshold=0.1)
        result = comparator.compare(baseline, current)
        assert not result.qps_delta.is_regression

    def test_improvement_detected(self):
        """Lower latency should show negative delta, not regression."""
        baseline = _make_benchmark(ttft_base=100.0)
        current = _make_benchmark(ttft_base=80.0)  # 20% better
        comparator = BenchmarkComparator(threshold=0.1)
        result = comparator.compare(baseline, current)
        ttft_deltas = [d for d in result.latency_deltas if d.metric.startswith("ttft")]
        for d in ttft_deltas:
            assert d.relative_delta < 0
            assert not d.is_regression

    def test_threshold_boundary_not_regression(self):
        """Change exactly at threshold should not be flagged (need > threshold)."""
        baseline = _make_benchmark(ttft_base=100.0)
        # 10% worse = exactly at threshold
        current = _make_benchmark(ttft_base=110.0)
        comparator = BenchmarkComparator(threshold=0.1)
        result = comparator.compare(baseline, current)
        # P50 base values: 100*1.0 vs 110*1.0 → 10% exactly, not > 10%
        ttft_p50 = [d for d in result.latency_deltas if d.metric == "ttft_p50_ms"][0]
        assert not ttft_p50.is_regression

    def test_threshold_zero(self):
        """Threshold=0 should flag any degradation."""
        baseline = _make_benchmark(ttft_base=100.0)
        current = _make_benchmark(ttft_base=101.0)  # 1% worse
        comparator = BenchmarkComparator(threshold=0.0)
        result = comparator.compare(baseline, current)
        assert result.has_regression

    def test_invalid_threshold(self):
        """Negative threshold should raise."""
        with pytest.raises(ValueError, match="threshold must be >= 0"):
            BenchmarkComparator(threshold=-0.1)

    def test_regression_count(self):
        """Count should reflect total regressions across all metrics."""
        baseline = _make_benchmark(ttft_base=50.0, tpot_base=10.0, total_base=200.0, qps=100.0)
        current = _make_benchmark(ttft_base=70.0, tpot_base=14.0, total_base=280.0, qps=70.0)
        comparator = BenchmarkComparator(threshold=0.1)
        result = comparator.compare(baseline, current)
        # All 9 latency metrics + QPS = 10 potential regressions
        assert result.regression_count == 10

    def test_comparison_result_fields(self):
        """ComparisonResult should have all expected fields."""
        data = _make_benchmark(qps=50.0)
        comparator = BenchmarkComparator()
        result = comparator.compare(data, data)
        assert result.baseline_qps == 50.0
        assert result.current_qps == 50.0
        assert result.threshold == 0.1
        assert len(result.latency_deltas) == 9  # 3 metrics × 3 percentiles
        assert isinstance(result.qps_delta, MetricDelta)

    def test_metric_delta_fields(self):
        """MetricDelta should compute correct values."""
        baseline = _make_benchmark(ttft_base=100.0)
        current = _make_benchmark(ttft_base=150.0)
        comparator = BenchmarkComparator(threshold=0.1)
        result = comparator.compare(baseline, current)
        ttft_p50 = [d for d in result.latency_deltas if d.metric == "ttft_p50_ms"][0]
        assert ttft_p50.baseline > 0
        assert ttft_p50.current > ttft_p50.baseline
        assert ttft_p50.absolute_delta > 0
        assert ttft_p50.relative_delta > 0
        assert ttft_p50.is_regression

    def test_large_threshold_no_regression(self):
        """Very large threshold should not flag anything."""
        baseline = _make_benchmark(ttft_base=50.0)
        current = _make_benchmark(ttft_base=90.0)  # 80% worse
        comparator = BenchmarkComparator(threshold=1.0)  # 100% threshold
        result = comparator.compare(baseline, current)
        assert not result.has_regression

    def test_mixed_regression(self):
        """Some metrics regressed, others improved."""
        baseline = _make_benchmark(ttft_base=100.0, tpot_base=20.0, qps=50.0)
        current = _make_benchmark(ttft_base=120.0, tpot_base=15.0, qps=55.0)
        comparator = BenchmarkComparator(threshold=0.1)
        result = comparator.compare(baseline, current)
        # TTFT regressed, TPOT improved, QPS improved
        ttft_regressions = [
            d for d in result.latency_deltas if d.metric.startswith("ttft") and d.is_regression
        ]
        tpot_regressions = [
            d for d in result.latency_deltas if d.metric.startswith("tpot") and d.is_regression
        ]
        assert len(ttft_regressions) > 0
        assert len(tpot_regressions) == 0
        assert not result.qps_delta.is_regression

    def test_default_threshold(self):
        """Default threshold should be 0.1."""
        comparator = BenchmarkComparator()
        assert comparator.threshold == 0.1

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        comparator = BenchmarkComparator(threshold=0.05)
        assert comparator.threshold == 0.05


class TestCompareBenchmarks:
    """Tests for the compare_benchmarks() programmatic API."""

    def test_compare_from_files(self, tmp_path: Path):
        """Compare two benchmark files."""
        baseline = _make_benchmark(qps=100.0, ttft_base=50.0)
        current = _make_benchmark(qps=100.0, ttft_base=55.0)

        baseline_path = tmp_path / "baseline.json"
        current_path = tmp_path / "current.json"
        baseline_path.write_text(baseline.model_dump_json(indent=2))
        current_path.write_text(current.model_dump_json(indent=2))

        result = compare_benchmarks(baseline_path, current_path, threshold=0.1)
        assert isinstance(result, ComparisonResult)
        assert result.baseline_qps == 100.0

    def test_compare_with_regression(self, tmp_path: Path):
        """File comparison should detect regressions."""
        baseline = _make_benchmark(qps=100.0, ttft_base=50.0)
        current = _make_benchmark(qps=100.0, ttft_base=80.0)

        baseline_path = tmp_path / "baseline.json"
        current_path = tmp_path / "current.json"
        baseline_path.write_text(baseline.model_dump_json(indent=2))
        current_path.write_text(current.model_dump_json(indent=2))

        result = compare_benchmarks(baseline_path, current_path)
        assert result.has_regression

    def test_compare_invalid_file(self, tmp_path: Path):
        """Invalid JSON should raise."""
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text("not json")
        current_path = tmp_path / "current.json"
        current_path.write_text("{}")
        with pytest.raises(Exception):
            compare_benchmarks(baseline_path, current_path)

    def test_compare_default_threshold(self, tmp_path: Path):
        """Default threshold should be 0.1."""
        data = _make_benchmark()
        p = tmp_path / "data.json"
        p.write_text(data.model_dump_json(indent=2))
        result = compare_benchmarks(p, p)
        assert result.threshold == 0.1

    def test_compare_custom_threshold(self, tmp_path: Path):
        """Custom threshold should be passed through."""
        data = _make_benchmark()
        p = tmp_path / "data.json"
        p.write_text(data.model_dump_json(indent=2))
        result = compare_benchmarks(p, p, threshold=0.05)
        assert result.threshold == 0.05


class TestCompareCLI:
    """Tests for the compare CLI subcommand."""

    def test_compare_table_output(self, tmp_path: Path, capsys):
        """CLI compare should produce table output."""
        from xpyd_plan.cli import main

        baseline = _make_benchmark(qps=100.0, ttft_base=50.0)
        current = _make_benchmark(qps=100.0, ttft_base=60.0)

        (tmp_path / "baseline.json").write_text(baseline.model_dump_json(indent=2))
        (tmp_path / "current.json").write_text(current.model_dump_json(indent=2))

        main([
            "compare",
            "--baseline", str(tmp_path / "baseline.json"),
            "--current", str(tmp_path / "current.json"),
        ])

    def test_compare_json_output(self, tmp_path: Path, capsys):
        """CLI compare with --output-format json."""
        from xpyd_plan.cli import main

        data = _make_benchmark()
        p = tmp_path / "data.json"
        p.write_text(data.model_dump_json(indent=2))

        main(["compare", "--baseline", str(p), "--current", str(p), "--output-format", "json"])

    def test_compare_custom_threshold_cli(self, tmp_path: Path):
        """CLI compare with custom threshold."""
        from xpyd_plan.cli import main

        data = _make_benchmark()
        p = tmp_path / "data.json"
        p.write_text(data.model_dump_json(indent=2))

        main([
            "compare",
            "--baseline", str(p),
            "--current", str(p),
            "--threshold", "0.05",
        ])

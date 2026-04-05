"""Tests for benchmark coverage analysis."""

from __future__ import annotations

import json
import tempfile

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.coverage import (
    CoverageAnalyzer,
    CoverageGrade,
    CoverageReport,
    analyze_coverage,
)


def _make_dataset(num_prefill: int, num_decode: int, n_requests: int = 10) -> BenchmarkData:
    """Create a minimal benchmark dataset for testing."""
    requests = [
        BenchmarkRequest(
            request_id=f"req-{i}",
            prompt_tokens=100 + i,
            output_tokens=50 + i,
            ttft_ms=20.0 + i,
            tpot_ms=10.0 + i,
            total_latency_ms=100.0 + i * 5,
            timestamp=1000.0 + i,
        )
        for i in range(n_requests)
    ]
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=10.0,
        ),
        requests=requests,
    )


class TestCoverageAnalyzer:
    """Tests for CoverageAnalyzer."""

    def test_single_benchmark(self) -> None:
        """Single benchmark should have low coverage."""
        datasets = [_make_dataset(3, 3)]
        analyzer = CoverageAnalyzer(datasets)
        report = analyzer.analyze()

        assert isinstance(report, CoverageReport)
        assert report.total_instances == 6
        assert report.metrics.benchmarked_ratios == 1
        assert report.metrics.total_possible_ratios == 5  # 1P:5D through 5P:1D
        assert report.metrics.coverage_fraction == pytest.approx(0.2)
        assert len(report.gaps) == 4

    def test_full_coverage(self) -> None:
        """All ratios benchmarked should yield high score."""
        datasets = [_make_dataset(p, 4 - p) for p in range(1, 4)]
        analyzer = CoverageAnalyzer(datasets)
        report = analyzer.analyze()

        assert report.metrics.benchmarked_ratios == 3
        assert report.metrics.coverage_fraction == 1.0
        assert report.metrics.has_boundaries is True
        assert report.metrics.has_balanced is True
        assert report.metrics.max_gap_size == 0
        assert report.score >= 90.0
        assert report.grade == CoverageGrade.EXCELLENT
        assert len(report.gaps) == 0

    def test_boundary_ratios(self) -> None:
        """Benchmarking extremes should set has_boundaries."""
        datasets = [_make_dataset(1, 7), _make_dataset(7, 1)]
        analyzer = CoverageAnalyzer(datasets)
        report = analyzer.analyze()

        assert report.metrics.has_boundaries is True
        assert report.metrics.has_balanced is False

    def test_balanced_ratio(self) -> None:
        """Benchmarking near-midpoint should set has_balanced."""
        datasets = [_make_dataset(4, 4)]
        analyzer = CoverageAnalyzer(datasets)
        report = analyzer.analyze()

        assert report.metrics.has_balanced is True
        assert report.metrics.has_boundaries is False

    def test_balanced_detection_odd_total(self) -> None:
        """Balanced detection works for odd total instances."""
        datasets = [_make_dataset(3, 4)]  # total=7, mid=3.5
        analyzer = CoverageAnalyzer(datasets)
        report = analyzer.analyze()
        assert report.metrics.has_balanced is True

    def test_gap_detection(self) -> None:
        """Gaps should be detected between benchmarked ratios."""
        datasets = [_make_dataset(1, 9), _make_dataset(9, 1)]
        analyzer = CoverageAnalyzer(datasets)
        report = analyzer.analyze()

        assert report.metrics.max_gap_size == 7  # gap from P=1 to P=9
        critical_gaps = [g for g in report.gaps if g.priority == "critical"]
        assert len(critical_gaps) > 0

    def test_gap_distance(self) -> None:
        """Gap distance_to_nearest should be correct."""
        datasets = [_make_dataset(1, 5), _make_dataset(5, 1)]
        analyzer = CoverageAnalyzer(datasets)
        report = analyzer.analyze()

        # P=3 is distance 2 from both P=1 and P=5
        gap_3 = next(g for g in report.gaps if g.num_prefill == 3)
        assert gap_3.distance_to_nearest == 2

    def test_spread_score_even(self) -> None:
        """Evenly spread benchmarks should have high spread score."""
        # Benchmark P=1,3,5,7,9 — evenly spaced
        datasets = [_make_dataset(p, 10 - p) for p in [1, 3, 5, 7, 9]]
        analyzer = CoverageAnalyzer(datasets)
        report = analyzer.analyze()

        assert report.metrics.spread_score > 0.5

    def test_spread_score_clustered(self) -> None:
        """Clustered benchmarks should have low spread score."""
        # Benchmark P=4,5,6 — clustered in middle
        datasets = [_make_dataset(p, 10 - p) for p in [4, 5, 6]]
        analyzer = CoverageAnalyzer(datasets)
        report = analyzer.analyze()

        assert report.metrics.spread_score < 0.6

    def test_duplicate_benchmarks_deduplicated(self) -> None:
        """Duplicate P:D ratios should be counted once."""
        datasets = [_make_dataset(3, 3), _make_dataset(3, 3)]
        analyzer = CoverageAnalyzer(datasets)
        report = analyzer.analyze()

        assert report.metrics.benchmarked_ratios == 1

    def test_mismatched_total_instances_raises(self) -> None:
        """Datasets with different total instances should raise."""
        datasets = [_make_dataset(2, 2), _make_dataset(3, 3)]
        with pytest.raises(ValueError, match="same total instances"):
            CoverageAnalyzer(datasets)

    def test_empty_datasets_raises(self) -> None:
        """Empty dataset list should raise."""
        with pytest.raises(ValueError, match="At least one"):
            CoverageAnalyzer([])

    def test_recommendations_missing_boundaries(self) -> None:
        """Should recommend boundary benchmarks when missing."""
        datasets = [_make_dataset(3, 3)]  # total=6, no boundaries
        report = CoverageAnalyzer(datasets).analyze()

        boundary_rec = [r for r in report.recommendations if "boundary" in r.lower()]
        assert len(boundary_rec) > 0

    def test_recommendations_missing_balanced(self) -> None:
        """Should recommend balanced ratio when missing."""
        datasets = [_make_dataset(1, 9)]  # total=10, no balanced
        report = CoverageAnalyzer(datasets).analyze()

        balanced_rec = [r for r in report.recommendations if "balanced" in r.lower()]
        assert len(balanced_rec) > 0

    def test_score_range(self) -> None:
        """Score should be between 0 and 100."""
        datasets = [_make_dataset(1, 5)]
        report = CoverageAnalyzer(datasets).analyze()
        assert 0.0 <= report.score <= 100.0

    def test_grade_assignment(self) -> None:
        """Grade should correspond to score ranges."""
        # Full coverage
        datasets = [_make_dataset(p, 4 - p) for p in range(1, 4)]
        report = CoverageAnalyzer(datasets).analyze()
        assert report.grade in (CoverageGrade.EXCELLENT, CoverageGrade.GOOD)

    def test_gaps_sorted_by_distance(self) -> None:
        """Gaps should be sorted by distance descending."""
        datasets = [_make_dataset(1, 9), _make_dataset(9, 1)]
        report = CoverageAnalyzer(datasets).analyze()

        distances = [g.distance_to_nearest for g in report.gaps]
        assert distances == sorted(distances, reverse=True)

    def test_good_coverage_no_critical_recs(self) -> None:
        """Full coverage should produce benign recommendations."""
        datasets = [_make_dataset(p, 4 - p) for p in range(1, 4)]
        report = CoverageAnalyzer(datasets).analyze()

        # Should not recommend boundaries or balanced (already have them)
        assert not any("boundary" in r.lower() for r in report.recommendations)
        assert not any("balanced" in r.lower() for r in report.recommendations)


class TestAnalyzeCoverageAPI:
    """Tests for the programmatic API."""

    def test_returns_dict(self) -> None:
        """analyze_coverage should return a dict."""
        datasets = [_make_dataset(2, 2)]
        result = analyze_coverage(datasets)
        assert isinstance(result, dict)
        assert "score" in result
        assert "grade" in result
        assert "metrics" in result
        assert "gaps" in result

    def test_dict_values(self) -> None:
        """Dict values should be correct types."""
        datasets = [_make_dataset(p, 4 - p) for p in range(1, 4)]
        result = analyze_coverage(datasets)
        assert result["metrics"]["coverage_fraction"] == 1.0
        assert result["metrics"]["benchmarked_ratios"] == 3


class TestCoverageCLI:
    """Test CLI integration for coverage subcommand."""

    def test_cli_json_output(self) -> None:
        """CLI should produce valid JSON output."""
        from unittest.mock import MagicMock

        from xpyd_plan.cli._coverage import _cmd_coverage

        data = _make_dataset(3, 3)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data.model_dump(), f, default=str)
            f.flush()

            args = MagicMock()
            args.benchmark = [f.name]
            args.output_format = "json"

            import io
            import sys

            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                _cmd_coverage(args)
            finally:
                sys.stdout = old_stdout

            output = json.loads(captured.getvalue())
            assert "score" in output
            assert "grade" in output

    def test_cli_table_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """CLI table output should not raise."""
        from unittest.mock import MagicMock

        from xpyd_plan.cli._coverage import _cmd_coverage

        data = _make_dataset(3, 3)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data.model_dump(), f, default=str)
            f.flush()

            args = MagicMock()
            args.benchmark = [f.name]
            args.output_format = "table"
            _cmd_coverage(args)

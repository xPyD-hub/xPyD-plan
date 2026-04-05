"""Tests for benchmark diff report (M85)."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.diff_report import (
    ChangeDirection,
    DiffReport,
    DiffReporter,
    DiffSummary,
    generate_diff_report,
)


def _make_requests(
    n: int = 100,
    ttft_base: float = 50.0,
    tpot_base: float = 20.0,
    total_base: float = 200.0,
    prompt_tokens: int = 100,
    output_tokens: int = 50,
    timestamp_start: float = 1000.0,
) -> list[BenchmarkRequest]:
    """Generate N benchmark requests with slight variation."""
    import random

    rng = random.Random(42)
    reqs = []
    for i in range(n):
        reqs.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=prompt_tokens + rng.randint(-10, 10),
                output_tokens=output_tokens + rng.randint(-5, 5),
                ttft_ms=ttft_base + rng.uniform(-5, 5),
                tpot_ms=tpot_base + rng.uniform(-2, 2),
                total_latency_ms=total_base + rng.uniform(-20, 20),
                timestamp=timestamp_start + i * 0.1,
            )
        )
    return reqs


def _make_benchmark(
    requests: list[BenchmarkRequest],
    qps: float = 10.0,
    prefill: int = 2,
    decode: int = 6,
) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=prefill,
            num_decode_instances=decode,
            total_instances=prefill + decode,
            measured_qps=qps,
        ),
        requests=requests,
    )


@pytest.fixture
def baseline() -> BenchmarkData:
    reqs = _make_requests(
        n=100, ttft_base=50, tpot_base=20, total_base=200
    )
    return _make_benchmark(reqs, qps=10.0)


@pytest.fixture
def improved_target() -> BenchmarkData:
    return _make_benchmark(
        _make_requests(
            n=100, ttft_base=40, tpot_base=15, total_base=160
        ),
        qps=12.0,
    )


@pytest.fixture
def regressed_target() -> BenchmarkData:
    return _make_benchmark(
        _make_requests(
            n=100, ttft_base=70, tpot_base=30, total_base=300
        ),
        qps=8.0,
    )


@pytest.fixture
def equivalent_target() -> BenchmarkData:
    return _make_benchmark(
        _make_requests(
            n=100, ttft_base=50.5, tpot_base=20.2, total_base=201
        ),
        qps=10.1,
    )


class TestDiffReporterInit:
    def test_default_threshold(self):
        reporter = DiffReporter()
        assert reporter._threshold == 5.0

    def test_custom_threshold(self):
        reporter = DiffReporter(regression_threshold_pct=10.0)
        assert reporter._threshold == 10.0

    def test_negative_threshold_raises(self):
        with pytest.raises(
            ValueError, match="regression_threshold_pct must be >= 0"
        ):
            DiffReporter(regression_threshold_pct=-1.0)

    def test_zero_threshold(self):
        reporter = DiffReporter(regression_threshold_pct=0.0)
        assert reporter._threshold == 0.0


class TestDiffReporterCompare:
    def test_report_structure(self, baseline):
        reporter = DiffReporter()
        report = reporter.compare(baseline, baseline, "b", "t")
        assert isinstance(report, DiffReport)
        assert report.baseline_file == "b"
        assert report.target_file == "t"
        assert len(report.latency_diffs) == 3
        assert len(report.token_diffs) == 2
        assert isinstance(report.summary, DiffSummary)

    def test_improved_verdict(self, baseline, improved_target):
        reporter = DiffReporter()
        report = reporter.compare(baseline, improved_target)
        assert report.summary.verdict == "BETTER"
        assert report.summary.regressions == 0
        assert report.summary.improvements > 0

    def test_regressed_verdict(self, baseline, regressed_target):
        reporter = DiffReporter()
        report = reporter.compare(baseline, regressed_target)
        assert report.summary.verdict == "WORSE"
        assert report.summary.improvements == 0
        assert report.summary.regressions > 0

    def test_equivalent_verdict(self, baseline, equivalent_target):
        reporter = DiffReporter()
        report = reporter.compare(baseline, equivalent_target)
        assert report.summary.verdict == "EQUIVALENT"

    def test_mixed_verdict(self, baseline):
        mixed_reqs = _make_requests(
            n=100, ttft_base=60, tpot_base=25, total_base=250
        )
        mixed = _make_benchmark(mixed_reqs, qps=15.0)
        reporter = DiffReporter()
        report = reporter.compare(baseline, mixed)
        assert report.summary.verdict == "MIXED"

    def test_qps_higher_is_better(self, baseline):
        better_qps = _make_benchmark(
            _make_requests(
                n=100, ttft_base=50, tpot_base=20, total_base=200
            ),
            qps=20.0,
        )
        reporter = DiffReporter()
        report = reporter.compare(baseline, better_qps)
        assert report.qps_diff.direction == ChangeDirection.IMPROVED

    def test_qps_lower_is_regression(self, baseline):
        worse_qps = _make_benchmark(
            _make_requests(
                n=100, ttft_base=50, tpot_base=20, total_base=200
            ),
            qps=5.0,
        )
        reporter = DiffReporter()
        report = reporter.compare(baseline, worse_qps)
        assert report.qps_diff.direction == ChangeDirection.REGRESSED

    def test_latency_lower_is_better(self, baseline, improved_target):
        reporter = DiffReporter()
        report = reporter.compare(baseline, improved_target)
        for section in report.latency_diffs:
            assert section.p50.direction == ChangeDirection.IMPROVED
            assert section.p95.direction == ChangeDirection.IMPROVED

    def test_request_counts(self, baseline, improved_target):
        reporter = DiffReporter()
        report = reporter.compare(baseline, improved_target)
        assert report.baseline_requests == 100
        assert report.target_requests == 100


class TestMetricDiff:
    def test_relative_delta(self, baseline, improved_target):
        reporter = DiffReporter()
        report = reporter.compare(baseline, improved_target)
        qd = report.qps_diff
        assert qd.relative_delta_pct is not None
        assert qd.absolute_delta == pytest.approx(2.0)

    def test_zero_baseline_relative_is_none(self):
        reporter = DiffReporter()
        diff = reporter._make_diff(
            "test", 0.0, 10.0, higher_is_better=True
        )
        assert diff.relative_delta_pct is None
        assert diff.direction == ChangeDirection.UNCHANGED


class TestToMarkdown:
    def test_markdown_contains_sections(
        self, baseline, improved_target
    ):
        reporter = DiffReporter()
        report = reporter.compare(
            baseline, improved_target, "a.json", "b.json"
        )
        md = reporter.to_markdown(report)
        assert "# Benchmark Diff:" in md
        assert "## Overview" in md
        assert "## Latency Comparison" in md
        assert "## Token Distribution" in md
        assert "BETTER" in md

    def test_markdown_has_tables(self, baseline, improved_target):
        reporter = DiffReporter()
        report = reporter.compare(baseline, improved_target)
        md = reporter.to_markdown(report)
        assert "|" in md
        assert "P50" in md
        assert "P95" in md
        assert "P99" in md


class TestProgrammaticAPI:
    def test_returns_dict(self, baseline, improved_target):
        result = generate_diff_report(baseline, improved_target)
        assert isinstance(result, dict)
        assert "summary" in result
        assert result["summary"]["verdict"] == "BETTER"

    def test_custom_names(self, baseline, improved_target):
        result = generate_diff_report(
            baseline,
            improved_target,
            baseline_name="run1",
            target_name="run2",
        )
        assert result["baseline_file"] == "run1"
        assert result["target_file"] == "run2"

    def test_custom_threshold(self, baseline, equivalent_target):
        result = generate_diff_report(
            baseline,
            equivalent_target,
            regression_threshold_pct=0.01,
        )
        assert result["summary"]["verdict"] != "EQUIVALENT"


class TestThresholdSensitivity:
    def test_strict_detects_more(self, baseline, equivalent_target):
        strict = DiffReporter(regression_threshold_pct=0.01)
        lenient = DiffReporter(regression_threshold_pct=20.0)
        r_strict = strict.compare(baseline, equivalent_target)
        r_lenient = lenient.compare(baseline, equivalent_target)
        assert (
            r_strict.summary.unchanged
            <= r_lenient.summary.unchanged
        )

    def test_zero_threshold(self, baseline):
        same = _make_benchmark(
            _make_requests(
                n=100, ttft_base=50, tpot_base=20, total_base=200
            ),
            qps=10.0,
        )
        reporter = DiffReporter(regression_threshold_pct=0.0)
        report = reporter.compare(baseline, same)
        assert report.summary.verdict == "EQUIVALENT"


class TestEdgeCases:
    def test_single_request(self):
        reqs = [
            BenchmarkRequest(
                request_id="r1",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=50.0,
                tpot_ms=20.0,
                total_latency_ms=200.0,
                timestamp=1000.0,
            )
        ]
        b = _make_benchmark(reqs, qps=1.0)
        t = _make_benchmark(reqs, qps=1.0)
        reporter = DiffReporter()
        report = reporter.compare(b, t)
        assert report.summary.verdict == "EQUIVALENT"

    def test_different_request_counts(self, baseline):
        small = _make_benchmark(
            _make_requests(
                n=10, ttft_base=50, tpot_base=20, total_base=200
            ),
            qps=10.0,
        )
        reporter = DiffReporter()
        report = reporter.compare(baseline, small)
        assert report.baseline_requests == 100
        assert report.target_requests == 10

    def test_report_serializable(self, baseline, improved_target):
        reporter = DiffReporter()
        report = reporter.compare(baseline, improved_target)
        data = report.model_dump()
        json_str = json.dumps(data, default=str)
        assert json_str

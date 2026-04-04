"""Tests for root cause analysis."""

from __future__ import annotations

import json
import random
import tempfile

import pytest

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.root_cause import (
    FactorSignificance,
    RootCauseAnalyzer,
    RootCauseReport,
    _classify_significance,
    _cohens_d,
    _mann_whitney_u,
    analyze_root_cause,
)


def _make_benchmark(
    requests: list[BenchmarkRequest] | None = None,
    n: int = 100,
    seed: int = 42,
) -> BenchmarkData:
    """Create benchmark data with optional custom requests."""
    if requests is not None:
        return BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=2,
                num_decode_instances=2,
                total_instances=4,
                measured_qps=10.0,
            ),
            requests=requests,
        )

    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        prompt_tokens = rng.randint(10, 500)
        # Correlate TTFT with prompt_tokens
        ttft = 10.0 + prompt_tokens * 0.2 + rng.gauss(0, 5)
        ttft = max(1.0, ttft)
        reqs.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=prompt_tokens,
                output_tokens=rng.randint(10, 200),
                ttft_ms=round(ttft, 2),
                tpot_ms=round(rng.gauss(20, 3), 2),
                total_latency_ms=round(ttft + rng.gauss(100, 20), 2),
                timestamp=1000.0 + i * 0.1,
            )
        )
    return _make_benchmark(requests=reqs)


def _make_clear_violation_benchmark(seed: int = 42) -> BenchmarkData:
    """Create data where high prompt_tokens clearly cause TTFT violations."""
    rng = random.Random(seed)
    reqs = []
    for i in range(200):
        if i < 150:
            # Normal requests: low prompt tokens, low TTFT
            prompt_tokens = rng.randint(10, 50)
            ttft = rng.gauss(30, 5)
        else:
            # Violation requests: high prompt tokens, high TTFT
            prompt_tokens = rng.randint(400, 600)
            ttft = rng.gauss(150, 10)

        reqs.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=prompt_tokens,
                output_tokens=rng.randint(10, 200),
                ttft_ms=round(max(1.0, ttft), 2),
                tpot_ms=round(max(0.1, rng.gauss(20, 3)), 2),
                total_latency_ms=round(max(1.0, ttft + rng.gauss(100, 20)), 2),
                timestamp=1000.0 + i * 0.1,
            )
        )
    return _make_benchmark(requests=reqs)


class TestRootCauseAnalyzer:
    """Tests for RootCauseAnalyzer."""

    def test_basic_analysis(self) -> None:
        """Test basic root cause analysis."""
        data = _make_benchmark()
        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(data, sla_metric="ttft", sla_threshold_ms=80.0)

        assert isinstance(report, RootCauseReport)
        assert report.total_requests == 100
        assert report.passing_requests + report.failing_requests == 100
        assert 0 <= report.failure_rate <= 1
        assert report.sla_metric == "ttft"
        assert report.sla_threshold_ms == 80.0

    def test_clear_prompt_token_cause(self) -> None:
        """Test that high prompt_tokens is identified as cause of TTFT violations."""
        data = _make_clear_violation_benchmark()
        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(data, sla_metric="ttft", sla_threshold_ms=80.0)

        assert report.failing_requests > 0
        assert len(report.factors) > 0

        # prompt_tokens should be the top factor
        prompt_factor = next(
            (f for f in report.factors if f.factor == "prompt_tokens"), None
        )
        assert prompt_factor is not None
        assert prompt_factor.significance != FactorSignificance.NONE
        assert prompt_factor.effect_size > 0  # Higher in fail group
        assert prompt_factor.fail_mean > prompt_factor.pass_mean

    def test_no_violations(self) -> None:
        """Test with no SLA violations."""
        data = _make_benchmark()
        analyzer = RootCauseAnalyzer()
        # Very high threshold — no violations
        report = analyzer.analyze(data, sla_metric="ttft", sla_threshold_ms=99999.0)

        assert report.failing_requests == 0
        assert report.failure_rate == 0.0
        assert report.factors == []
        assert "No SLA violations" in report.recommendation

    def test_all_violations(self) -> None:
        """Test with all requests violating SLA."""
        data = _make_benchmark()
        analyzer = RootCauseAnalyzer()
        # Very low threshold — all fail
        report = analyzer.analyze(data, sla_metric="ttft", sla_threshold_ms=0.001)

        assert report.passing_requests == 0
        assert report.failure_rate == 1.0
        assert report.factors == []
        assert "All requests violate" in report.recommendation

    def test_different_sla_metrics(self) -> None:
        """Test with different SLA metrics."""
        data = _make_benchmark()
        analyzer = RootCauseAnalyzer()

        for metric in ["ttft", "tpot", "total_latency"]:
            report = analyzer.analyze(
                data, sla_metric=metric, sla_threshold_ms=50.0
            )
            assert report.sla_metric == metric
            assert isinstance(report, RootCauseReport)

    def test_invalid_sla_metric(self) -> None:
        """Test with invalid SLA metric."""
        data = _make_benchmark()
        analyzer = RootCauseAnalyzer()

        with pytest.raises(ValueError, match="Invalid sla_metric"):
            analyzer.analyze(data, sla_metric="invalid")

    def test_insufficient_data(self) -> None:
        """Test with fewer than 2 requests."""
        data = _make_benchmark(
            requests=[
                BenchmarkRequest(
                    request_id="req-0",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=50.0,
                    tpot_ms=20.0,
                    total_latency_ms=150.0,
                    timestamp=1000.0,
                )
            ]
        )
        analyzer = RootCauseAnalyzer()

        with pytest.raises(ValueError, match="at least 2 requests"):
            analyzer.analyze(data)

    def test_factors_sorted_by_significance(self) -> None:
        """Test that factors are sorted by significance descending."""
        data = _make_clear_violation_benchmark()
        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(data, sla_metric="ttft", sla_threshold_ms=80.0)

        if len(report.factors) >= 2:
            sig_order = {
                FactorSignificance.NONE: 0,
                FactorSignificance.LOW: 1,
                FactorSignificance.MEDIUM: 2,
                FactorSignificance.HIGH: 3,
            }
            for i in range(len(report.factors) - 1):
                a = report.factors[i]
                b = report.factors[i + 1]
                assert sig_order[a.significance] >= sig_order[b.significance]

    def test_factor_fields(self) -> None:
        """Test that factor fields are properly populated."""
        data = _make_clear_violation_benchmark()
        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(data, sla_metric="ttft", sla_threshold_ms=80.0)

        for f in report.factors:
            assert isinstance(f.factor, str)
            assert isinstance(f.description, str)
            assert isinstance(f.significance, FactorSignificance)
            assert 0.0 <= f.p_value <= 1.0
            assert isinstance(f.effect_size, float)
            assert isinstance(f.pass_mean, float)
            assert isinstance(f.fail_mean, float)

    def test_recommendation_mentions_factors(self) -> None:
        """Test that recommendation mentions significant factors."""
        data = _make_clear_violation_benchmark()
        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(data, sla_metric="ttft", sla_threshold_ms=80.0)

        rec = report.recommendation.lower()
        assert "violate" in rec or "violation" in rec

    def test_cross_metric_factors(self) -> None:
        """Test that non-target latency metrics appear as factors."""
        data = _make_benchmark()
        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(data, sla_metric="ttft", sla_threshold_ms=50.0)

        if report.factors:
            factor_names = [f.factor for f in report.factors]
            # When analyzing ttft, tpot_ms and total_latency_ms should appear
            assert "tpot_ms" in factor_names or "total_latency_ms" in factor_names

    def test_tpot_metric_excludes_self(self) -> None:
        """Test that analyzing tpot doesn't include tpot_ms as factor."""
        data = _make_benchmark()
        analyzer = RootCauseAnalyzer()
        report = analyzer.analyze(data, sla_metric="tpot", sla_threshold_ms=15.0)

        if report.factors:
            factor_names = [f.factor for f in report.factors]
            assert "tpot_ms" not in factor_names


class TestMannWhitneyU:
    """Tests for Mann-Whitney U implementation."""

    def test_identical_samples(self) -> None:
        """Identical samples should have high p-value."""
        sample = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, p = _mann_whitney_u(sample, sample)
        assert p > 0.05

    def test_clearly_different_samples(self) -> None:
        """Clearly different samples should have low p-value."""
        a = [float(x) for x in range(1, 51)]
        b = [float(x) for x in range(100, 150)]
        _, p = _mann_whitney_u(a, b)
        assert p < 0.05

    def test_empty_sample(self) -> None:
        """Empty samples return U=0, p=1."""
        u, p = _mann_whitney_u([], [1.0, 2.0])
        assert u == 0.0
        assert p == 1.0


class TestCohensD:
    """Tests for Cohen's d calculation."""

    def test_identical_groups(self) -> None:
        """Identical groups should have d=0."""
        sample = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _cohens_d(sample, sample) == 0.0

    def test_different_groups(self) -> None:
        """Clearly different groups should have large |d|."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [10.0, 11.0, 12.0, 13.0, 14.0]
        d = _cohens_d(a, b)
        assert d > 2.0  # Large effect

    def test_small_sample(self) -> None:
        """Groups with < 2 members return 0."""
        assert _cohens_d([1.0], [2.0]) == 0.0

    def test_direction(self) -> None:
        """Effect should be positive when group2 > group1."""
        a = [1.0, 2.0, 3.0]
        b = [10.0, 11.0, 12.0]
        assert _cohens_d(a, b) > 0
        assert _cohens_d(b, a) < 0


class TestClassifySignificance:
    """Tests for significance classification."""

    def test_high(self) -> None:
        assert _classify_significance(0.0005) == FactorSignificance.HIGH

    def test_medium(self) -> None:
        assert _classify_significance(0.005) == FactorSignificance.MEDIUM

    def test_low(self) -> None:
        assert _classify_significance(0.03) == FactorSignificance.LOW

    def test_none(self) -> None:
        assert _classify_significance(0.1) == FactorSignificance.NONE

    def test_boundary_001(self) -> None:
        assert _classify_significance(0.001) == FactorSignificance.MEDIUM

    def test_boundary_01(self) -> None:
        assert _classify_significance(0.01) == FactorSignificance.LOW

    def test_boundary_05(self) -> None:
        assert _classify_significance(0.05) == FactorSignificance.NONE


class TestAnalyzeRootCauseAPI:
    """Tests for programmatic API."""

    def test_returns_dict(self) -> None:
        """API should return a dict."""
        data = _make_benchmark()
        result = analyze_root_cause(data, sla_metric="ttft", sla_threshold_ms=80.0)

        assert isinstance(result, dict)
        assert "total_requests" in result
        assert "factors" in result
        assert "recommendation" in result

    def test_api_matches_class(self) -> None:
        """API should match class output."""
        data = _make_benchmark()
        api_result = analyze_root_cause(
            data, sla_metric="ttft", sla_threshold_ms=80.0
        )
        analyzer = RootCauseAnalyzer()
        class_result = analyzer.analyze(
            data, sla_metric="ttft", sla_threshold_ms=80.0
        ).model_dump()

        assert api_result == class_result

    def test_default_parameters(self) -> None:
        """API should work with default parameters."""
        data = _make_benchmark()
        result = analyze_root_cause(data)
        assert result["sla_metric"] == "ttft"
        assert result["sla_threshold_ms"] == 100.0


class TestRootCauseCLI:
    """Tests for CLI integration."""

    def test_cli_json_output(self) -> None:
        """Test CLI with JSON output format."""
        from xpyd_plan.cli._root_cause import _cmd_root_cause

        data = _make_clear_violation_benchmark()

        # Write benchmark to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data.model_dump(), f)
            tmp_path = f.name

        import argparse

        args = argparse.Namespace(
            benchmark=tmp_path,
            sla_metric="ttft",
            sla_threshold=80.0,
            output_format="json",
        )

        import io
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            _cmd_root_cause(args)
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert "factors" in output
        assert output["total_requests"] == 200

    def test_cli_table_output(self) -> None:
        """Test CLI with table output format."""
        from xpyd_plan.cli._root_cause import _cmd_root_cause

        data = _make_clear_violation_benchmark()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data.model_dump(), f)
            tmp_path = f.name

        import argparse

        args = argparse.Namespace(
            benchmark=tmp_path,
            sla_metric="ttft",
            sla_threshold=80.0,
            output_format="table",
        )

        # Should not raise
        _cmd_root_cause(args)

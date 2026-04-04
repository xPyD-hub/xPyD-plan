"""Tests for fairness analysis."""

from __future__ import annotations

import json
import subprocess
import tempfile

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.fairness import (
    FairnessAnalyzer,
    FairnessClassification,
    FairnessReport,
    _jain_index,
    analyze_fairness,
)


def _make_data(
    requests: list[dict] | None = None,
    n: int = 100,
) -> BenchmarkData:
    """Build a BenchmarkData with custom or generated requests."""
    if requests is not None:
        reqs = [BenchmarkRequest(**r) for r in requests]
    else:
        # Generate requests with varying prompt_tokens and latency correlated to tokens
        reqs = []
        for i in range(n):
            pt = 50 + i * 10  # 50..1040
            reqs.append(
                BenchmarkRequest(
                    request_id=f"r{i}",
                    prompt_tokens=pt,
                    output_tokens=50,
                    ttft_ms=10.0 + pt * 0.05,
                    tpot_ms=5.0 + pt * 0.01,
                    total_latency_ms=100.0 + pt * 0.1,
                    timestamp=1000.0 + i,
                )
            )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=reqs,
    )


def _make_uniform_data(n: int = 100) -> BenchmarkData:
    """All requests have same latency — perfectly fair."""
    reqs = [
        BenchmarkRequest(
            request_id=f"r{i}",
            prompt_tokens=50 + i * 10,
            output_tokens=50,
            ttft_ms=10.0,
            tpot_ms=5.0,
            total_latency_ms=100.0,
            timestamp=1000.0 + i,
        )
        for i in range(n)
    ]
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=reqs,
    )


def _make_unfair_data(n: int = 100) -> BenchmarkData:
    """Long prompts have drastically worse latency."""
    reqs = []
    for i in range(n):
        pt = 50 + i * 10
        factor = 1.0 if pt < 500 else 10.0  # step change
        reqs.append(
            BenchmarkRequest(
                request_id=f"r{i}",
                prompt_tokens=pt,
                output_tokens=50,
                ttft_ms=10.0 * factor,
                tpot_ms=5.0 * factor,
                total_latency_ms=100.0 * factor,
                timestamp=1000.0 + i,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=reqs,
    )


class TestJainIndex:
    def test_equal_values(self):
        assert _jain_index([5.0, 5.0, 5.0]) == 1.0

    def test_single_value(self):
        assert _jain_index([42.0]) == 1.0

    def test_empty(self):
        assert _jain_index([]) == 1.0

    def test_all_zero(self):
        assert _jain_index([0.0, 0.0]) == 1.0

    def test_two_values(self):
        # J = (1+2)^2 / (2*(1+4)) = 9/10 = 0.9
        assert abs(_jain_index([1.0, 2.0]) - 0.9) < 1e-6

    def test_very_unfair(self):
        # [1, 100] -> J = 101^2 / (2*10001) = 10201/20002 ≈ 0.51
        j = _jain_index([1.0, 100.0])
        assert j < 0.7


class TestFairnessAnalyzer:
    def test_basic_analysis(self):
        data = _make_data()
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(data, num_buckets=4)
        assert isinstance(report, FairnessReport)
        assert len(report.fairness_indices) == 3
        assert all(0 <= fi.jain_index <= 1 for fi in report.fairness_indices)

    def test_uniform_is_fair(self):
        data = _make_uniform_data()
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(data, num_buckets=4)
        assert report.overall_classification == FairnessClassification.FAIR

    def test_unfair_data(self):
        data = _make_unfair_data()
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(data, num_buckets=4)
        assert report.overall_classification in (
            FairnessClassification.MODERATE,
            FairnessClassification.UNFAIR,
        )

    def test_bucket_count(self):
        data = _make_data()
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(data, num_buckets=5)
        # May have fewer buckets if quantiles collapse
        assert report.num_buckets >= 1

    def test_total_requests_match(self):
        data = _make_data(n=80)
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(data, num_buckets=4)
        total = sum(b.request_count for b in report.buckets)
        assert total == 80

    def test_too_few_requests(self):
        data = _make_data(n=2)
        analyzer = FairnessAnalyzer()
        with pytest.raises(ValueError, match="Need at least"):
            analyzer.analyze(data, num_buckets=4)

    def test_num_buckets_too_small(self):
        data = _make_data()
        analyzer = FairnessAnalyzer()
        with pytest.raises(ValueError, match="num_buckets must be at least 2"):
            analyzer.analyze(data, num_buckets=1)

    def test_same_prompt_tokens(self):
        """All requests have same prompt_tokens — should still work."""
        reqs = [
            dict(
                request_id=f"r{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=10.0 + i,
                tpot_ms=5.0,
                total_latency_ms=100.0 + i,
                timestamp=1000.0 + i,
            )
            for i in range(20)
        ]
        data = _make_data(requests=reqs)
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(data, num_buckets=4)
        # Should collapse to 1 bucket
        assert report.num_buckets == 1

    def test_recommendation_text(self):
        data = _make_uniform_data()
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(data, num_buckets=4)
        assert "No action needed" in report.recommendation

    def test_two_buckets(self):
        data = _make_data()
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(data, num_buckets=2)
        assert report.num_buckets >= 1


class TestAnalyzeFairnessAPI:
    def test_returns_dict(self):
        data = _make_data()
        result = analyze_fairness(data, num_buckets=4)
        assert isinstance(result, dict)
        assert "fairness_indices" in result
        assert "buckets" in result
        assert "overall_classification" in result

    def test_default_buckets(self):
        data = _make_data()
        result = analyze_fairness(data)
        assert isinstance(result, dict)


class TestFairnessCLI:
    def test_json_output(self):
        data = _make_data()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json())
            f.flush()

            result = subprocess.run(
                [
                    "xpyd-plan", "fairness",
                    "--benchmark", f.name, "--output-format", "json",
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            output = json.loads(result.stdout)
            assert "fairness_indices" in output

    def test_table_output(self):
        data = _make_data()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json())
            f.flush()

            result = subprocess.run(
                ["xpyd-plan", "fairness", "--benchmark", f.name],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert "Fairness" in result.stdout or "fairness" in result.stdout.lower()

    def test_custom_buckets(self):
        data = _make_data()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json())
            f.flush()

            result = subprocess.run(
                [
                    "xpyd-plan", "fairness",
                    "--benchmark", f.name,
                    "--buckets", "3", "--output-format", "json",
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0


class TestFairnessClassifications:
    def test_fair_threshold(self):
        assert FairnessClassification.FAIR.value == "fair"

    def test_moderate_threshold(self):
        assert FairnessClassification.MODERATE.value == "moderate"

    def test_unfair_threshold(self):
        assert FairnessClassification.UNFAIR.value == "unfair"


class TestEdgeCases:
    def test_minimum_requests(self):
        """Exactly num_buckets requests should work."""
        reqs = [
            dict(
                request_id=f"r{i}",
                prompt_tokens=50 + i * 100,
                output_tokens=50,
                ttft_ms=10.0,
                tpot_ms=5.0,
                total_latency_ms=100.0,
                timestamp=1000.0 + i,
            )
            for i in range(4)
        ]
        data = _make_data(requests=reqs)
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(data, num_buckets=4)
        assert report.num_buckets >= 1

    def test_large_bucket_count(self):
        data = _make_data(n=100)
        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(data, num_buckets=10)
        assert report.num_buckets >= 1

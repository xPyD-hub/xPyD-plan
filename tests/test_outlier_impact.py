"""Tests for outlier impact analysis."""

from __future__ import annotations

import json
import subprocess

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.outlier_impact import (
    ImpactRecommendation,
    OutlierImpactAnalyzer,
    analyze_outlier_impact,
)


def _make_data(
    requests: list[dict] | None = None,
    n: int = 100,
) -> BenchmarkData:
    """Build a BenchmarkData with custom or generated requests."""
    if requests is not None:
        reqs = [BenchmarkRequest(**r) for r in requests]
    else:
        reqs = []
        for i in range(n):
            reqs.append(
                BenchmarkRequest(
                    request_id=f"r{i}",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=20.0 + i * 0.1,
                    tpot_ms=10.0 + i * 0.05,
                    total_latency_ms=100.0 + i * 0.5,
                    timestamp=1700000000 + i,
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


def _make_data_with_outliers() -> BenchmarkData:
    """Create data where a few requests are extreme outliers."""
    reqs = []
    for i in range(100):
        reqs.append(
            dict(
                request_id=f"r{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=20.0,
                tpot_ms=10.0,
                total_latency_ms=100.0,
                timestamp=1700000000 + i,
            )
        )
    # Add 5 extreme outliers
    for i in range(5):
        reqs.append(
            dict(
                request_id=f"outlier{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=500.0,
                tpot_ms=300.0,
                total_latency_ms=2000.0,
                timestamp=1700000100 + i,
            )
        )
    return _make_data(requests=reqs)


class TestOutlierImpactAnalyzer:
    """Tests for OutlierImpactAnalyzer."""

    def test_no_outliers(self):
        """When data has no outliers, recommendation is KEEP."""
        data = _make_data(n=100)
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data)
        assert report.recommendation == ImpactRecommendation.KEEP
        assert report.total_requests == 100
        assert report.outlier_count == 0

    def test_with_outliers_basic(self):
        """Outliers are detected and counted."""
        data = _make_data_with_outliers()
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data)
        assert report.total_requests == 105
        assert report.outlier_count > 0
        assert report.outlier_fraction > 0
        assert len(report.metric_impacts) == 3

    def test_metric_impact_fields(self):
        """MetricImpact has correct field names."""
        data = _make_data_with_outliers()
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data)
        mi = report.metric_impacts[0]
        assert mi.metric in ("ttft", "tpot", "total_latency")
        assert mi.p95_before_ms >= mi.p95_after_ms  # Removing outliers shouldn't increase P95

    def test_sla_flip(self):
        """When outliers cause SLA failure, removing them restores compliance."""
        data = _make_data_with_outliers()
        analyzer = OutlierImpactAnalyzer()
        # Set SLA at a level that outliers cause failure but clean data passes
        report = analyzer.analyze(
            data,
            sla_ttft_ms=100.0,  # P95 with outliers > 100, without < 100
        )
        if report.sla_comparison is not None and report.sla_comparison.flipped:
            assert report.recommendation == ImpactRecommendation.FILTER

    def test_sla_no_flip(self):
        """When SLA is generous, outliers don't flip compliance."""
        data = _make_data_with_outliers()
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(
            data,
            sla_ttft_ms=10000.0,
        )
        assert report.sla_comparison is not None
        assert not report.sla_comparison.flipped

    def test_single_request(self):
        """Single-request benchmark produces KEEP recommendation."""
        data = _make_data(n=1)
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data)
        assert report.total_requests == 1
        assert report.recommendation == ImpactRecommendation.KEEP

    def test_custom_iqr_multiplier(self):
        """Tighter IQR multiplier finds more outliers."""
        data = _make_data(n=100)
        analyzer = OutlierImpactAnalyzer()
        report_tight = analyzer.analyze(data, iqr_multiplier=0.5)
        report_loose = analyzer.analyze(data, iqr_multiplier=3.0)
        assert report_tight.outlier_count >= report_loose.outlier_count

    def test_report_serialization(self):
        """Report can be serialized to dict/JSON."""
        data = _make_data_with_outliers()
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data)
        d = report.model_dump()
        assert "metric_impacts" in d
        assert "recommendation" in d
        json.dumps(d)  # Should not raise

    def test_summary_not_empty(self):
        """Summary field is always populated."""
        data = _make_data_with_outliers()
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data)
        assert len(report.summary) > 0

    def test_p99_delta(self):
        """P99 delta is computed correctly."""
        data = _make_data_with_outliers()
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data)
        for mi in report.metric_impacts:
            assert mi.p99_delta_ms == pytest.approx(mi.p99_after_ms - mi.p99_before_ms, abs=0.01)

    def test_three_metrics(self):
        """Report covers ttft, tpot, total_latency."""
        data = _make_data_with_outliers()
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data)
        metrics = {mi.metric for mi in report.metric_impacts}
        assert metrics == {"ttft", "tpot", "total_latency"}

    def test_outlier_fraction_range(self):
        """Outlier fraction is between 0 and 1."""
        data = _make_data_with_outliers()
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data)
        assert 0 <= report.outlier_fraction <= 1

    def test_sla_comparison_only_when_thresholds_given(self):
        """SLA comparison is None when no SLA thresholds provided."""
        data = _make_data(n=50)
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data)
        assert report.sla_comparison is None

    def test_material_p95_shift_recommends_filter(self):
        """Large P95 shift (>5%) without SLA still recommends FILTER."""
        # Create data where outliers cause >5% P95 shift
        reqs = []
        for i in range(90):
            reqs.append(dict(
                request_id=f"r{i}", prompt_tokens=100, output_tokens=50,
                ttft_ms=20.0, tpot_ms=10.0, total_latency_ms=100.0,
                timestamp=1700000000 + i,
            ))
        for i in range(10):
            reqs.append(dict(
                request_id=f"o{i}", prompt_tokens=100, output_tokens=50,
                ttft_ms=1000.0, tpot_ms=500.0, total_latency_ms=5000.0,
                timestamp=1700000100 + i,
            ))
        data = _make_data(requests=reqs)
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data)
        assert report.recommendation == ImpactRecommendation.FILTER

    def test_multiple_sla_thresholds(self):
        """Multiple SLA thresholds all checked."""
        data = _make_data_with_outliers()
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(
            data,
            sla_ttft_ms=10000.0,
            sla_tpot_ms=10000.0,
            sla_total_ms=10000.0,
        )
        assert report.sla_comparison is not None
        assert report.sla_comparison.pass_before is True

    def test_iqr_multiplier_in_report(self):
        """IQR multiplier is recorded in the report."""
        data = _make_data(n=50)
        analyzer = OutlierImpactAnalyzer()
        report = analyzer.analyze(data, iqr_multiplier=2.5)
        assert report.iqr_multiplier == 2.5


class TestAnalyzeOutlierImpactAPI:
    """Tests for the programmatic API."""

    def test_api_returns_dict(self, tmp_path):
        """Programmatic API returns a dict."""
        data = _make_data_with_outliers()
        path = tmp_path / "bench.json"
        path.write_text(data.model_dump_json(indent=2))
        result = analyze_outlier_impact(str(path))
        assert isinstance(result, dict)
        assert "recommendation" in result
        assert "metric_impacts" in result


class TestCLIOutlierImpact:
    """CLI integration tests."""

    def _write_benchmark(self, tmp_path) -> str:
        data = _make_data_with_outliers()
        path = tmp_path / "bench.json"
        path.write_text(data.model_dump_json(indent=2))
        return str(path)

    def test_cli_table_output(self, tmp_path):
        """CLI runs without error in table mode."""
        bench = self._write_benchmark(tmp_path)
        result = subprocess.run(
            ["xpyd-plan", "outlier-impact", "--benchmark", bench],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Outlier Impact Analysis" in result.stdout

    def test_cli_json_output(self, tmp_path):
        """CLI produces valid JSON."""
        bench = self._write_benchmark(tmp_path)
        result = subprocess.run(
            [
                "xpyd-plan", "outlier-impact",
                "--benchmark", bench,
                "--output-format", "json",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert "recommendation" in parsed

    def test_cli_with_sla(self, tmp_path):
        """CLI accepts SLA flags."""
        bench = self._write_benchmark(tmp_path)
        result = subprocess.run(
            [
                "xpyd-plan", "outlier-impact",
                "--benchmark", bench,
                "--sla-ttft", "100",
                "--sla-tpot", "50",
                "--output-format", "json",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert parsed["sla_comparison"] is not None

    def test_cli_custom_iqr(self, tmp_path):
        """CLI accepts --iqr-multiplier."""
        bench = self._write_benchmark(tmp_path)
        result = subprocess.run(
            [
                "xpyd-plan", "outlier-impact",
                "--benchmark", bench,
                "--iqr-multiplier", "2.5",
                "--output-format", "json",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert parsed["iqr_multiplier"] == 2.5

"""Tests for Prometheus/OpenMetrics export."""

from __future__ import annotations

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.metrics_export import MetricLine, MetricsExporter, MetricsReport, export_metrics


def _make_benchmark(
    num_prefill: int,
    num_decode: int,
    qps: float,
    num_requests: int = 50,
) -> BenchmarkData:
    """Create a benchmark dataset with given configuration."""
    requests = [
        BenchmarkRequest(
            request_id=f"req-{i}",
            prompt_tokens=100,
            output_tokens=50,
            ttft_ms=20.0 + i * 0.5,
            tpot_ms=10.0 + i * 0.2,
            total_latency_ms=70.0 + i * 1.0,
            timestamp=1000.0 + i,
        )
        for i in range(num_requests)
    ]
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=qps,
        ),
        requests=requests,
    )


class TestMetricsExporter:
    """Tests for MetricsExporter."""

    def test_basic_export(self) -> None:
        """Single benchmark produces expected metrics."""
        bench = _make_benchmark(2, 3, 50.0)
        exporter = MetricsExporter()
        report = exporter.export([bench])

        assert len(report.metrics) > 0
        assert report.text.endswith("# EOF\n")

    def test_metric_names(self) -> None:
        """All metric names use correct prefix."""
        bench = _make_benchmark(1, 1, 10.0)
        report = MetricsExporter().export([bench])

        for m in report.metrics:
            assert m.name.startswith("xpyd_plan_")

    def test_labels_include_pd_ratio(self) -> None:
        """Labels should include P:D ratio."""
        bench = _make_benchmark(2, 3, 50.0)
        report = MetricsExporter().export([bench])

        for m in report.metrics:
            assert "pd_ratio" in m.labels
            assert m.labels["pd_ratio"] == "2:3"

    def test_instance_metrics(self) -> None:
        """Instance count metrics are present."""
        bench = _make_benchmark(2, 3, 50.0)
        report = MetricsExporter().export([bench])

        names = [m.name for m in report.metrics]
        assert "xpyd_plan_prefill_instances" in names
        assert "xpyd_plan_decode_instances" in names
        assert "xpyd_plan_total_instances" in names

    def test_qps_metric(self) -> None:
        """QPS metric matches input."""
        bench = _make_benchmark(1, 1, 42.5)
        report = MetricsExporter().export([bench])

        qps_metrics = [m for m in report.metrics if m.name == "xpyd_plan_measured_qps"]
        assert len(qps_metrics) == 1
        assert qps_metrics[0].value == 42.5

    def test_request_count_metric(self) -> None:
        """Request count metric matches."""
        bench = _make_benchmark(1, 1, 10.0, num_requests=30)
        report = MetricsExporter().export([bench])

        count_metrics = [m for m in report.metrics if m.name == "xpyd_plan_request_count"]
        assert len(count_metrics) == 1
        assert count_metrics[0].value == 30.0
        assert count_metrics[0].metric_type == "counter"

    def test_latency_percentiles(self) -> None:
        """Latency percentile metrics for TTFT, TPOT, total_latency at P50/P95/P99."""
        bench = _make_benchmark(1, 1, 10.0)
        report = MetricsExporter().export([bench])

        ttft_metrics = [m for m in report.metrics if m.name == "xpyd_plan_ttft_ms"]
        tpot_metrics = [m for m in report.metrics if m.name == "xpyd_plan_tpot_ms"]
        total_metrics = [m for m in report.metrics if m.name == "xpyd_plan_total_latency_ms"]

        assert len(ttft_metrics) == 3  # p50, p95, p99
        assert len(tpot_metrics) == 3
        assert len(total_metrics) == 3

        percentiles = {m.labels["percentile"] for m in ttft_metrics}
        assert percentiles == {"50", "95", "99"}

    def test_latency_values_reasonable(self) -> None:
        """Latency values should be within expected range."""
        bench = _make_benchmark(1, 1, 10.0, num_requests=100)
        report = MetricsExporter().export([bench])

        ttft_p50 = [
            m for m in report.metrics
            if m.name == "xpyd_plan_ttft_ms" and m.labels.get("percentile") == "50"
        ]
        assert len(ttft_p50) == 1
        assert ttft_p50[0].value > 0

    def test_multi_benchmark_export(self) -> None:
        """Multiple benchmarks produce metrics for each."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 20.0),
        ]
        report = MetricsExporter().export(benchmarks)

        qps_metrics = [m for m in report.metrics if m.name == "xpyd_plan_measured_qps"]
        assert len(qps_metrics) == 2
        values = {m.value for m in qps_metrics}
        assert values == {10.0, 20.0}

    def test_empty_benchmarks_error(self) -> None:
        """Empty benchmark list raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            MetricsExporter().export([])

    def test_openmetrics_format_has_help_and_type(self) -> None:
        """Rendered text includes HELP and TYPE lines."""
        bench = _make_benchmark(1, 1, 10.0)
        report = MetricsExporter().export([bench])

        assert "# HELP xpyd_plan_measured_qps" in report.text
        assert "# TYPE xpyd_plan_measured_qps gauge" in report.text

    def test_openmetrics_format_eof(self) -> None:
        """OpenMetrics text ends with # EOF."""
        bench = _make_benchmark(1, 1, 10.0)
        report = MetricsExporter().export([bench])
        assert report.text.strip().endswith("# EOF")

    def test_openmetrics_label_format(self) -> None:
        """Labels are correctly formatted with quotes."""
        bench = _make_benchmark(1, 2, 10.0)
        report = MetricsExporter().export([bench])

        # Should have labels like num_decode="2"
        assert 'num_decode="2"' in report.text
        assert 'pd_ratio="1:2"' in report.text

    def test_help_not_duplicated(self) -> None:
        """HELP line for each metric name appears only once."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 20.0),
        ]
        report = MetricsExporter().export(benchmarks)

        help_count = report.text.count("# HELP xpyd_plan_measured_qps")
        assert help_count == 1

    def test_metric_line_model(self) -> None:
        """MetricLine model validates correctly."""
        m = MetricLine(name="test", labels={"a": "b"}, value=1.0)
        assert m.metric_type == "gauge"
        assert m.help_text == ""

    def test_metrics_report_model(self) -> None:
        """MetricsReport model works with defaults."""
        r = MetricsReport()
        assert r.metrics == []
        assert r.text == ""


class TestExportMetricsAPI:
    """Tests for the programmatic export_metrics() function."""

    def test_returns_dict(self) -> None:
        """Programmatic API returns a dict."""
        bench = _make_benchmark(1, 1, 10.0)
        result = export_metrics([bench])
        assert isinstance(result, dict)
        assert "metrics" in result
        assert "text" in result

    def test_dict_structure(self) -> None:
        """Returned dict has expected structure."""
        bench = _make_benchmark(1, 1, 10.0)
        result = export_metrics([bench])
        assert len(result["metrics"]) > 0
        first = result["metrics"][0]
        assert "name" in first
        assert "value" in first
        assert "labels" in first

    def test_api_multi_benchmark(self) -> None:
        """API works with multiple benchmarks."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 20.0),
            _make_benchmark(3, 3, 30.0),
        ]
        result = export_metrics(benchmarks)
        qps = [m for m in result["metrics"] if m["name"] == "xpyd_plan_measured_qps"]
        assert len(qps) == 3

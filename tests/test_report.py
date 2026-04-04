"""Tests for report generation (M7)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2", reason="jinja2 required for report tests")

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.models import SLAConfig
from xpyd_plan.report import (
    ReportGenerator,
    _render_heatmap_svg,
    _render_latency_bars_svg,
    _waste_color,
)


def _make_requests(n: int = 50, base_ttft: float = 30.0, base_tpot: float = 5.0):
    """Generate synthetic benchmark requests."""
    import random

    random.seed(42)
    reqs = []
    for i in range(n):
        reqs.append(BenchmarkRequest(
            request_id=f"req-{i:04d}",
            prompt_tokens=random.randint(50, 500),
            output_tokens=random.randint(20, 200),
            ttft_ms=base_ttft + random.uniform(-10, 20),
            tpot_ms=base_tpot + random.uniform(-2, 3),
            total_latency_ms=base_ttft + base_tpot * random.randint(20, 200),
            timestamp=1000.0 + i * 0.1,
        ))
    return reqs


def _make_benchmark(num_p: int = 2, num_d: int = 2, qps: float = 10.0, n: int = 50):
    """Create a BenchmarkData fixture."""
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_p,
            num_decode_instances=num_d,
            total_instances=num_p + num_d,
            measured_qps=qps,
        ),
        requests=_make_requests(n),
    )


def _make_sla():
    return SLAConfig(ttft_ms=100.0, tpot_ms=20.0, max_latency_ms=5000.0)


def _run_analysis(num_p=2, num_d=2, total=4):
    """Run analysis and return result."""
    analyzer = BenchmarkAnalyzer()
    data = _make_benchmark(num_p, num_d)
    analyzer.load_data_from_dict(data.model_dump())
    sla = _make_sla()
    return analyzer.find_optimal_ratio(total, sla)


# --- Tests for helper functions ---


class TestWasteColor:
    def test_low_waste_is_greenish(self):
        color = _waste_color(0.0)
        assert color.startswith("#")
        # Red component should be low
        r = int(color[1:3], 16)
        assert r < 50

    def test_high_waste_is_reddish(self):
        color = _waste_color(1.0)
        r = int(color[1:3], 16)
        assert r > 200

    def test_mid_waste(self):
        color = _waste_color(0.5)
        assert len(color) == 7  # #RRGGBB


class TestHeatmapSvg:
    def test_empty_candidates(self):
        assert _render_heatmap_svg([]) == ""

    def test_generates_svg(self):
        result = _run_analysis()
        svg = _render_heatmap_svg(result.candidates)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "1P:3D" in svg

    def test_contains_sla_markers(self):
        result = _run_analysis()
        svg = _render_heatmap_svg(result.candidates)
        # At least some candidates should have checkmarks
        assert "✓" in svg


class TestLatencyBarsSvg:
    def test_empty_candidates(self):
        assert _render_latency_bars_svg([]) == ""

    def test_ttft_bars(self):
        result = _run_analysis()
        svg = _render_latency_bars_svg(result.candidates, metric="ttft")
        assert "<svg" in svg
        assert "TTFT P95" in svg

    def test_tpot_bars(self):
        result = _run_analysis()
        svg = _render_latency_bars_svg(result.candidates, metric="tpot")
        assert "<svg" in svg
        assert "TPOT P95" in svg


# --- Tests for ReportGenerator ---


class TestSingleReport:
    def test_generates_html(self):
        result = _run_analysis()
        gen = ReportGenerator()
        html = gen.generate_single(result)
        assert "<!DOCTYPE html>" in html
        assert "xPyD-plan Analysis Report" in html

    def test_contains_candidate_table(self):
        result = _run_analysis()
        gen = ReportGenerator()
        html = gen.generate_single(result)
        assert "Candidate Comparison" in html
        assert "1P:3D" in html

    def test_contains_heatmap(self):
        result = _run_analysis()
        gen = ReportGenerator()
        html = gen.generate_single(result)
        assert "Utilization Heatmap" in html
        assert "<svg" in html

    def test_contains_latency_charts(self):
        result = _run_analysis()
        gen = ReportGenerator()
        html = gen.generate_single(result)
        assert "TTFT Distribution" in html
        assert "TPOT Distribution" in html

    def test_shows_recommendation(self):
        result = _run_analysis()
        gen = ReportGenerator()
        html = gen.generate_single(result)
        if result.best:
            assert "Recommended" in html

    def test_no_passing_candidates(self):
        """Report still generates when no candidate meets SLA."""
        sla = SLAConfig(ttft_ms=1.0, tpot_ms=0.1)  # impossibly tight
        analyzer = BenchmarkAnalyzer()
        data = _make_benchmark()
        analyzer.load_data_from_dict(data.model_dump())
        result = analyzer.find_optimal_ratio(4, sla)
        assert result.best is None

        gen = ReportGenerator()
        html = gen.generate_single(result)
        assert "<!DOCTYPE html>" in html
        assert "No P:D ratio meets SLA" in html

    def test_no_sla_constraints(self):
        """Report works with no SLA constraints set."""
        sla = SLAConfig()
        analyzer = BenchmarkAnalyzer()
        data = _make_benchmark()
        analyzer.load_data_from_dict(data.model_dump())
        result = analyzer.find_optimal_ratio(4, sla)

        gen = ReportGenerator()
        html = gen.generate_single(result)
        assert "<!DOCTYPE html>" in html

    def test_write_to_file(self):
        result = _run_analysis()
        gen = ReportGenerator()
        html = gen.generate_single(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            written = gen.write(html, path)
            assert written.exists()
            content = written.read_text()
            assert "<!DOCTYPE html>" in content

    def test_total_instances_override(self):
        result = _run_analysis()
        gen = ReportGenerator()
        html = gen.generate_single(result, total_instances=8)
        assert "Total instances: 8" in html


class TestMultiReport:
    def _make_multi_result(self):
        analyzer = BenchmarkAnalyzer()
        datasets = []
        for qps in [5.0, 10.0, 20.0]:
            data = _make_benchmark(num_p=2, num_d=2, qps=qps)
            datasets.append(data.model_dump())
        analyzer.load_multi_data_from_dicts(datasets)
        sla = _make_sla()
        return analyzer.find_optimal_ratio_multi(4, sla)

    def test_generates_html(self):
        result = self._make_multi_result()
        gen = ReportGenerator()
        html = gen.generate_multi(result)
        assert "<!DOCTYPE html>" in html
        assert "Multi-Scenario Analysis" in html

    def test_contains_scenarios(self):
        result = self._make_multi_result()
        gen = ReportGenerator()
        html = gen.generate_multi(result)
        assert "QPS=5.0" in html
        assert "QPS=10.0" in html
        assert "QPS=20.0" in html

    def test_contains_unified_recommendation(self):
        result = self._make_multi_result()
        gen = ReportGenerator()
        html = gen.generate_multi(result)
        if result.unified_best:
            assert "Unified Recommendation" in html

    def test_write_to_file(self):
        result = self._make_multi_result()
        gen = ReportGenerator()
        html = gen.generate_multi(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi_report.html"
            written = gen.write(html, path)
            assert written.exists()
            assert "Multi-Scenario" in written.read_text()

    def test_each_scenario_has_charts(self):
        result = self._make_multi_result()
        gen = ReportGenerator()
        html = gen.generate_multi(result)
        # Should have multiple heatmap sections
        assert html.count("Utilization Heatmap") == len(result.scenarios)


class TestCliReport:
    """Test CLI --report flag integration."""

    def test_single_report_cli(self):
        """Test that CLI produces HTML report file."""
        import json
        import tempfile

        from xpyd_plan.cli import main

        data = _make_benchmark()
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_path = Path(tmpdir) / "bench.json"
            bench_path.write_text(json.dumps(data.model_dump()))
            report_path = Path(tmpdir) / "report.html"

            main([
                "analyze",
                "--benchmark", str(bench_path),
                "--sla-ttft", "100",
                "--sla-tpot", "20",
                "--report", str(report_path),
            ])

            assert report_path.exists()
            content = report_path.read_text()
            assert "<!DOCTYPE html>" in content
            assert "xPyD-plan Analysis Report" in content

    def test_multi_report_cli(self):
        """Test CLI multi-scenario report."""
        import json
        import tempfile

        from xpyd_plan.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for qps in [5.0, 15.0]:
                data = _make_benchmark(qps=qps)
                p = Path(tmpdir) / f"bench_{qps}.json"
                p.write_text(json.dumps(data.model_dump()))
                paths.append(str(p))

            report_path = Path(tmpdir) / "multi_report.html"

            main([
                "analyze",
                "--benchmark", *paths,
                "--sla-ttft", "100",
                "--report", str(report_path),
            ])

            assert report_path.exists()
            content = report_path.read_text()
            assert "Multi-Scenario" in content

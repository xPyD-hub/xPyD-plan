"""Tests for Markdown report generation (M25)."""

from __future__ import annotations

from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import (
    AnalysisResult,
    MultiScenarioResult,
    RatioCandidate,
    ScenarioResult,
    SLACheck,
)
from xpyd_plan.md_report import (
    MarkdownReportConfig,
    MarkdownReporter,
    generate_markdown_report,
)

# ── Fixtures ────────────────────────────────────────────────────────────


def _sla_check(meets: bool = True) -> SLACheck:
    return SLACheck(
        ttft_p95_ms=50.0,
        ttft_p99_ms=80.0,
        tpot_p95_ms=10.0,
        tpot_p99_ms=15.0,
        total_latency_p95_ms=200.0,
        total_latency_p99_ms=300.0,
        meets_ttft=meets,
        meets_tpot=meets,
        meets_total_latency=meets,
        meets_all=meets,
    )


def _candidate(p: int, d: int, waste: float, meets_sla: bool = True) -> RatioCandidate:
    return RatioCandidate(
        num_prefill=p,
        num_decode=d,
        prefill_utilization=0.8,
        decode_utilization=0.7,
        waste_rate=waste,
        meets_sla=meets_sla,
        sla_check=_sla_check(meets_sla),
    )


def _analysis_result(
    best_pd: tuple[int, int] | None = (2, 6),
    n_candidates: int = 3,
    has_sla_check: bool = True,
) -> AnalysisResult:
    candidates = [
        _candidate(2, 6, 0.05),
        _candidate(3, 5, 0.12),
        _candidate(4, 4, 0.25),
    ][:n_candidates]
    best = candidates[0] if best_pd else None
    return AnalysisResult(
        best=best,
        candidates=candidates,
        total_instances=8,
        current_sla_check=_sla_check() if has_sla_check else None,
    )


def _multi_result() -> MultiScenarioResult:
    s1 = ScenarioResult(qps=100.0, analysis=_analysis_result())
    s2 = ScenarioResult(qps=200.0, analysis=_analysis_result())
    return MultiScenarioResult(
        scenarios=[s1, s2],
        total_instances=8,
        unified_best=_candidate(2, 6, 0.08),
        per_scenario_best=[_candidate(2, 6, 0.05), _candidate(2, 6, 0.08)],
    )


# ── MarkdownReportConfig tests ─────────────────────────────────────────


class TestMarkdownReportConfig:
    def test_defaults(self):
        cfg = MarkdownReportConfig()
        assert cfg.title == "xPyD-plan Analysis Report"
        assert cfg.include_executive_summary is True
        assert cfg.include_sla_compliance is True
        assert cfg.include_cost_analysis is True
        assert cfg.include_recommendations is True
        assert cfg.include_all_candidates is False
        assert cfg.top_n == 5
        assert cfg.timestamp is True

    def test_custom_values(self):
        cfg = MarkdownReportConfig(title="Custom", top_n=10, timestamp=False)
        assert cfg.title == "Custom"
        assert cfg.top_n == 10
        assert cfg.timestamp is False

    def test_top_n_min(self):
        with pytest.raises(Exception):
            MarkdownReportConfig(top_n=0)


# ── MarkdownReporter single-scenario tests ─────────────────────────────


class TestMarkdownReporterSingle:
    def test_basic_report(self):
        r = MarkdownReporter()
        md = r.generate_single(_analysis_result(), total_instances=8)
        assert "# xPyD-plan Analysis Report" in md
        assert "2P:6D" in md
        assert "Executive Summary" in md

    def test_no_timestamp(self):
        cfg = MarkdownReportConfig(timestamp=False)
        r = MarkdownReporter(cfg)
        md = r.generate_single(_analysis_result(), total_instances=8)
        assert "Generated:" not in md

    def test_with_timestamp(self):
        r = MarkdownReporter()
        md = r.generate_single(_analysis_result(), total_instances=8)
        assert "Generated:" in md
        assert "UTC" in md

    def test_executive_summary_with_best(self):
        r = MarkdownReporter()
        md = r.generate_single(_analysis_result(), total_instances=8)
        assert "Optimal P:D Ratio" in md
        assert "2P:6D" in md
        assert "✅ Yes" in md

    def test_executive_summary_no_best(self):
        result = AnalysisResult(
            best=None,
            candidates=[_candidate(2, 6, 0.05, meets_sla=False)],
            total_instances=8,
        )
        r = MarkdownReporter()
        md = r.generate_single(result, total_instances=8)
        assert "No configuration meets all SLA" in md

    def test_sla_compliance_table(self):
        r = MarkdownReporter()
        md = r.generate_single(_analysis_result(), total_instances=8)
        assert "SLA Compliance" in md
        assert "TTFT" in md
        assert "TPOT" in md
        assert "Total Latency" in md

    def test_sla_from_best_when_no_current(self):
        result = _analysis_result(has_sla_check=False)
        r = MarkdownReporter()
        md = r.generate_single(result, total_instances=8)
        assert "TTFT" in md

    def test_no_sla_data(self):
        result = AnalysisResult(
            best=None,
            candidates=[],
            total_instances=8,
            current_sla_check=None,
        )
        r = MarkdownReporter()
        md = r.generate_single(result, total_instances=8)
        assert "No SLA data available" in md

    def test_candidates_table(self):
        r = MarkdownReporter()
        md = r.generate_single(_analysis_result(), total_instances=8)
        assert "Candidate Ratios" in md
        assert "⭐ Best" in md
        assert "3P:5D" in md

    def test_top_n_limits(self):
        cfg = MarkdownReportConfig(top_n=1)
        r = MarkdownReporter(cfg)
        md = r.generate_single(_analysis_result(), total_instances=8)
        assert "2P:6D" in md
        assert "4P:4D" not in md

    def test_include_all_candidates(self):
        cfg = MarkdownReportConfig(include_all_candidates=True)
        r = MarkdownReporter(cfg)
        md = r.generate_single(_analysis_result(), total_instances=8)
        assert "4P:4D" in md

    def test_cost_analysis(self):
        r = MarkdownReporter()
        md = r.generate_single(_analysis_result(), total_instances=8, cost_per_hour=3.5)
        assert "Cost Analysis" in md
        assert "USD" in md
        assert "3.50" in md

    def test_cost_analysis_custom_currency(self):
        r = MarkdownReporter()
        md = r.generate_single(
            _analysis_result(), total_instances=8, cost_per_hour=2.0, currency="EUR"
        )
        assert "EUR" in md

    def test_no_cost_without_param(self):
        r = MarkdownReporter()
        md = r.generate_single(_analysis_result(), total_instances=8)
        assert "Cost Analysis" not in md

    def test_recommendations_with_best(self):
        r = MarkdownReporter()
        md = r.generate_single(_analysis_result(), total_instances=8)
        assert "Recommendations" in md
        assert "Deploy 2P:6D" in md

    def test_recommendations_high_waste(self):
        result = AnalysisResult(
            best=_candidate(2, 6, 0.4),
            candidates=[_candidate(2, 6, 0.4)],
            total_instances=8,
        )
        r = MarkdownReporter()
        md = r.generate_single(result, total_instances=8)
        assert "Waste rate is high" in md

    def test_recommendations_no_best(self):
        result = AnalysisResult(best=None, candidates=[], total_instances=8)
        r = MarkdownReporter()
        md = r.generate_single(result, total_instances=8)
        assert "No ratio meets SLA" in md

    def test_disable_sections(self):
        cfg = MarkdownReportConfig(
            include_executive_summary=False,
            include_sla_compliance=False,
            include_cost_analysis=False,
            include_recommendations=False,
        )
        r = MarkdownReporter(cfg)
        md = r.generate_single(_analysis_result(), total_instances=8, cost_per_hour=3.0)
        assert "Executive Summary" not in md
        assert "SLA Compliance" not in md
        assert "Cost Analysis" not in md
        assert "Recommendations" not in md


# ── MarkdownReporter multi-scenario tests ──────────────────────────────


class TestMarkdownReporterMulti:
    def test_basic_multi(self):
        r = MarkdownReporter()
        md = r.generate_multi(_multi_result())
        assert "Scenario 1" in md
        assert "Scenario 2" in md
        assert "QPS 100.0" in md
        assert "QPS 200.0" in md

    def test_unified_best(self):
        r = MarkdownReporter()
        md = r.generate_multi(_multi_result())
        assert "Unified optimal" in md
        assert "2P:6D" in md

    def test_no_unified_best(self):
        result = _multi_result()
        result.unified_best = None
        r = MarkdownReporter()
        md = r.generate_multi(result)
        assert "No single ratio meets SLA" in md

    def test_multi_cost(self):
        r = MarkdownReporter()
        md = r.generate_multi(_multi_result(), cost_per_hour=4.0)
        assert "Cost Analysis" in md

    def test_multi_recommendations(self):
        r = MarkdownReporter()
        md = r.generate_multi(_multi_result())
        assert "Recommendations" in md
        assert "Deploy 2P:6D" in md


# ── write() tests ──────────────────────────────────────────────────────


class TestMarkdownReporterWrite:
    def test_write_creates_file(self, tmp_path: Path):
        r = MarkdownReporter()
        md = r.generate_single(_analysis_result(), total_instances=8)
        out = tmp_path / "report.md"
        result_path = r.write(md, out)
        assert result_path == out
        assert out.read_text().startswith("# xPyD-plan")

    def test_write_creates_parent_dirs(self, tmp_path: Path):
        r = MarkdownReporter()
        out = tmp_path / "sub" / "dir" / "report.md"
        r.write("# Test", out)
        assert out.exists()


# ── Programmatic API tests ─────────────────────────────────────────────


class TestGenerateMarkdownReport:
    def test_single_scenario(self):
        md = generate_markdown_report(_analysis_result(), total_instances=8)
        assert "# xPyD-plan Analysis Report" in md
        assert "2P:6D" in md

    def test_multi_scenario(self):
        md = generate_markdown_report(_multi_result())
        assert "Scenario 1" in md

    def test_custom_config(self):
        cfg = MarkdownReportConfig(title="My Report")
        md = generate_markdown_report(_analysis_result(), config=cfg, total_instances=8)
        assert "# My Report" in md

    def test_with_cost(self):
        md = generate_markdown_report(
            _analysis_result(), total_instances=8, cost_per_hour=5.0, currency="GBP"
        )
        assert "GBP" in md

    def test_empty_candidates(self):
        result = AnalysisResult(best=None, candidates=[], total_instances=4)
        md = generate_markdown_report(result, total_instances=4)
        assert "No candidates to display" in md

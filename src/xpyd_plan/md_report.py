"""Markdown report generation for xPyD-plan analysis results.

Generates standalone Markdown reports suitable for GitHub PRs, wiki pages,
and documentation. Complements the existing HTML report generator.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import (
    AnalysisResult,
    MultiScenarioResult,
    RatioCandidate,
)


class MarkdownReportConfig(BaseModel):
    """Configuration for Markdown report generation."""

    title: str = Field("xPyD-plan Analysis Report", description="Report title")
    include_executive_summary: bool = Field(True, description="Include executive summary section")
    include_sla_compliance: bool = Field(True, description="Include SLA compliance section")
    include_cost_analysis: bool = Field(True, description="Include cost analysis section")
    include_recommendations: bool = Field(True, description="Include recommendations section")
    include_all_candidates: bool = Field(
        False, description="Include full candidate table (not just top N)"
    )
    top_n: int = Field(5, ge=1, description="Number of top candidates to show")
    timestamp: bool = Field(True, description="Include generation timestamp")


class MarkdownReporter:
    """Generate Markdown reports from analysis results."""

    def __init__(self, config: MarkdownReportConfig | None = None) -> None:
        self.config = config or MarkdownReportConfig()

    def generate_single(
        self,
        result: AnalysisResult,
        *,
        total_instances: int | None = None,
        cost_per_hour: float | None = None,
        currency: str = "USD",
    ) -> str:
        """Generate a Markdown report for a single-scenario analysis."""
        total = total_instances or result.total_instances
        sections: list[str] = []

        sections.append(f"# {self.config.title}\n")

        if self.config.timestamp:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            sections.append(f"*Generated: {ts}*\n")

        if self.config.include_executive_summary:
            sections.append(self._executive_summary(result, total))

        if self.config.include_sla_compliance:
            sections.append(self._sla_compliance(result))

        candidates = result.candidates
        if self.config.include_all_candidates:
            top = candidates
        else:
            top = candidates[: self.config.top_n]
        sections.append(self._candidates_table(top, result.best))

        if self.config.include_cost_analysis and cost_per_hour is not None:
            sections.append(self._cost_analysis(result, cost_per_hour, currency))

        if self.config.include_recommendations:
            sections.append(self._recommendations_single(result, total))

        return "\n".join(sections)

    def generate_multi(
        self,
        result: MultiScenarioResult,
        *,
        cost_per_hour: float | None = None,
        currency: str = "USD",
    ) -> str:
        """Generate a Markdown report for a multi-scenario analysis."""
        sections: list[str] = []

        sections.append(f"# {self.config.title}\n")

        if self.config.timestamp:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            sections.append(f"*Generated: {ts}*\n")

        if self.config.include_executive_summary:
            sections.append(self._executive_summary_multi(result))

        for i, scenario in enumerate(result.scenarios):
            sections.append(f"## Scenario {i + 1}: QPS {scenario.qps:.1f}\n")
            if self.config.include_sla_compliance:
                sections.append(self._sla_compliance(scenario.analysis))
            candidates = scenario.analysis.candidates
            if self.config.include_all_candidates:
                top = candidates
            else:
                top = candidates[: self.config.top_n]
            sections.append(self._candidates_table(top, scenario.analysis.best))

        if self.config.include_cost_analysis and cost_per_hour is not None:
            # Use unified best for cost analysis
            unified = result.unified_best
            if unified:
                total_cost = unified.total * cost_per_hour
                sections.append("## Cost Analysis\n")
                sections.append(
                    f"| Metric | Value |\n|--------|-------|\n"
                    f"| Unified Optimal | {unified.ratio_str} |\n"
                    f"| Total Instances | {unified.total} |\n"
                    f"| Hourly Cost | {currency} {total_cost:.2f} |\n"
                )

        if self.config.include_recommendations:
            sections.append(self._recommendations_multi(result))

        return "\n".join(sections)

    def write(self, content: str, path: str | Path) -> Path:
        """Write Markdown content to a file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    # ── Private section builders ───────────────────────────────────────

    def _executive_summary(self, result: AnalysisResult, total: int) -> str:
        lines = ["## Executive Summary\n"]
        if result.best:
            b = result.best
            lines.append(
                f"**Optimal P:D Ratio:** {b.ratio_str} "
                f"(waste {b.waste_rate:.1%})\n"
            )
            lines.append(f"- Total instances: {total}")
            lines.append(f"- Prefill: {b.num_prefill}, Decode: {b.num_decode}")
            lines.append(f"- SLA met: {'✅ Yes' if b.meets_sla else '❌ No'}")
        else:
            lines.append("**⚠️ No configuration meets all SLA constraints.**\n")
            sla_pass = [c for c in result.candidates if c.meets_sla]
            lines.append(f"- Candidates evaluated: {len(result.candidates)}")
            lines.append(f"- Candidates meeting SLA: {len(sla_pass)}")
        lines.append("")
        return "\n".join(lines)

    def _executive_summary_multi(self, result: MultiScenarioResult) -> str:
        lines = ["## Executive Summary\n"]
        lines.append(f"- Scenarios analyzed: {len(result.scenarios)}")
        lines.append(f"- Total instances: {result.total_instances}")
        if result.unified_best:
            b = result.unified_best
            lines.append(
                f"- **Unified optimal:** {b.ratio_str} (waste {b.waste_rate:.1%})"
            )
        else:
            lines.append("- **⚠️ No single ratio meets SLA across all scenarios.**")
        lines.append("")
        return "\n".join(lines)

    def _sla_compliance(self, result: AnalysisResult) -> str:
        lines = ["### SLA Compliance\n"]
        if result.current_sla_check:
            s = result.current_sla_check
            lines.append(
                f"| Metric | P95 (ms) | P99 (ms) | Status |\n"
                f"|--------|----------|----------|--------|\n"
                f"| TTFT | {s.ttft_p95_ms:.1f} | {s.ttft_p99_ms:.1f} | "
                f"{'✅' if s.meets_ttft else '❌'} |\n"
                f"| TPOT | {s.tpot_p95_ms:.1f} | {s.tpot_p99_ms:.1f} | "
                f"{'✅' if s.meets_tpot else '❌'} |\n"
                f"| Total Latency | {s.total_latency_p95_ms:.1f} | "
                f"{s.total_latency_p99_ms:.1f} | "
                f"{'✅' if s.meets_total_latency else '❌'} |\n"
            )
        elif result.best and result.best.sla_check:
            s = result.best.sla_check
            lines.append(
                f"| Metric | P95 (ms) | P99 (ms) | Status |\n"
                f"|--------|----------|----------|--------|\n"
                f"| TTFT | {s.ttft_p95_ms:.1f} | {s.ttft_p99_ms:.1f} | "
                f"{'✅' if s.meets_ttft else '❌'} |\n"
                f"| TPOT | {s.tpot_p95_ms:.1f} | {s.tpot_p99_ms:.1f} | "
                f"{'✅' if s.meets_tpot else '❌'} |\n"
                f"| Total Latency | {s.total_latency_p95_ms:.1f} | "
                f"{s.total_latency_p99_ms:.1f} | "
                f"{'✅' if s.meets_total_latency else '❌'} |\n"
            )
        else:
            lines.append("*No SLA data available for current configuration.*\n")
        return "\n".join(lines)

    def _candidates_table(
        self,
        candidates: list[RatioCandidate],
        best: RatioCandidate | None,
    ) -> str:
        lines = ["### Candidate Ratios\n"]
        if not candidates:
            lines.append("*No candidates to display.*\n")
            return "\n".join(lines)

        lines.append(
            "| Ratio | Prefill Util | Decode Util | Waste | SLA | Note |"
        )
        lines.append("|-------|-------------|-------------|-------|-----|------|")
        for c in candidates:
            is_best = (
                best
                and c.num_prefill == best.num_prefill
                and c.num_decode == best.num_decode
            )
            note = "⭐ Best" if is_best else ""
            sla = "✅" if c.meets_sla else "❌"
            lines.append(
                f"| {c.ratio_str} | {c.prefill_utilization:.1%} | "
                f"{c.decode_utilization:.1%} | {c.waste_rate:.1%} | {sla} | {note} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _cost_analysis(
        self,
        result: AnalysisResult,
        cost_per_hour: float,
        currency: str,
    ) -> str:
        lines = ["## Cost Analysis\n"]
        if result.best:
            b = result.best
            total_cost = b.total * cost_per_hour
            lines.append(
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Optimal Ratio | {b.ratio_str} |\n"
                f"| Total Instances | {b.total} |\n"
                f"| Cost/Instance/Hour | {currency} {cost_per_hour:.2f} |\n"
                f"| Total Hourly Cost | {currency} {total_cost:.2f} |\n"
            )
        else:
            lines.append("*No optimal configuration found for cost analysis.*\n")
        return "\n".join(lines)

    def _recommendations_single(self, result: AnalysisResult, total: int) -> str:
        lines = ["## Recommendations\n"]
        if result.best:
            b = result.best
            lines.append(
                f"1. **Deploy {b.ratio_str}** — lowest waste "
                f"({b.waste_rate:.1%}) while meeting SLA."
            )
            if b.waste_rate > 0.3:
                lines.append(
                    f"2. ⚠️ Waste rate is high ({b.waste_rate:.1%}). "
                    "Consider adjusting total instance count."
                )
            sla_candidates = [c for c in result.candidates if c.meets_sla]
            if len(sla_candidates) > 1:
                runner_up = [c for c in sla_candidates if c.ratio_str != b.ratio_str]
                if runner_up:
                    r = runner_up[0]
                    lines.append(
                        f"3. Runner-up: {r.ratio_str} (waste {r.waste_rate:.1%})."
                    )
        else:
            lines.append(
                "1. **No ratio meets SLA.** Consider:\n"
                "   - Increasing total instance count\n"
                "   - Relaxing SLA thresholds\n"
                "   - Investigating bottlenecks in benchmark data"
            )
        lines.append("")
        return "\n".join(lines)

    def _recommendations_multi(self, result: MultiScenarioResult) -> str:
        lines = ["## Recommendations\n"]
        if result.unified_best:
            b = result.unified_best
            lines.append(
                f"1. **Deploy {b.ratio_str}** — meets SLA across "
                f"all {len(result.scenarios)} scenarios."
            )
            lines.append(f"2. Worst-case waste: {b.waste_rate:.1%}.")
        else:
            lines.append(
                "1. **No single ratio meets SLA across all scenarios.** Consider:\n"
                "   - Per-scenario tuning\n"
                "   - Increasing total instance count\n"
                "   - Relaxing SLA thresholds"
            )
        lines.append("")
        return "\n".join(lines)


def generate_markdown_report(
    result: AnalysisResult | MultiScenarioResult,
    *,
    config: MarkdownReportConfig | None = None,
    total_instances: int | None = None,
    cost_per_hour: float | None = None,
    currency: str = "USD",
) -> str:
    """Programmatic API: generate a Markdown report from analysis results.

    Args:
        result: Single or multi-scenario analysis result.
        config: Optional report configuration.
        total_instances: Override total instances (single-scenario only).
        cost_per_hour: GPU cost per hour for cost section.
        currency: Currency label for cost section.

    Returns:
        Markdown string.
    """
    reporter = MarkdownReporter(config)
    if isinstance(result, MultiScenarioResult):
        return reporter.generate_multi(
            result, cost_per_hour=cost_per_hour, currency=currency
        )
    return reporter.generate_single(
        result,
        total_instances=total_instances,
        cost_per_hour=cost_per_hour,
        currency=currency,
    )

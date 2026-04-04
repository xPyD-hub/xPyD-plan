"""Report generation — HTML export with inline SVG visualizations.

Generates self-contained HTML reports from analysis results.
No external dependencies for viewing — all CSS and SVG are inline.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import BaseLoader, Environment

from xpyd_plan.benchmark_models import (
    AnalysisResult,
    MultiScenarioResult,
    RatioCandidate,
)


def _waste_color(waste: float) -> str:
    """Return an RGB hex color for a waste rate (0=green, 1=red)."""
    # Green (low waste) -> Yellow -> Red (high waste)
    if waste <= 0.5:
        r = int(255 * waste * 2)
        g = 200
    else:
        r = 255
        g = int(200 * (1 - (waste - 0.5) * 2))
    return f"#{r:02x}{g:02x}30"


def _render_heatmap_svg(candidates: list[RatioCandidate], width: int = 600) -> str:
    """Render a utilization/waste heatmap as inline SVG.

    Each cell represents a P:D ratio, colored by waste rate.
    """
    if not candidates:
        return ""

    n = len(candidates)
    cell_w = max(width // n, 30)
    actual_w = cell_w * n
    height = 100
    bar_h = 60

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{actual_w}" height="{height}">']
    for i, c in enumerate(candidates):
        x = i * cell_w
        color = _waste_color(c.waste_rate)
        border = "stroke:#333;stroke-width:2" if c.meets_sla else "stroke:#999;stroke-width:1"
        parts.append(
            f'<rect x="{x}" y="0" width="{cell_w}" height="{bar_h}"'
            f' fill="{color}" style="{border}"/>'
        )
        # Label
        fs = min(10, cell_w // 4)
        parts.append(
            f'<text x="{x + cell_w // 2}" y="{bar_h + 15}" text-anchor="middle"'
            f' font-size="{fs}" font-family="monospace">{c.num_prefill}P:{c.num_decode}D</text>'
        )
        # SLA marker
        if c.meets_sla:
            parts.append(
                f'<text x="{x + cell_w // 2}" y="{bar_h // 2 + 5}" text-anchor="middle"'
                f' font-size="12" fill="white">✓</text>'
            )
    # Legend
    parts.append(
        f'<text x="{actual_w - 5}" y="{height - 2}" text-anchor="end"'
        f' font-size="9" fill="#666">Green=low waste, Red=high waste, ✓=meets SLA</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _render_latency_bars_svg(
    candidates: list[RatioCandidate],
    metric: str = "ttft",
    width: int = 600,
) -> str:
    """Render latency bar chart as inline SVG.

    Args:
        candidates: List of ratio candidates.
        metric: One of 'ttft' or 'tpot'.
        width: SVG width.
    """
    if not candidates:
        return ""

    values_p95 = []
    values_p99 = []
    for c in candidates:
        if c.sla_check is None:
            values_p95.append(0.0)
            values_p99.append(0.0)
        elif metric == "ttft":
            values_p95.append(c.sla_check.ttft_p95_ms)
            values_p99.append(c.sla_check.ttft_p99_ms)
        else:
            values_p95.append(c.sla_check.tpot_p95_ms)
            values_p99.append(c.sla_check.tpot_p99_ms)

    max_val = max(max(values_p95, default=1), max(values_p99, default=1), 1)

    n = len(candidates)
    cell_w = max(width // n, 40)
    actual_w = cell_w * n
    chart_h = 120
    height = chart_h + 30
    bar_w = cell_w // 3

    label = "TTFT" if metric == "ttft" else "TPOT"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{actual_w}" height="{height}">']

    for i, c in enumerate(candidates):
        x = i * cell_w
        h95 = (values_p95[i] / max_val) * (chart_h - 10)
        h99 = (values_p99[i] / max_val) * (chart_h - 10)

        # P95 bar
        parts.append(
            f'<rect x="{x + 2}" y="{chart_h - h95}" width="{bar_w}"'
            f' height="{h95}" fill="#4a90d9" opacity="0.8"/>'
        )
        # P99 bar
        parts.append(
            f'<rect x="{x + 2 + bar_w}" y="{chart_h - h99}" width="{bar_w}"'
            f' height="{h99}" fill="#d94a4a" opacity="0.8"/>'
        )
        # Label
        fs = min(10, cell_w // 4)
        parts.append(
            f'<text x="{x + cell_w // 2}" y="{chart_h + 15}" text-anchor="middle"'
            f' font-size="{fs}" font-family="monospace">{c.num_prefill}P:{c.num_decode}D</text>'
        )

    # Legend
    parts.append(
        f'<rect x="5" y="{height - 12}" width="8" height="8" fill="#4a90d9" opacity="0.8"/>'
        f'<text x="16" y="{height - 4}" font-size="9" fill="#666">{label} P95</text>'
        f'<rect x="65" y="{height - 12}" width="8" height="8" fill="#d94a4a" opacity="0.8"/>'
        f'<text x="76" y="{height - 4}" font-size="9" fill="#666">{label} P99</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>xPyD-plan Analysis Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }
  h1 { color: #1a1a2e; border-bottom: 2px solid #4a90d9; padding-bottom: 8px; }
  h2 { color: #2d3436; margin-top: 30px; }
  h3 { color: #4a90d9; }
  .summary-box { background: #f8f9fa; border-left: 4px solid #4a90d9;
                  padding: 15px; margin: 15px 0; border-radius: 4px; }
  .pass { color: #27ae60; font-weight: bold; }
  .fail { color: #e74c3c; font-weight: bold; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; }
  th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
  th { background: #4a90d9; color: white; }
  tr:nth-child(even) { background: #f8f9fa; }
  tr.best { background: #d5f5e3; font-weight: bold; }
  .chart-section { margin: 20px 0; overflow-x: auto; }
  .footer { margin-top: 40px; padding-top: 10px; border-top: 1px solid #ddd;
            font-size: 0.85em; color: #999; }
  .scenario-section { border: 1px solid #e0e0e0; border-radius: 8px;
                      padding: 15px; margin: 15px 0; }
</style>
</head>
<body>
<h1>xPyD-plan Analysis Report</h1>

{% if multi %}
<h2>Multi-Scenario Analysis ({{ scenarios|length }} scenarios)</h2>

{% if unified_best %}
<div class="summary-box">
  <h3>🎯 Unified Recommendation</h3>
  <p><strong>{{ unified_best.num_prefill }}P:{{ unified_best.num_decode }}D</strong>
     — worst-case waste: {{ "%.1f"|format(unified_best.waste_rate * 100) }}%</p>
</div>
{% else %}
<div class="summary-box">
  <h3>❌ No Unified Recommendation</h3>
  <p>No single P:D ratio meets SLA across all scenarios.</p>
</div>
{% endif %}

{% for sc in scenarios %}
<div class="scenario-section">
  <h3>Scenario: QPS={{ "%.1f"|format(sc.qps) }}</h3>
  {% set analysis = sc.analysis %}
  {% include 'single_analysis' %}
</div>
{% endfor %}

{% else %}
{% set analysis = single_analysis %}
{% include 'single_analysis' %}
{% endif %}

<div class="footer">
  Generated by xPyD-plan | Total instances: {{ total_instances }}
</div>
</body>
</html>
"""

_SINGLE_ANALYSIS_TEMPLATE = """\
{% if analysis.current_sla_check %}
<div class="summary-box">
  <h3>📊 Current Configuration</h3>
  <p>SLA: <span class="{{ 'pass' if analysis.current_sla_check.meets_all else 'fail' }}">
    {{ '✅ PASS' if analysis.current_sla_check.meets_all else '❌ FAIL' }}</span></p>
  <p>TTFT P95: {{ "%.1f"|format(analysis.current_sla_check.ttft_p95_ms) }}ms
     | P99: {{ "%.1f"|format(analysis.current_sla_check.ttft_p99_ms) }}ms</p>
  <p>TPOT P95: {{ "%.1f"|format(analysis.current_sla_check.tpot_p95_ms) }}ms
     | P99: {{ "%.1f"|format(analysis.current_sla_check.tpot_p99_ms) }}ms</p>
</div>
{% endif %}

{% if analysis.best %}
<div class="summary-box">
  <h3>✅ Recommended: {{ analysis.best.num_prefill }}P:{{ analysis.best.num_decode }}D</h3>
  <p>Waste: {{ "%.1f"|format(analysis.best.waste_rate * 100) }}%
     | P util: {{ "%.1f"|format(analysis.best.prefill_utilization * 100) }}%
     | D util: {{ "%.1f"|format(analysis.best.decode_utilization * 100) }}%</p>
  {% if analysis.best.sla_check %}
  <p>TTFT P95: {{ "%.1f"|format(analysis.best.sla_check.ttft_p95_ms) }}ms
     | TPOT P95: {{ "%.1f"|format(analysis.best.sla_check.tpot_p95_ms) }}ms</p>
  {% endif %}
</div>
{% else %}
<div class="summary-box">
  <h3>❌ No P:D ratio meets SLA constraints</h3>
</div>
{% endif %}

{% if heatmap_svg %}
<div class="chart-section">
  <h3>Utilization Heatmap</h3>
  {{ heatmap_svg }}
</div>
{% endif %}

{% if ttft_svg %}
<div class="chart-section">
  <h3>TTFT Distribution</h3>
  {{ ttft_svg }}
</div>
{% endif %}

{% if tpot_svg %}
<div class="chart-section">
  <h3>TPOT Distribution</h3>
  {{ tpot_svg }}
</div>
{% endif %}

<h3>Candidate Comparison</h3>
<table>
  <tr>
    <th style="text-align:left">Config</th>
    <th>P Util</th>
    <th>D Util</th>
    <th>Waste</th>
    <th>TTFT P95</th>
    <th>TPOT P95</th>
    <th>SLA</th>
  </tr>
  {% for c in analysis.candidates %}
  <tr{% if analysis.best and c.num_prefill == analysis.best.num_prefill %} class="best"{% endif %}>
    <td style="text-align:left">{{ c.num_prefill }}P:{{ c.num_decode }}D</td>
    <td>{{ "%.1f"|format(c.prefill_utilization * 100) }}%</td>
    <td>{{ "%.1f"|format(c.decode_utilization * 100) }}%</td>
    <td>{{ "%.1f"|format(c.waste_rate * 100) }}%</td>
    <td>{{ "%.1f"|format(c.sla_check.ttft_p95_ms) if c.sla_check else 'N/A' }}</td>
    <td>{{ "%.1f"|format(c.sla_check.tpot_p95_ms) if c.sla_check else 'N/A' }}</td>
    <td>{{ '✅' if c.meets_sla else '❌' }}</td>
  </tr>
  {% endfor %}
</table>
"""


class ReportGenerator:
    """Generate self-contained HTML reports from analysis results."""

    def __init__(self) -> None:
        self._env = Environment(loader=BaseLoader(), autoescape=False)
        self._env.globals["True"] = True
        self._env.globals["False"] = False
        self._single_tmpl = self._env.from_string(_SINGLE_ANALYSIS_TEMPLATE)
        # Register single_analysis as an includable block
        self._main_tmpl_str = _HTML_TEMPLATE

    def generate_single(
        self,
        analysis: AnalysisResult,
        total_instances: int | None = None,
    ) -> str:
        """Generate HTML report for a single-scenario analysis.

        Args:
            analysis: Analysis result to render.
            total_instances: Override for total instances display.

        Returns:
            Self-contained HTML string.
        """
        total = total_instances or analysis.total_instances
        candidates = analysis.candidates

        heatmap_svg = _render_heatmap_svg(candidates)
        ttft_svg = _render_latency_bars_svg(candidates, metric="ttft")
        tpot_svg = _render_latency_bars_svg(candidates, metric="tpot")

        single_html = self._single_tmpl.render(
            analysis=analysis,
            heatmap_svg=heatmap_svg,
            ttft_svg=ttft_svg,
            tpot_svg=tpot_svg,
        )

        # Build main template with single_analysis inlined
        main_tmpl_str = _HTML_TEMPLATE.replace(
            "{% include 'single_analysis' %}", single_html
        )
        tmpl = self._env.from_string(main_tmpl_str)

        return tmpl.render(
            multi=False,
            single_analysis=analysis,
            total_instances=total,
        )

    def generate_multi(
        self,
        result: MultiScenarioResult,
    ) -> str:
        """Generate HTML report for a multi-scenario analysis.

        Args:
            result: Multi-scenario analysis result.

        Returns:
            Self-contained HTML string.
        """
        scenario_htmls = []
        for sc in result.scenarios:
            candidates = sc.analysis.candidates
            heatmap_svg = _render_heatmap_svg(candidates)
            ttft_svg = _render_latency_bars_svg(candidates, metric="ttft")
            tpot_svg = _render_latency_bars_svg(candidates, metric="tpot")
            scenario_htmls.append(self._single_tmpl.render(
                analysis=sc.analysis,
                heatmap_svg=heatmap_svg,
                ttft_svg=ttft_svg,
                tpot_svg=tpot_svg,
            ))

        # Replace include with per-scenario rendered content
        # We need a custom approach: render each scenario separately
        # Build a modified template
        main_str = _HTML_TEMPLATE

        # Replace the scenario loop with pre-rendered content
        loop_start = "{% for sc in scenarios %}"
        loop_end = "{% endfor %}"

        # Find the scenario loop section and replace it
        pre_loop = main_str[: main_str.index(loop_start)]
        post_loop = main_str[main_str.index(loop_end, main_str.index(loop_start)) + len(loop_end) :]

        # Build scenario sections manually
        scenario_sections = []
        for i, sc in enumerate(result.scenarios):
            scenario_sections.append(
                f'<div class="scenario-section">\n'
                f'  <h3>Scenario: QPS={sc.qps:.1f}</h3>\n'
                f'  {scenario_htmls[i]}\n'
                f'</div>'
            )

        combined = pre_loop + "\n".join(scenario_sections) + post_loop

        tmpl = self._env.from_string(combined)
        return tmpl.render(
            multi=True,
            scenarios=result.scenarios,
            unified_best=result.unified_best,
            total_instances=result.total_instances,
        )

    def write(self, html: str, path: str | Path) -> Path:
        """Write HTML report to a file.

        Args:
            html: HTML content.
            path: Output file path.

        Returns:
            Path to the written file.
        """
        p = Path(path)
        p.write_text(html, encoding="utf-8")
        return p

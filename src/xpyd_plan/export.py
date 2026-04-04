"""JSON/CSV export and programmatic API for xpyd-plan.

Provides machine-readable output formats and a clean Python API
for programmatic usage without the CLI.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.benchmark_models import (
    AnalysisResult,
    MultiScenarioResult,
    RatioCandidate,
)
from xpyd_plan.models import SLAConfig


def analyze(
    benchmark_paths: list[str] | str,
    sla_config: SLAConfig | dict | None = None,
    total_instances: int | None = None,
    cost_model_path: str | None = None,
    budget_ceiling: float | None = None,
) -> dict[str, Any]:
    """Programmatic API: analyze benchmark data and return structured results.

    Args:
        benchmark_paths: Path(s) to benchmark JSON file(s).
        sla_config: SLA configuration (SLAConfig object or dict).
        total_instances: Total instances to optimize for (default: from benchmark).
        cost_model_path: Optional YAML path for cost model.
        budget_ceiling: Optional max hourly cost budget.

    Returns:
        Dictionary with analysis results, optionally including cost data.
    """
    from xpyd_plan.bench_adapter import load_benchmark_auto

    if isinstance(benchmark_paths, str):
        benchmark_paths = [benchmark_paths]

    if isinstance(sla_config, dict):
        sla_config = SLAConfig(**sla_config)
    elif sla_config is None:
        sla_config = SLAConfig()

    analyzer = BenchmarkAnalyzer()
    output: dict[str, Any] = {}

    if len(benchmark_paths) == 1:
        data = load_benchmark_auto(benchmark_paths[0])
        analyzer._data = data
        total = total_instances or data.metadata.total_instances
        result = analyzer.find_optimal_ratio(total, sla_config)
        output["mode"] = "single"
        output["analysis"] = json.loads(result.model_dump_json())
    else:
        datasets = [load_benchmark_auto(b) for b in benchmark_paths]
        datasets.sort(key=lambda d: d.metadata.measured_qps)
        analyzer._multi_data = datasets
        analyzer._data = datasets[0]
        total = total_instances or datasets[0].metadata.total_instances
        multi_result = analyzer.find_optimal_ratio_multi(total, sla_config)
        output["mode"] = "multi"
        output["analysis"] = json.loads(multi_result.model_dump_json())

    if cost_model_path:
        from xpyd_plan.cost import CostAnalyzer, CostConfig

        cost_config = CostConfig.from_yaml(cost_model_path)
        cost_analyzer = CostAnalyzer(cost_config)
        if len(benchmark_paths) == 1:
            result = analyzer.find_optimal_ratio(total, sla_config)
            qps = analyzer.data.metadata.measured_qps
            comparison = cost_analyzer.compare(result, qps, budget_ceiling)
            output["cost"] = json.loads(comparison.model_dump_json())

    return output


def result_to_json(result: AnalysisResult | MultiScenarioResult, indent: int = 2) -> str:
    """Convert analysis result to JSON string."""
    return result.model_dump_json(indent=indent)


def _candidate_to_row(candidate: RatioCandidate, scenario_qps: float | None = None) -> dict:
    """Convert a RatioCandidate to a flat dict for CSV export."""
    row: dict[str, Any] = {}
    if scenario_qps is not None:
        row["qps"] = scenario_qps
    row["config"] = candidate.ratio_str
    row["num_prefill"] = candidate.num_prefill
    row["num_decode"] = candidate.num_decode
    row["total_instances"] = candidate.total
    row["prefill_utilization"] = round(candidate.prefill_utilization, 4)
    row["decode_utilization"] = round(candidate.decode_utilization, 4)
    row["waste_rate"] = round(candidate.waste_rate, 4)
    row["meets_sla"] = candidate.meets_sla

    if candidate.sla_check:
        row["ttft_p95_ms"] = round(candidate.sla_check.ttft_p95_ms, 2)
        row["ttft_p99_ms"] = round(candidate.sla_check.ttft_p99_ms, 2)
        row["tpot_p95_ms"] = round(candidate.sla_check.tpot_p95_ms, 2)
        row["tpot_p99_ms"] = round(candidate.sla_check.tpot_p99_ms, 2)
        row["total_latency_p95_ms"] = round(candidate.sla_check.total_latency_p95_ms, 2)
        row["total_latency_p99_ms"] = round(candidate.sla_check.total_latency_p99_ms, 2)
    else:
        for k in [
            "ttft_p95_ms", "ttft_p99_ms", "tpot_p95_ms", "tpot_p99_ms",
            "total_latency_p95_ms", "total_latency_p99_ms",
        ]:
            row[k] = ""

    return row


def result_to_csv(result: AnalysisResult | MultiScenarioResult) -> str:
    """Convert analysis result to CSV string.

    For single analysis: one row per candidate.
    For multi-scenario: one row per candidate per scenario, with qps column.
    """
    rows: list[dict] = []

    if isinstance(result, MultiScenarioResult):
        for scenario in result.scenarios:
            for candidate in scenario.analysis.candidates:
                rows.append(_candidate_to_row(candidate, scenario_qps=scenario.qps))
    else:
        for candidate in result.candidates:
            rows.append(_candidate_to_row(candidate))

    if not rows:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def export_batch(
    benchmark_dir: str,
    sla_config: SLAConfig | dict | None = None,
    output_format: str = "json",
    total_instances: int | None = None,
) -> str:
    """Batch export all benchmark JSON files in a directory.

    Args:
        benchmark_dir: Directory containing benchmark JSON files.
        sla_config: SLA configuration.
        output_format: "json" or "csv".
        total_instances: Total instances to optimize for.

    Returns:
        Formatted string (JSON or CSV).
    """
    from xpyd_plan.bench_adapter import load_benchmark_auto

    if isinstance(sla_config, dict):
        sla_config = SLAConfig(**sla_config)
    elif sla_config is None:
        sla_config = SLAConfig()

    benchmark_files = sorted(Path(benchmark_dir).glob("*.json"))
    if not benchmark_files:
        raise FileNotFoundError(f"No JSON files found in {benchmark_dir}")

    results: list[dict[str, Any]] = []

    for bf in benchmark_files:
        analyzer = BenchmarkAnalyzer()
        data = load_benchmark_auto(str(bf))
        analyzer._data = data
        total = total_instances or data.metadata.total_instances
        result = analyzer.find_optimal_ratio(total, sla_config)

        entry = {
            "file": bf.name,
            "qps": data.metadata.measured_qps,
            "total_instances": total,
            "analysis": json.loads(result.model_dump_json()),
        }
        results.append(entry)

    if output_format == "json":
        return json.dumps(results, indent=2)
    elif output_format == "csv":
        all_rows: list[dict] = []
        for entry in results:
            result_obj = AnalysisResult(**entry["analysis"])
            for candidate in result_obj.candidates:
                row = _candidate_to_row(candidate, scenario_qps=entry["qps"])
                row["file"] = entry["file"]
                all_rows.append(row)
        if not all_rows:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
        return output.getvalue()
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

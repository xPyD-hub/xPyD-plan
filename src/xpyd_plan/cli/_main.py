"""CLI entry point for xpyd-plan."""

from __future__ import annotations

import argparse
import sys

from xpyd_plan.cli._ab_test import _cmd_ab_test
from xpyd_plan.cli._alert import _cmd_alert
from xpyd_plan.cli._analyze import _cmd_analyze
from xpyd_plan.cli._annotate import _cmd_annotate
from xpyd_plan.cli._arrival_pattern import _cmd_arrival_pattern, add_arrival_pattern_parser
from xpyd_plan.cli._batch_analysis import add_batch_analysis_parser
from xpyd_plan.cli._budget import _cmd_budget
from xpyd_plan.cli._budget_tracker import _cmd_budget_tracker
from xpyd_plan.cli._capacity import _cmd_plan_capacity
from xpyd_plan.cli._cdf import _cmd_cdf, add_cdf_parser
from xpyd_plan.cli._cold_start import add_cold_start_parser
from xpyd_plan.cli._compare import _cmd_compare
from xpyd_plan.cli._concurrency_util import _cmd_concurrency_util, add_concurrency_util_parser
from xpyd_plan.cli._confidence import _cmd_confidence
from xpyd_plan.cli._config import _add_config_flag, _apply_config_defaults, _cmd_config
from xpyd_plan.cli._convergence import add_convergence_parser
from xpyd_plan.cli._correlation import _cmd_correlation, add_correlation_parser
from xpyd_plan.cli._cross_validate import add_cross_validate_parser
from xpyd_plan.cli._dashboard import _cmd_dashboard
from xpyd_plan.cli._decompose import _cmd_decompose, add_decompose_parser
from xpyd_plan.cli._dedup import add_dedup_parser
from xpyd_plan.cli._diff_report import register as _register_diff_report
from xpyd_plan.cli._discover import _cmd_discover, add_discover_parser
from xpyd_plan.cli._drift import _cmd_drift, add_drift_parser
from xpyd_plan.cli._error_budget import register as _register_error_budget
from xpyd_plan.cli._export import _cmd_export
from xpyd_plan.cli._fairness import _cmd_fairness, add_fairness_parser
from xpyd_plan.cli._filter import _cmd_filter
from xpyd_plan.cli._fingerprint import _cmd_fingerprint
from xpyd_plan.cli._fleet import _cmd_fleet
from xpyd_plan.cli._forecast import add_forecast_parser
from xpyd_plan.cli._generate import _cmd_generate
from xpyd_plan.cli._goodput import add_goodput_parser
from xpyd_plan.cli._health_check import _cmd_health_check, add_health_check_parser
from xpyd_plan.cli._heatmap import _cmd_heatmap, add_heatmap_parser
from xpyd_plan.cli._interpolate import _cmd_interpolate
from xpyd_plan.cli._jitter import add_jitter_parser
from xpyd_plan.cli._load_profile import add_load_profile_parser
from xpyd_plan.cli._merge import _cmd_merge
from xpyd_plan.cli._metrics import _cmd_metrics, add_metrics_parser
from xpyd_plan.cli._migrate import add_migrate_parser
from xpyd_plan.cli._model_compare import _cmd_model_compare
from xpyd_plan.cli._normalize import add_normalize_parser
from xpyd_plan.cli._outlier_impact import (
    add_outlier_impact_parser,
)
from xpyd_plan.cli._pareto import _cmd_pareto
from xpyd_plan.cli._parquet import add_parquet_parser
from xpyd_plan.cli._pd_imbalance import add_pd_imbalance_parser
from xpyd_plan.cli._pipeline import _cmd_pipeline
from xpyd_plan.cli._plan_benchmarks import _cmd_plan_benchmarks, add_plan_benchmarks_parser
from xpyd_plan.cli._qps_curve import add_qps_curve_parser
from xpyd_plan.cli._queue import add_queue_parser
from xpyd_plan.cli._ratio_compare import add_ratio_compare_parser
from xpyd_plan.cli._recommend import _cmd_recommend
from xpyd_plan.cli._regression import _cmd_regression, add_regression_parser
from xpyd_plan.cli._replay import add_replay_parser
from xpyd_plan.cli._reproducibility import _cmd_reproducibility, add_reproducibility_parser
from xpyd_plan.cli._retry_optimize import add_retry_optimize_parser
from xpyd_plan.cli._retry_sim import add_retry_sim_parser
from xpyd_plan.cli._roi import add_roi_parser
from xpyd_plan.cli._root_cause import _cmd_root_cause, add_root_cause_parser
from xpyd_plan.cli._sample import add_sample_parser
from xpyd_plan.cli._saturation import _cmd_saturation, add_saturation_parser
from xpyd_plan.cli._scaling import _cmd_scaling, add_scaling_parser
from xpyd_plan.cli._scaling_policy import add_scaling_policy_parser
from xpyd_plan.cli._scorecard import _cmd_scorecard, add_scorecard_parser
from xpyd_plan.cli._session import _cmd_session
from xpyd_plan.cli._size_distribution import _cmd_size_distribution, add_size_distribution_parser
from xpyd_plan.cli._sla_tier import add_sla_tier_parser
from xpyd_plan.cli._spike import add_spike_parser
from xpyd_plan.cli._stat_summary import add_stat_summary_parser
from xpyd_plan.cli._summary import _cmd_summary, add_summary_parser
from xpyd_plan.cli._tail import _cmd_tail, add_tail_parser
from xpyd_plan.cli._threshold_advisor import _cmd_threshold_advisor, add_threshold_advisor_parser
from xpyd_plan.cli._throughput import add_throughput_parser
from xpyd_plan.cli._timeline import _cmd_timeline, add_timeline_parser
from xpyd_plan.cli._timeout import add_timeout_parser
from xpyd_plan.cli._token_budget import add_token_budget_parser
from xpyd_plan.cli._token_efficiency import add_token_efficiency_parser, handle_token_efficiency
from xpyd_plan.cli._trend import _cmd_trend
from xpyd_plan.cli._validate import _cmd_validate
from xpyd_plan.cli._warmup_filter import _cmd_warmup_filter, add_warmup_filter_parser
from xpyd_plan.cli._weighted_goodput import register as _register_weighted_goodput
from xpyd_plan.cli._whatif import _cmd_what_if
from xpyd_plan.cli._workload import _cmd_workload


def main(argv: list[str] | None = None) -> None:
    """Entry point for `xpyd-plan` command."""
    parser = argparse.ArgumentParser(
        prog="xpyd-plan",
        description="Analyze benchmark data to find optimal Prefill:Decode instance ratio",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # config subcommand
    config_parser = subparsers.add_parser(
        "config",
        help="Manage configuration profiles",
    )
    config_parser.add_argument(
        "config_action", choices=["init", "show"],
        help="'init' creates a starter config; 'show' displays resolved config",
    )
    config_parser.add_argument(
        "--output-path", type=str, default=None,
        help="Path for 'init' output (default: ./xpyd-plan.yaml)",
    )
    _add_config_flag(config_parser)

    # analyze subcommand (primary)
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze benchmark data to find optimal P:D ratio",
    )
    _add_config_flag(analyze_parser)
    analyze_parser.add_argument(
        "--benchmark", type=str, nargs="+", default=None,
        help="Path(s) to benchmark JSON file(s). Multiple files enable multi-scenario analysis.",
    )
    analyze_parser.add_argument(
        "--format", type=str, choices=["auto", "native", "xpyd-bench"],
        default="auto",
        help="Benchmark data format (default: auto-detect)",
    )
    analyze_parser.add_argument(
        "--stream", action="store_true", default=False,
        help="Streaming mode: read JSONL records from stdin for live analysis",
    )
    analyze_parser.add_argument(
        "--snapshot-interval", type=int, default=None,
        help="Number of requests between streaming snapshots (default: 10)",
    )
    analyze_parser.add_argument(
        "--sla-ttft", type=float, default=None, help="SLA: max TTFT P95 (ms)"
    )
    analyze_parser.add_argument(
        "--sla-tpot", type=float, default=None, help="SLA: max TPOT P95 (ms)"
    )
    analyze_parser.add_argument(
        "--sla-max-latency", type=float, default=None, help="SLA: max total latency P95 (ms)"
    )
    analyze_parser.add_argument(
        "--sla-percentile", type=float, default=95.0,
        help="SLA percentile for evaluation (default: 95.0, range: 1-100)",
    )
    analyze_parser.add_argument(
        "--total-instances", type=int, default=None,
        help="Total instances to optimize for (default: same as benchmark)",
    )
    analyze_parser.add_argument("--top", type=int, default=5, help="Top N candidates to show")
    analyze_parser.add_argument(
        "--sensitivity", action="store_true", default=False,
        help="Run sensitivity analysis (P:D ratio vs SLA margin curves)",
    )
    analyze_parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    analyze_parser.add_argument(
        "--report", type=str, default=None,
        help="Generate report to the given path (e.g. report.html or report.md)",
    )
    analyze_parser.add_argument(
        "--report-format", type=str, default="html", choices=["html", "markdown"],
        help="Report format: html (default) or markdown",
    )
    analyze_parser.add_argument(
        "--cost-model", type=str, default=None,
        help="YAML file with GPU cost config (gpu_hourly_rate, currency)",
    )
    analyze_parser.add_argument(
        "--budget-ceiling", type=float, default=None,
        help="Max hourly cost budget (used with --cost-model)",
    )
    analyze_parser.add_argument(
        "--output-format", type=str, choices=["table", "json", "csv"],
        default="table",
        help="Output format: table (Rich), json, or csv (default: table)",
    )
    analyze_parser.add_argument(
        "--validate", action="store_true", default=False,
        help="Validate data and filter outliers before analysis",
    )
    analyze_parser.add_argument(
        "--outlier-method", type=str, choices=["iqr", "zscore"], default="iqr",
        help="Outlier detection method for --validate (default: iqr)",
    )

    # export subcommand
    export_parser = subparsers.add_parser(
        "export",
        help="Batch export benchmark analysis results from a directory",
    )
    _add_config_flag(export_parser)
    export_parser.add_argument(
        "--dir", type=str, required=True,
        help="Directory containing benchmark JSON files",
    )
    export_parser.add_argument(
        "--output-format", type=str, choices=["json", "csv"], default="json",
        help="Export format (default: json)",
    )
    export_parser.add_argument(
        "--sla-ttft", type=float, default=None, help="SLA: max TTFT P95 (ms)",
    )
    export_parser.add_argument(
        "--sla-tpot", type=float, default=None, help="SLA: max TPOT P95 (ms)",
    )
    export_parser.add_argument(
        "--sla-max-latency", type=float, default=None, help="SLA: max total latency P95 (ms)",
    )
    export_parser.add_argument(
        "--sla-percentile", type=float, default=95.0,
        help="SLA percentile for evaluation (default: 95.0, range: 1-100)",
    )
    export_parser.add_argument(
        "--total-instances", type=int, default=None,
        help="Total instances to optimize for",
    )

    # plan-capacity subcommand
    cap_parser = subparsers.add_parser(
        "plan-capacity",
        help="Recommend minimum instances and P:D ratio for a target QPS",
    )
    _add_config_flag(cap_parser)
    cap_parser.add_argument(
        "--benchmark", type=str, nargs="+", required=True,
        help="Benchmark JSON files at different cluster sizes/QPS levels",
    )
    cap_parser.add_argument(
        "--target-qps", type=float, required=True,
        help="Target QPS to achieve",
    )
    cap_parser.add_argument(
        "--sla-ttft", type=float, default=None, help="SLA: max TTFT P95 (ms)",
    )
    cap_parser.add_argument(
        "--sla-tpot", type=float, default=None, help="SLA: max TPOT P95 (ms)",
    )
    cap_parser.add_argument(
        "--sla-max-latency", type=float, default=None, help="SLA: max total latency P95 (ms)",
    )
    cap_parser.add_argument(
        "--sla-percentile", type=float, default=95.0,
        help="SLA percentile for evaluation (default: 95.0, range: 1-100)",
    )
    cap_parser.add_argument(
        "--max-instances", type=int, default=64,
        help="Maximum instances to consider (default: 64)",
    )
    cap_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # what-if subcommand
    whatif_parser = subparsers.add_parser(
        "what-if",
        help="Simulate what-if scenarios: scale QPS or change instance count",
    )
    _add_config_flag(whatif_parser)
    whatif_parser.add_argument(
        "--benchmark", type=str, required=True,
        help="Path to benchmark JSON file",
    )
    whatif_parser.add_argument(
        "--scale-qps", type=str, default=None,
        help="QPS multiplier(s), comma-separated (e.g. '0.5,1.5,2.0' or '2x')",
    )
    whatif_parser.add_argument(
        "--add-instances", type=int, default=None,
        help="Number of instances to add (negative to remove)",
    )
    whatif_parser.add_argument(
        "--sla-ttft", type=float, default=None, help="SLA: max TTFT P95 (ms)",
    )
    whatif_parser.add_argument(
        "--sla-tpot", type=float, default=None, help="SLA: max TPOT P95 (ms)",
    )
    whatif_parser.add_argument(
        "--sla-max-latency", type=float, default=None, help="SLA: max total latency P95 (ms)",
    )
    whatif_parser.add_argument(
        "--sla-percentile", type=float, default=95.0,
        help="SLA percentile for evaluation (default: 95.0, range: 1-100)",
    )
    whatif_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two benchmark datasets and detect regressions",
    )
    compare_parser.add_argument(
        "--baseline", type=str, required=True,
        help="Path to baseline benchmark JSON file",
    )
    compare_parser.add_argument(
        "--current", type=str, required=True,
        help="Path to current benchmark JSON file",
    )
    compare_parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Regression threshold as fraction (default: 0.1 = 10%%)",
    )
    compare_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate benchmark data quality and detect outliers",
    )
    validate_parser.add_argument(
        "--benchmark", type=str, required=True,
        help="Path to benchmark JSON file",
    )
    validate_parser.add_argument(
        "--outlier-method", type=str, choices=["iqr", "zscore"], default="iqr",
        help="Outlier detection method (default: iqr)",
    )
    validate_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # trend subcommand
    trend_parser = subparsers.add_parser(
        "trend",
        help="Track benchmark performance trends over time",
    )
    trend_parser.add_argument(
        "trend_action", choices=["add", "show", "check"],
        help="'add' records a benchmark, 'show' lists history, 'check' detects degradation",
    )
    trend_parser.add_argument(
        "--benchmark", type=str, default=None,
        help="Path to benchmark JSON file (required for 'add')",
    )
    trend_parser.add_argument(
        "--label", type=str, default=None,
        help="Run label/tag (required for 'add')",
    )
    trend_parser.add_argument(
        "--db", type=str, default=None,
        help="Path to trend SQLite database (default: xpyd-plan-trend.db)",
    )
    trend_parser.add_argument(
        "--lookback", type=int, default=10,
        help="Number of recent entries to analyze (default: 10)",
    )
    trend_parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Degradation threshold as fraction (default: 0.1 = 10%%)",
    )
    trend_parser.add_argument(
        "--limit", type=int, default=None,
        help="Max entries to show (default: all)",
    )
    trend_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # dashboard subcommand
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Interactive TUI dashboard for real-time benchmark monitoring",
    )

    # --- interpolate subcommand ---
    interp_parser = subparsers.add_parser(
        "interpolate",
        help="Predict performance at untested P:D ratios via interpolation",
    )
    interp_parser.add_argument(
        "--benchmark", type=str, nargs="+", required=True,
        help="Benchmark JSON files (at least 2, different P:D ratios, same total instances)",
    )
    interp_parser.add_argument(
        "--method", type=str, default="linear", choices=["linear", "spline"],
        help="Interpolation method (default: linear)",
    )
    interp_parser.add_argument(
        "--ratios", type=str, nargs="*", default=None,
        help="P:D ratios to predict, e.g. '2:6' '3:5'. Omit to predict all.",
    )
    interp_parser.add_argument(
        "--output-format", type=str, default="table", choices=["table", "json"],
        help="Output format (default: table)",
    )

    dashboard_parser.add_argument(
        "--benchmark", type=str, default=None,
        help="Path to benchmark JSON file (static mode)",
    )
    dashboard_parser.add_argument(
        "--stream", action="store_true", default=False,
        help="Read JSONL from stdin (streaming mode)",
    )
    dashboard_parser.add_argument(
        "--refresh-interval", type=float, default=2.0,
        help="Refresh interval in seconds (default: 2.0)",
    )
    dashboard_parser.add_argument(
        "--max-ttft-ms", type=float, default=None,
        help="Max TTFT SLA threshold (ms)",
    )
    dashboard_parser.add_argument(
        "--max-tpot-ms", type=float, default=None,
        help="Max TPOT SLA threshold (ms)",
    )
    dashboard_parser.add_argument(
        "--max-total-latency-ms", type=float, default=None,
        help="Max total latency SLA threshold (ms)",
    )
    dashboard_parser.add_argument(
        "--total-instances", type=int, default=None,
        help="Override total instance count",
    )
    dashboard_parser.add_argument(
        "--num-prefill", type=int, default=None,
        help="Prefill instances for streaming metadata",
    )
    dashboard_parser.add_argument(
        "--num-decode", type=int, default=None,
        help="Decode instances for streaming metadata",
    )

    alert_parser = subparsers.add_parser(
        "alert", help="Evaluate benchmark against alert rules",
    )
    alert_parser.add_argument(
        "--benchmark", required=True, help="Path to benchmark JSON file",
    )
    alert_parser.add_argument(
        "--rules", required=True, help="Path to alert rules YAML file",
    )
    alert_parser.add_argument(
        "--output-format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # -- annotate subcommand --
    annotate_parser = subparsers.add_parser(
        "annotate", help="Manage benchmark annotations and tags",
    )
    annotate_sub = annotate_parser.add_subparsers(dest="annotate_action", help="Annotation actions")

    ann_add = annotate_sub.add_parser("add", help="Add tags to a benchmark file")
    ann_add.add_argument("--benchmark", required=True, help="Path to benchmark JSON file")
    ann_add.add_argument(
        "--tag", action="append", required=True, metavar="KEY=VALUE",
        help="Tag to add (can be repeated)",
    )

    ann_list = annotate_sub.add_parser("list", help="List tags on a benchmark file")
    ann_list.add_argument("--benchmark", required=True, help="Path to benchmark JSON file")
    ann_list.add_argument(
        "--output-format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    ann_remove = annotate_sub.add_parser("remove", help="Remove tags from a benchmark file")
    ann_remove.add_argument("--benchmark", required=True, help="Path to benchmark JSON file")
    ann_remove.add_argument(
        "--key", action="append", required=True, help="Tag key to remove (can be repeated)",
    )

    ann_filter = annotate_sub.add_parser("filter", help="Filter benchmarks by tags")
    ann_filter.add_argument("--dir", required=True, help="Directory to scan")
    ann_filter.add_argument(
        "--tag", action="append", required=True, metavar="KEY=VALUE",
        help="Required tag (can be repeated, all must match)",
    )
    ann_filter.add_argument(
        "--output-format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    ann_clear = annotate_sub.add_parser("clear", help="Remove all tags from a benchmark file")
    ann_clear.add_argument("--benchmark", required=True, help="Path to benchmark JSON file")

    # pareto subcommand
    pareto_parser = subparsers.add_parser(
        "pareto",
        help="Find Pareto-optimal P:D ratios across latency, cost, and waste",
    )
    _add_config_flag(pareto_parser)
    pareto_parser.add_argument(
        "--benchmark", type=str, nargs="+", required=True,
        help="Benchmark JSON file(s)",
    )
    pareto_parser.add_argument(
        "--sla-ttft", type=float, default=None, help="SLA: max TTFT P95 (ms)",
    )
    pareto_parser.add_argument(
        "--sla-tpot", type=float, default=None, help="SLA: max TPOT P95 (ms)",
    )
    pareto_parser.add_argument(
        "--sla-max-latency", type=float, default=None, help="SLA: max total latency P95 (ms)",
    )
    pareto_parser.add_argument(
        "--sla-percentile", type=float, default=95.0,
        help="SLA percentile for evaluation (default: 95.0, range: 1-100)",
    )
    pareto_parser.add_argument(
        "--total-instances", type=int, default=None,
        help="Total instances to optimize for",
    )
    pareto_parser.add_argument(
        "--cost-model", type=str, default=None,
        help="YAML file with GPU cost config",
    )
    pareto_parser.add_argument(
        "--objectives", type=str, nargs="+", default=None,
        choices=["latency", "cost", "waste"],
        help="Objectives to optimize (default: latency waste; adds cost with --cost-model)",
    )
    pareto_parser.add_argument(
        "--weights", type=str, default=None,
        help="Objective weights as 'latency=1,cost=2,waste=1'",
    )
    pareto_parser.add_argument(
        "--include-dominated", action="store_true", default=False,
        help="Also show dominated (non-optimal) candidates",
    )
    pareto_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # recommend subcommand
    recommend_parser = subparsers.add_parser(
        "recommend",
        help="Generate actionable P:D ratio recommendations",
    )
    _add_config_flag(recommend_parser)
    recommend_parser.add_argument(
        "--benchmark", type=str, nargs="+", required=True,
        help="Benchmark JSON file(s)",
    )
    recommend_parser.add_argument(
        "--sla-ttft", type=float, default=None, help="SLA: max TTFT P95 (ms)",
    )
    recommend_parser.add_argument(
        "--sla-tpot", type=float, default=None, help="SLA: max TPOT P95 (ms)",
    )
    recommend_parser.add_argument(
        "--sla-max-latency", type=float, default=None, help="SLA: max total latency P95 (ms)",
    )
    recommend_parser.add_argument(
        "--sla-percentile", type=float, default=95.0,
        help="SLA percentile for evaluation (default: 95.0, range: 1-100)",
    )
    recommend_parser.add_argument(
        "--total-instances", type=int, default=None,
        help="Total instances to optimize for",
    )
    recommend_parser.add_argument(
        "--cost-model", type=str, default=None,
        help="YAML file with GPU cost config",
    )
    recommend_parser.add_argument(
        "--waste-threshold", type=float, default=0.3,
        help="Waste rate threshold for rebalance recommendations (default: 0.3)",
    )
    recommend_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # fleet subcommand
    fleet_parser = subparsers.add_parser(
        "fleet",
        help="Calculate optimal fleet sizing across multiple GPU types",
    )
    _add_config_flag(fleet_parser)
    fleet_parser.add_argument(
        "--gpu-configs", type=str, required=True,
        help="YAML file with GPU type configurations (name, benchmark_file, hourly_rate)",
    )
    fleet_parser.add_argument(
        "--target-qps", type=float, required=True,
        help="Target QPS to achieve",
    )
    fleet_parser.add_argument(
        "--sla-ttft", type=float, default=None, help="SLA: max TTFT P95 (ms)",
    )
    fleet_parser.add_argument(
        "--sla-tpot", type=float, default=None, help="SLA: max TPOT P95 (ms)",
    )
    fleet_parser.add_argument(
        "--sla-max-latency", type=float, default=None, help="SLA: max total latency P95 (ms)",
    )
    fleet_parser.add_argument(
        "--sla-percentile", type=float, default=95.0,
        help="SLA percentile for evaluation (default: 95.0)",
    )
    fleet_parser.add_argument(
        "--budget-ceiling", type=float, default=None,
        help="Max hourly budget (filters out options above this)",
    )
    fleet_parser.add_argument(
        "--max-options", type=int, default=20,
        help="Max fleet options to return (default: 20)",
    )
    fleet_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # pipeline subcommand
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run a batch analysis pipeline from YAML config",
    )
    pipeline_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to pipeline YAML config file",
    )
    pipeline_parser.add_argument(
        "--benchmark", type=str, nargs="+",
        help="Benchmark JSON file(s)",
    )
    pipeline_parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Preview pipeline steps without executing",
    )
    pipeline_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # generate subcommand
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic benchmark data for testing and demos",
    )
    generate_parser.add_argument(
        "--config", type=str, default=None,
        help="Path to generator YAML config file",
    )
    generate_parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output benchmark JSON file path",
    )
    generate_parser.add_argument(
        "--num-requests", type=int, default=None,
        help="Number of requests to generate (overrides config)",
    )
    generate_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    generate_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # budget subcommand
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge multiple benchmark files into one aggregated dataset",
    )
    merge_parser.add_argument(
        "--benchmark", type=str, action="append", required=True,
        help="Benchmark JSON file (specify multiple times)",
    )
    merge_parser.add_argument(
        "--output", type=str, required=True,
        help="Output file path for merged benchmark JSON",
    )
    merge_parser.add_argument(
        "--strategy", type=str, default="union",
        choices=["union", "intersection"],
        help="Merge strategy for overlapping request IDs (default: union)",
    )
    merge_parser.add_argument(
        "--no-config-check", action="store_true",
        help="Allow merging benchmarks with different cluster configurations",
    )
    merge_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    budget_parser = subparsers.add_parser(
        "budget",
        help="Allocate SLA budget across prefill and decode stages",
    )
    budget_parser.add_argument(
        "--benchmark", type=str, required=True,
        help="Benchmark JSON file",
    )
    budget_parser.add_argument(
        "--total-budget-ms", type=float, required=True,
        help="Total latency budget in milliseconds",
    )
    budget_parser.add_argument(
        "--strategy", type=str, default="proportional",
        choices=["proportional", "balanced", "ttft-priority", "tpot-priority"],
        help="Budget allocation strategy (default: proportional)",
    )
    budget_parser.add_argument(
        "--sla-percentile", type=float, default=95.0,
        help="Percentile for observed latency analysis (default: 95)",
    )
    budget_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # filter subcommand
    filter_parser = subparsers.add_parser(
        "filter",
        help="Filter benchmark data by token count, latency, time window, or sampling",
    )
    filter_parser.add_argument(
        "--benchmark", type=str, required=True,
        help="Benchmark JSON file",
    )
    filter_parser.add_argument(
        "--output", type=str, required=True,
        help="Output file path for filtered benchmark JSON",
    )
    filter_parser.add_argument(
        "--min-prompt-tokens", type=int, default=None,
        help="Minimum prompt token count",
    )
    filter_parser.add_argument(
        "--max-prompt-tokens", type=int, default=None,
        help="Maximum prompt token count",
    )
    filter_parser.add_argument(
        "--min-output-tokens", type=int, default=None,
        help="Minimum output token count",
    )
    filter_parser.add_argument(
        "--max-output-tokens", type=int, default=None,
        help="Maximum output token count",
    )
    filter_parser.add_argument(
        "--min-ttft-ms", type=float, default=None,
        help="Minimum TTFT (ms)",
    )
    filter_parser.add_argument(
        "--max-ttft-ms", type=float, default=None,
        help="Maximum TTFT (ms)",
    )
    filter_parser.add_argument(
        "--min-tpot-ms", type=float, default=None,
        help="Minimum TPOT (ms)",
    )
    filter_parser.add_argument(
        "--max-tpot-ms", type=float, default=None,
        help="Maximum TPOT (ms)",
    )
    filter_parser.add_argument(
        "--min-total-latency-ms", type=float, default=None,
        help="Minimum total latency (ms)",
    )
    filter_parser.add_argument(
        "--max-total-latency-ms", type=float, default=None,
        help="Maximum total latency (ms)",
    )
    filter_parser.add_argument(
        "--time-start", type=float, default=None,
        help="Start timestamp (epoch seconds)",
    )
    filter_parser.add_argument(
        "--time-end", type=float, default=None,
        help="End timestamp (epoch seconds)",
    )
    filter_parser.add_argument(
        "--sample-count", type=int, default=None,
        help="Random sample N requests",
    )
    filter_parser.add_argument(
        "--sample-fraction", type=float, default=None,
        help="Random sample fraction (0, 1]",
    )
    filter_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible sampling",
    )
    filter_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # --- confidence subcommand ---
    confidence_parser = subparsers.add_parser(
        "confidence",
        help="Bootstrap confidence intervals for latency percentiles",
    )
    confidence_parser.add_argument(
        "--benchmark", type=str, nargs="+", required=True,
        help="Benchmark JSON file(s)",
    )
    confidence_parser.add_argument(
        "--percentile", type=float, default=95.0,
        help="Latency percentile to evaluate (default: 95)",
    )
    confidence_parser.add_argument(
        "--confidence-level", type=float, default=0.95,
        help="Confidence level for intervals (default: 0.95)",
    )
    confidence_parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of bootstrap iterations (default: 1000)",
    )
    confidence_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    confidence_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # --- model-compare subcommand ---
    mc_parser = subparsers.add_parser(
        "model-compare",
        help="Compare benchmark results across different LLM models",
    )
    mc_parser.add_argument(
        "--benchmarks", type=str, nargs="+", required=True,
        help="Benchmark JSON files (one per model)",
    )
    mc_parser.add_argument(
        "--models", type=str, required=True,
        help="Comma-separated model names (must match number of benchmark files)",
    )
    mc_parser.add_argument(
        "--gpu-hourly-rate", type=float, default=None,
        help="GPU hourly rate for cost-efficiency ranking",
    )
    mc_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # --- ab-test subcommand ---
    ab_parser = subparsers.add_parser(
        "ab-test",
        help="A/B test analysis: compare control vs treatment benchmarks",
    )
    ab_parser.add_argument(
        "--control", type=str, required=True,
        help="Control (baseline) benchmark JSON file",
    )
    ab_parser.add_argument(
        "--treatment", type=str, required=True,
        help="Treatment benchmark JSON file",
    )
    ab_parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level (default: 0.05)",
    )
    ab_parser.add_argument(
        "--metric", type=str, nargs="+", default=None,
        help="Metrics to compare (default: ttft_ms tpot_ms total_latency_ms)",
    )
    ab_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # --- workload subcommand ---
    workload_parser = subparsers.add_parser(
        "workload",
        help="Workload characterization: segment requests by token patterns",
    )
    workload_parser.add_argument(
        "--benchmark", type=str, nargs="+", required=True,
        help="Benchmark JSON file(s)",
    )
    workload_parser.add_argument(
        "--sla-ttft", type=float, default=None,
        help="SLA TTFT threshold (ms)",
    )
    workload_parser.add_argument(
        "--sla-tpot", type=float, default=None,
        help="SLA TPOT threshold (ms)",
    )
    workload_parser.add_argument(
        "--sla-max-latency", type=float, default=None,
        help="SLA max total latency threshold (ms)",
    )
    workload_parser.add_argument(
        "--sla-percentile", type=float, default=95.0,
        help="SLA evaluation percentile (default: 95)",
    )
    workload_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    # --- metrics subcommand ---
    add_metrics_parser(subparsers)

    # --- scaling subcommand ---
    add_saturation_parser(subparsers)
    add_scaling_parser(subparsers)

    # --- correlation subcommand ---
    add_correlation_parser(subparsers)
    add_jitter_parser(subparsers)
    add_cold_start_parser(subparsers)
    add_dedup_parser(subparsers)
    add_timeout_parser(subparsers)
    add_ratio_compare_parser(subparsers)
    add_pd_imbalance_parser(subparsers)
    add_spike_parser(subparsers)

    # --- goodput subcommand ---
    add_goodput_parser(subparsers)

    # --- weighted-goodput subcommand ---
    _register_weighted_goodput(subparsers)

    # --- diff-report subcommand ---
    _register_diff_report(subparsers)
    _register_error_budget(subparsers)

    # --- fairness subcommand ---
    add_fairness_parser(subparsers)

    # --- timeline subcommand ---
    add_throughput_parser(subparsers)
    add_queue_parser(subparsers)
    add_batch_analysis_parser(subparsers)
    add_stat_summary_parser(subparsers)
    add_migrate_parser(subparsers)
    add_cdf_parser(subparsers)
    add_concurrency_util_parser(subparsers)
    add_reproducibility_parser(subparsers)
    add_regression_parser(subparsers)
    add_replay_parser(subparsers)
    add_roi_parser(subparsers)
    add_sample_parser(subparsers)
    add_size_distribution_parser(subparsers)
    add_token_budget_parser(subparsers)
    add_normalize_parser(subparsers)
    add_cross_validate_parser(subparsers)
    add_scaling_policy_parser(subparsers)
    add_parquet_parser(subparsers)
    add_qps_curve_parser(subparsers)
    add_retry_optimize_parser(subparsers)
    add_retry_sim_parser(subparsers)
    add_token_efficiency_parser(subparsers)
    add_timeline_parser(subparsers)

    # --- drift subcommand ---
    add_drift_parser(subparsers)

    # --- root-cause subcommand ---
    add_root_cause_parser(subparsers)

    # --- summary subcommand ---
    add_summary_parser(subparsers)

    # --- outlier-impact subcommand ---
    add_outlier_impact_parser(subparsers)

    # --- tail subcommand ---
    add_tail_parser(subparsers)
    add_convergence_parser(subparsers)
    add_load_profile_parser(subparsers)

    # --- scorecard subcommand ---
    add_scorecard_parser(subparsers)

    # --- discover subcommand ---
    add_discover_parser(subparsers)
    add_heatmap_parser(subparsers)
    add_health_check_parser(subparsers)
    add_warmup_filter_parser(subparsers)
    add_arrival_pattern_parser(subparsers)

    # --- plan-benchmarks subcommand ---
    add_plan_benchmarks_parser(subparsers)

    # --- threshold-advisor subcommand ---
    add_threshold_advisor_parser(subparsers)

    # --- forecast subcommand ---
    add_forecast_parser(subparsers)
    add_sla_tier_parser(subparsers)
    add_decompose_parser(subparsers)

    # --- plugins subcommand ---
    from xpyd_plan.cli._plugins import add_plugins_subcommand

    add_plugins_subcommand(subparsers)

    # Let plugins register their own CLI subcommands
        # --- fingerprint subcommand ---
    fingerprint_parser = subparsers.add_parser(
        "fingerprint",
        help="Extract or compare environment fingerprints from benchmark data",
    )
    fingerprint_parser.add_argument(
        "--benchmark", type=str, required=True,
        help="Path to benchmark JSON file",
    )
    fingerprint_parser.add_argument(
        "--compare", type=str, default=None,
        help="Path to second benchmark file to compare against",
    )
    fingerprint_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

        # --- budget-tracker subcommand ---
    bt_parser = subparsers.add_parser(
        "budget-tracker",
        help="Track how close requests are to exhausting SLA budgets",
    )
    bt_parser.add_argument(
        "--benchmark", type=str, required=True,
        help="Path to benchmark JSON file",
    )
    bt_parser.add_argument(
        "--sla-ttft", type=float, default=None,
        help="TTFT SLA threshold in milliseconds",
    )
    bt_parser.add_argument(
        "--sla-tpot", type=float, default=None,
        help="TPOT SLA threshold in milliseconds",
    )
    bt_parser.add_argument(
        "--sla-total", type=float, default=None,
        help="Total latency SLA threshold in milliseconds",
    )
    bt_parser.add_argument(
        "--near-miss-threshold", type=float, default=0.8,
        help="Budget consumption ratio above which requests are near-miss (default: 0.8)",
    )
    bt_parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of worst requests to show (default: 10)",
    )
    bt_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

        # --- session subcommand ---
    session_parser = subparsers.add_parser(
        "session",
        help="Manage benchmark sessions (group related benchmark files)",
    )
    session_parser.add_argument(
        "session_action", choices=["create", "add", "list", "show", "delete", "remove"],
        help="Session action",
    )
    session_parser.add_argument(
        "--name", type=str, default=None,
        help="Session name",
    )
    session_parser.add_argument(
        "--description", type=str, default=None,
        help="Session description (for 'create')",
    )
    session_parser.add_argument(
        "--tags", type=str, default=None,
        help="Comma-separated tags (for 'create')",
    )
    session_parser.add_argument(
        "--benchmark", type=str, default=None,
        help="Path to benchmark JSON file (for 'add' and 'remove')",
    )
    session_parser.add_argument(
        "--db", type=str, default=None,
        help="Path to session SQLite database (default: xpyd-plan-sessions.db)",
    )
    session_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    from xpyd_plan.plugin import get_registry

    get_registry().register_all_cli(subparsers)

    args = parser.parse_args(argv)

    if args.command == "config":
        _cmd_config(args)
    elif args.command == "analyze":
        _apply_config_defaults(args)
        _cmd_analyze(args)
    elif args.command == "export":
        _apply_config_defaults(args)
        _cmd_export(args)
    elif args.command == "plan-capacity":
        _apply_config_defaults(args)
        _cmd_plan_capacity(args)
    elif args.command == "what-if":
        _apply_config_defaults(args)
        _cmd_what_if(args)
    elif args.command == "compare":
        _cmd_compare(args)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "trend":
        _cmd_trend(args)
    elif args.command == "dashboard":
        _cmd_dashboard(args)
    elif args.command == "interpolate":
        _cmd_interpolate(args)
    elif args.command == "alert":
        _cmd_alert(args)
    elif args.command == "annotate":
        _cmd_annotate(args)
    elif args.command == "pareto":
        _apply_config_defaults(args)
        _cmd_pareto(args)
    elif args.command == "recommend":
        _apply_config_defaults(args)
        _cmd_recommend(args)
    elif args.command == "fleet":
        _apply_config_defaults(args)
        _cmd_fleet(args)
    elif args.command == "pipeline":
        _cmd_pipeline(args)
    elif args.command == "generate":
        _cmd_generate(args)
    elif args.command == "budget":
        _cmd_budget(args)
    elif args.command == "merge":
        _cmd_merge(args)
    elif args.command == "filter":
        _cmd_filter(args)
    elif args.command == "confidence":
        _cmd_confidence(args)
    elif args.command == "model-compare":
        _cmd_model_compare(args)
    elif args.command == "ab-test":
        _cmd_ab_test(args)
    elif args.command == "workload":
        _cmd_workload(args)
    elif args.command == "saturation":
        _cmd_saturation(args)
    elif args.command == "scaling":
        _cmd_scaling(args)
    elif args.command == "metrics":
        _cmd_metrics(args)
    elif args.command == "correlation":
        _cmd_correlation(args)
    elif args.command == "fairness":
        _cmd_fairness(args)
    elif args.command == "timeline":
        _cmd_timeline(args)
    elif args.command == "drift":
        _cmd_drift(args)
    elif args.command == "root-cause":
        _cmd_root_cause(args)
    elif args.command == "tail":
        _cmd_tail(args)
    elif args.command == "convergence":
        from xpyd_plan.cli._convergence import _cmd_convergence
        _cmd_convergence(args)
    elif args.command == "load-profile":
        from xpyd_plan.cli._load_profile import _cmd_load_profile
        _cmd_load_profile(args)
    elif args.command == "summary":
        _cmd_summary(args)
    elif args.command == "outlier-impact":
        from xpyd_plan.cli._outlier_impact import _cmd_outlier_impact
        _cmd_outlier_impact(args)
    elif args.command == "scorecard":
        _cmd_scorecard(args)
    elif args.command == "discover":
        _cmd_discover(args)
    elif args.command == "heatmap":
        _cmd_heatmap(args)
    elif args.command == "health-check":
        _cmd_health_check(args)
    elif args.command == "warmup-filter":
        _cmd_warmup_filter(args)
    elif args.command == "arrival-pattern":
        _cmd_arrival_pattern(args)
    elif args.command == "plan-benchmarks":
        _cmd_plan_benchmarks(args)
    elif args.command == "threshold-advisor":
        _cmd_threshold_advisor(args)
    elif args.command == "forecast":
        from xpyd_plan.cli._forecast import _run_forecast

        _run_forecast(args)
    elif args.command == "sla-tier":
        from xpyd_plan.cli._sla_tier import _run_sla_tier

        _run_sla_tier(args)
    elif args.command == "decompose":
        _cmd_decompose(args)
    elif args.command == "plugins":
        from xpyd_plan.cli._plugins import _run_plugins

        _run_plugins(args)
    elif args.command == "token-efficiency":
        handle_token_efficiency(args)
    elif args.command == "roi":
        from xpyd_plan.cli._roi import _cmd_roi
        _cmd_roi(args)
    elif args.command == "regression":
        _cmd_regression(args)
    elif args.command == "cdf":
        _cmd_cdf(args)
    elif args.command == "concurrency-util":
        _cmd_concurrency_util(args)
    elif args.command == "reproducibility":
        _cmd_reproducibility(args)
    elif args.command == "size-distribution":
        _cmd_size_distribution(args)
    elif args.command == "jitter":
        from xpyd_plan.cli._jitter import _cmd_jitter
        _cmd_jitter(args)
    elif args.command == "cold-start":
        from xpyd_plan.cli._cold_start import _cmd_cold_start
        _cmd_cold_start(args)
    elif args.command == "dedup":
        from xpyd_plan.cli._dedup import _cmd_dedup
        _cmd_dedup(args)
    elif args.command == "spike":
        from xpyd_plan.cli._spike import _cmd_spike
        _cmd_spike(args)
    elif args.command == "cross-validate":
        from xpyd_plan.cli._cross_validate import _cmd_cross_validate
        _cmd_cross_validate(args)
    elif args.command == "parquet":
        from xpyd_plan.cli._parquet import _cmd_parquet
        _cmd_parquet(args)
    elif args.command == "qps-curve":
        from xpyd_plan.cli._qps_curve import _cmd_qps_curve
        _cmd_qps_curve(args)
    elif args.command == "scaling-policy":
        from xpyd_plan.cli._scaling_policy import _cmd_scaling_policy
        _cmd_scaling_policy(args)
    elif args.command == "session":
        _cmd_session(args)
    elif args.command == "fingerprint":
        _cmd_fingerprint(args)
    elif args.command == "budget-tracker":
        _cmd_budget_tracker(args)
    else:
        parser.print_help()
        sys.exit(1)

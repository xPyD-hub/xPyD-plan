"""CLI entry point for xpyd-plan."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.models import DatasetStats, GPUProfile, SLAConfig


def _load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def _load_dataset(path: str) -> list[dict[str, int]]:
    """Load dataset from JSON lines file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _print_cost_analysis(
    console: Console,
    analyzer: BenchmarkAnalyzer,
    sla: SLAConfig,
    total: int,
    args: argparse.Namespace,
) -> None:
    """Print cost-aware analysis output."""
    from xpyd_plan.cost import CostAnalyzer, CostConfig

    cost_config = CostConfig.from_yaml(args.cost_model)
    cost_analyzer = CostAnalyzer(cost_config)
    result = analyzer.find_optimal_ratio(total, sla)
    qps = analyzer.data.metadata.measured_qps

    comparison = cost_analyzer.compare(result, qps, args.budget_ceiling)

    console.print(f"\n[bold]💰 Cost Analysis ({cost_config.currency})[/bold]")
    console.print(f"   GPU hourly rate: {cost_config.currency} {cost_config.gpu_hourly_rate:.2f}")
    if args.budget_ceiling is not None:
        console.print(f"   Budget ceiling: {cost_config.currency} {args.budget_ceiling:.2f}/hr")

    if comparison.sla_optimal:
        s = comparison.sla_optimal
        console.print(
            f"\n   [bold]SLA-Optimal:[/bold] {s.ratio_str}"
            f"  — {s.currency} {s.hourly_cost:.2f}/hr"
        )
        if s.cost_per_request is not None:
            console.print(f"   Cost/request: {s.currency} {s.cost_per_request:.6f}")

    if comparison.cost_optimal:
        c = comparison.cost_optimal
        console.print(
            f"   [bold]Cost-Optimal:[/bold] {c.ratio_str}"
            f"  — {c.currency} {c.hourly_cost:.2f}/hr"
        )
        if c.cost_per_request is not None:
            console.print(f"   Cost/request: {c.currency} {c.cost_per_request:.6f}")

    if comparison.sla_optimal and comparison.cost_optimal:
        diff = comparison.sla_optimal.hourly_cost - comparison.cost_optimal.hourly_cost
        if abs(diff) > 0.01:
            console.print(
                f"\n   💡 Switching to cost-optimal saves"
                f" {comparison.sla_optimal.currency} {diff:.2f}/hr"
            )
        else:
            console.print("\n   ✅ SLA-optimal and cost-optimal are the same!")

    # Cost table
    if comparison.all_costs:
        table = Table(title="\nCost Breakdown by P:D Ratio")
        table.add_column("Config", style="cyan")
        table.add_column("Instances", justify="right")
        table.add_column("Hourly Cost", justify="right")
        table.add_column("Cost/Request", justify="right")
        table.add_column("SLA", justify="center")

        for cost in comparison.all_costs:
            sla_str = "✅" if cost.meets_sla else "❌"
            cpr = f"{cost.currency} {cost.cost_per_request:.6f}" if cost.cost_per_request else "N/A"
            table.add_row(
                cost.ratio_str,
                str(cost.total_instances),
                f"{cost.currency} {cost.hourly_cost:.2f}",
                cpr,
                sla_str,
            )
        console.print(table)


def _print_single_analysis(
    console: Console, analyzer: BenchmarkAnalyzer, sla: SLAConfig, total: int, top: int
) -> dict:
    """Print single-scenario analysis and return the result dict."""
    # Show current config analysis
    console.print("\n[bold]📊 Current Configuration Analysis[/bold]")
    meta = analyzer.data.metadata
    console.print(
        f"   Config: {meta.num_prefill_instances}P:{meta.num_decode_instances}D"
        f"  (total: {meta.total_instances} instances, QPS: {meta.measured_qps:.1f})"
    )

    current_sla = analyzer.check_sla(sla)
    sla_status = "[green]✅ PASS[/green]" if current_sla.meets_all else "[red]❌ FAIL[/red]"
    console.print(f"   SLA: {sla_status}")
    console.print(
        f"   TTFT P95: {current_sla.ttft_p95_ms:.1f}ms  P99: {current_sla.ttft_p99_ms:.1f}ms"
    )
    console.print(
        f"   TPOT P95: {current_sla.tpot_p95_ms:.1f}ms  P99: {current_sla.tpot_p99_ms:.1f}ms"
    )

    current_util = analyzer.compute_utilization()
    console.print(
        f"   Utilization: P={current_util.prefill_utilization:.1%}"
        f"  D={current_util.decode_utilization:.1%}"
        f"  Waste={current_util.waste_rate:.1%}"
    )

    result = analyzer.find_optimal_ratio(total, sla)

    console.print(f"\n[bold]🔍 Optimal P:D Ratio Search (total={total} instances)[/bold]")

    if result.best:
        b = result.best
        console.print(
            f"\n[bold green]✅ Recommended: {b.ratio_str}[/bold green]"
            f"  (waste: {b.waste_rate:.1%})"
        )
        console.print(
            f"   P util: {b.prefill_utilization:.1%}"
            f"  D util: {b.decode_utilization:.1%}"
        )
        if b.sla_check:
            console.print(
                f"   TTFT P95: {b.sla_check.ttft_p95_ms:.1f}ms"
                f"  TPOT P95: {b.sla_check.tpot_p95_ms:.1f}ms"
            )
    else:
        console.print(
            "\n[bold red]❌ No P:D ratio meets SLA constraints"
            f" with {total} instances.[/bold red]"
        )

    # Candidates table
    table = Table(title=f"\nTop {top} Candidates")
    table.add_column("Config", style="cyan")
    table.add_column("P Util", justify="right")
    table.add_column("D Util", justify="right")
    table.add_column("Waste", justify="right")
    table.add_column("TTFT P95", justify="right")
    table.add_column("TPOT P95", justify="right")
    table.add_column("SLA", justify="center")

    for c in result.candidates[:top]:
        sla_str = "✅" if c.meets_sla else "❌"
        ttft_str = f"{c.sla_check.ttft_p95_ms:.1f}" if c.sla_check else "N/A"
        tpot_str = f"{c.sla_check.tpot_p95_ms:.1f}" if c.sla_check else "N/A"
        table.add_row(
            c.ratio_str,
            f"{c.prefill_utilization:.1%}",
            f"{c.decode_utilization:.1%}",
            f"{c.waste_rate:.1%}",
            ttft_str,
            tpot_str,
            sla_str,
        )

    console.print(table)
    return result.model_dump()


def _print_sensitivity(
    console: Console, analyzer: BenchmarkAnalyzer, sla: SLAConfig, total: int
) -> None:
    """Print sensitivity analysis output."""
    from xpyd_plan.sensitivity import analyze_sensitivity

    sens = analyze_sensitivity(analyzer, total, sla)

    table = Table(title="\n📈 Sensitivity Analysis — P:D vs SLA Margin")
    table.add_column("Config", style="cyan")
    table.add_column("SLA", justify="center")
    table.add_column("Waste", justify="right")
    if sla.ttft_ms is not None:
        table.add_column("TTFT Margin", justify="right")
    if sla.tpot_ms is not None:
        table.add_column("TPOT Margin", justify="right")
    if sla.max_latency_ms is not None:
        table.add_column("Latency Margin", justify="right")

    for p in sens.points:
        sla_str = "[green]✅[/green]" if p.meets_sla else "[red]❌[/red]"
        row = [p.ratio_str, sla_str, f"{p.waste_rate:.1%}"]
        if sla.ttft_ms is not None and p.ttft_margin_pct is not None:
            color = "green" if p.ttft_margin_pct >= 0 else "red"
            row.append(f"[{color}]{p.ttft_margin_pct:+.1%}[/{color}]")
        if sla.tpot_ms is not None and p.tpot_margin_pct is not None:
            color = "green" if p.tpot_margin_pct >= 0 else "red"
            row.append(f"[{color}]{p.tpot_margin_pct:+.1%}[/{color}]")
        if sla.max_latency_ms is not None and p.total_latency_margin_pct is not None:
            color = "green" if p.total_latency_margin_pct >= 0 else "red"
            row.append(f"[{color}]{p.total_latency_margin_pct:+.1%}[/{color}]")
        table.add_row(*row)

    console.print(table)

    if sens.cliffs:
        console.print("\n[bold yellow]⚠️  SLA Cliff Points[/bold yellow]")
        for cliff in sens.cliffs:
            if cliff.last_pass and cliff.first_fail:
                console.print(
                    f"   {cliff.last_pass.ratio_str} → {cliff.first_fail.ratio_str}"
                    f"  ({cliff.direction}, fails on: {cliff.failing_metric})"
                )

    if sens.recommendation and sens.recommendation.recommended:
        rec = sens.recommendation
        console.print("\n[bold]🛡️  Safety Recommendation[/bold]")
        console.print(
            f"   Recommended: [bold green]{rec.recommended.ratio_str}[/bold green]"
            f"  (cliff distance: {rec.cliff_distance} steps)"
        )
        if rec.min_margin_pct is not None:
            console.print(f"   Min SLA margin: {rec.min_margin_pct:.1%}")
        if rec.optimal and rec.recommended.num_prefill != rec.optimal.num_prefill:
            console.print(
                f"   (Optimal by waste: {rec.optimal.ratio_str},"
                f" but too close to cliff)"
            )


def _cmd_analyze(args: argparse.Namespace) -> None:
    """Handle the 'analyze' subcommand."""
    console = Console()

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
    )

    # Handle streaming mode
    if args.stream:
        from xpyd_plan.streaming import stream_from_stdin

        console.print("[bold]📡 Streaming mode — reading JSONL from stdin...[/bold]")
        snapshot_interval = args.snapshot_interval or 10
        stream_from_stdin(sla=sla, snapshot_interval=snapshot_interval)
        return

    if not args.benchmark:
        console.print("[red]Error: --benchmark is required unless --stream is used.[/red]")
        sys.exit(1)

    analyzer = BenchmarkAnalyzer()
    benchmarks = args.benchmark
    fmt = args.format

    # Load with format detection
    if fmt == "auto":
        load_fn = load_benchmark_auto
    elif fmt == "xpyd-bench":
        from xpyd_plan.bench_adapter import XpydBenchAdapter

        adapter = XpydBenchAdapter()
        load_fn = adapter.load
    else:
        # native — use analyzer's built-in loader
        load_fn = None

    if len(benchmarks) == 1:
        # Single-file mode (original behavior)
        if load_fn:
            data = load_fn(benchmarks[0])
            analyzer._data = data
        else:
            analyzer.load_data(benchmarks[0])
        total = args.total_instances or analyzer.data.metadata.total_instances
        _print_single_analysis(console, analyzer, sla, total, args.top)

        if args.cost_model:
            _print_cost_analysis(console, analyzer, sla, total, args)

        if args.sensitivity:
            _print_sensitivity(console, analyzer, sla, total)

        if args.report:
            from xpyd_plan.report import ReportGenerator

            result = analyzer.find_optimal_ratio(total, sla)
            gen = ReportGenerator()
            html = gen.generate_single(result, total_instances=total)
            gen.write(html, args.report)
            console.print(f"\n[dim]HTML report written to {args.report}[/dim]")

        if args.output:
            result = analyzer.find_optimal_ratio(total, sla)
            Path(args.output).write_text(result.model_dump_json(indent=2))
            console.print(f"\n[dim]Result written to {args.output}[/dim]")

        # Machine-readable output format
        if args.output_format in ("json", "csv"):
            from xpyd_plan.export import result_to_csv, result_to_json

            result = analyzer.find_optimal_ratio(total, sla)
            if args.output_format == "json":
                print(result_to_json(result))
            else:
                print(result_to_csv(result), end="")
    else:
        # Multi-file mode
        if load_fn:
            datasets = [load_fn(b) for b in benchmarks]
            datasets.sort(key=lambda d: d.metadata.measured_qps)
            analyzer._multi_data = datasets
            analyzer._data = datasets[0]
        else:
            analyzer.load_multi_data(benchmarks)
        total = args.total_instances or analyzer.multi_data[0].metadata.total_instances

        console.print(f"\n[bold]📊 Multi-Scenario Analysis ({len(benchmarks)} scenarios)[/bold]")

        multi_result = analyzer.find_optimal_ratio_multi(total, sla)

        # Per-scenario summary table
        summary_table = Table(title="\nPer-Scenario Summary")
        summary_table.add_column("QPS", justify="right")
        summary_table.add_column("Config", style="cyan")
        summary_table.add_column("Best P:D", style="green")
        summary_table.add_column("Waste", justify="right")
        summary_table.add_column("SLA", justify="center")

        for scenario in multi_result.scenarios:
            sla_check = scenario.analysis.current_sla_check
            best = scenario.analysis.best
            sla_str = "✅" if sla_check and sla_check.meets_all else "❌"
            best_str = best.ratio_str if best else "N/A"
            waste_str = f"{best.waste_rate:.1%}" if best else "N/A"
            summary_table.add_row(
                f"{scenario.qps:.1f}",
                f"{scenario.analysis.total_instances}",
                best_str,
                waste_str,
                sla_str,
            )

        console.print(summary_table)

        # Unified recommendation
        console.print(
            f"\n[bold]🎯 Unified Recommendation (total={total} instances)[/bold]"
        )
        if multi_result.unified_best:
            u = multi_result.unified_best
            console.print(
                f"\n[bold green]✅ Unified Best: {u.ratio_str}[/bold green]"
                f"  (worst-case waste: {u.waste_rate:.1%})"
            )
        else:
            console.print(
                "\n[bold red]❌ No single P:D ratio meets SLA"
                " across all scenarios.[/bold red]"
            )
            console.print("   Consider per-scenario scaling:")
            for i, scenario in enumerate(multi_result.scenarios):
                best = scenario.analysis.best
                if best:
                    console.print(f"   QPS {scenario.qps:.1f}: {best.ratio_str}")
                else:
                    console.print(f"   QPS {scenario.qps:.1f}: No ratio meets SLA")

        if args.sensitivity:
            for dataset in analyzer.multi_data:
                analyzer._data = dataset
                meta = dataset.metadata
                console.print(
                    f"\n[bold]📈 Sensitivity Analysis"
                    f" (QPS={meta.measured_qps:.1f})[/bold]"
                )
                _print_sensitivity(console, analyzer, sla, total)
            analyzer._data = analyzer.multi_data[0]

        if args.report:
            from xpyd_plan.report import ReportGenerator

            gen = ReportGenerator()
            html = gen.generate_multi(multi_result)
            gen.write(html, args.report)
            console.print(f"\n[dim]HTML report written to {args.report}[/dim]")

        if args.output:
            Path(args.output).write_text(multi_result.model_dump_json(indent=2))
            console.print(f"\n[dim]Result written to {args.output}[/dim]")

        # Machine-readable output format
        if args.output_format in ("json", "csv"):
            from xpyd_plan.export import result_to_csv, result_to_json

            if args.output_format == "json":
                print(result_to_json(multi_result))
            else:
                print(result_to_csv(multi_result), end="")


def _cmd_plan_legacy(args: argparse.Namespace) -> None:
    """Handle the legacy 'plan' subcommand (deprecated)."""
    warnings.warn(
        "The 'plan' subcommand is deprecated. Use 'analyze' with benchmark data instead.",
        DeprecationWarning,
        stacklevel=1,
    )

    from xpyd_plan.planner import plan

    cfg = _load_config(args.config)
    sla = SLAConfig(**(cfg.get("sla") or {}))
    gpu = GPUProfile(**(cfg["gpu"]))
    budget: int = cfg["budget"]

    if args.dataset:
        records = _load_dataset(args.dataset)
        num_requests = cfg.get("num_requests", 1000)
        dataset = DatasetStats.from_records(records, num_requests=num_requests)
    elif "dataset" in cfg:
        dataset = DatasetStats(**cfg["dataset"])
    else:
        print("Error: provide --dataset or include 'dataset' section in config", file=sys.stderr)
        sys.exit(1)

    result = plan(sla=sla, dataset=dataset, gpu=gpu, budget=budget)

    console = Console()
    if result.best:
        console.print(
            f"\n[bold green]✅ Best: {result.best.config.ratio_str}[/bold green]"
            f"  (score: {result.best.score:.2f})"
        )
    else:
        console.print("\n[bold red]❌ No configuration meets SLA constraints.[/bold red]")

    table = Table(title=f"\nTop {args.top} Candidates (budget={budget} GPUs)")
    table.add_column("Config", style="cyan")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("TPOT (ms)", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Cost/h", justify="right")
    table.add_column("SLA", justify="center")
    table.add_column("Score", justify="right")

    for c in result.candidates[: args.top]:
        sla_str = "✅" if c.meets_sla else "❌"
        table.add_row(
            c.config.ratio_str,
            f"{c.performance.ttft_ms:.1f}",
            f"{c.performance.tpot_ms:.1f}",
            f"{c.performance.throughput_rps:.1f}",
            f"${c.performance.total_cost_per_hour:.2f}",
            sla_str,
            f"{c.score:.2f}",
        )

    console.print(table)

    if args.output:
        Path(args.output).write_text(result.model_dump_json(indent=2))
        console.print(f"\n[dim]Result written to {args.output}[/dim]")


def _cmd_export(args: argparse.Namespace) -> None:
    """Handle the 'export' subcommand for batch export."""
    from xpyd_plan.export import export_batch

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
    )
    output = export_batch(
        benchmark_dir=args.dir,
        sla_config=sla,
        output_format=args.output_format,
        total_instances=args.total_instances,
    )
    print(output, end="" if args.output_format == "csv" else "\n")


def _cmd_plan_capacity(args: argparse.Namespace) -> None:
    """Handle the 'plan-capacity' subcommand."""
    from xpyd_plan.bench_adapter import load_benchmark_auto
    from xpyd_plan.capacity import CapacityPlanner

    console = Console()

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
    )

    datasets = [load_benchmark_auto(b) for b in args.benchmark]
    planner = CapacityPlanner()
    planner.fit(datasets, sla=sla)
    rec = planner.recommend(
        target_qps=args.target_qps,
        sla=sla,
        max_instances=args.max_instances,
    )

    if args.output_format == "json":
        print(rec.model_dump_json(indent=2))
        return

    console.print(f"\n[bold]📐 Capacity Planning — Target QPS: {rec.target_qps:.1f}[/bold]")
    confidence_color = {"high": "green", "medium": "yellow", "low": "red"}[rec.confidence.value]
    console.print(
        f"\n   [bold {confidence_color}]Recommendation: {rec.recommended_ratio}"
        f" ({rec.recommended_instances} instances)[/bold {confidence_color}]"
    )
    console.print(f"   Confidence: [{confidence_color}]{rec.confidence.value}[/{confidence_color}]")
    console.print(f"   Estimated headroom: {rec.estimated_headroom_pct:+.1f}%")

    if rec.notes:
        for note in rec.notes:
            console.print(f"   ⚠️  {note}")

    # Scaling data table
    table = Table(title="\nScaling Data Points")
    table.add_column("Instances", justify="right")
    table.add_column("Config", style="cyan")
    table.add_column("QPS", justify="right")
    table.add_column("Meets SLA", justify="center")
    table.add_column("TTFT P95", justify="right")
    table.add_column("TPOT P95", justify="right")

    for p in rec.scaling_points:
        sla_str = "✅" if p.max_qps_meeting_sla is not None else "❌"
        table.add_row(
            str(p.total_instances),
            f"{p.num_prefill}P:{p.num_decode}D",
            f"{p.measured_qps:.1f}",
            sla_str,
            f"{p.ttft_p95_ms:.1f}",
            f"{p.tpot_p95_ms:.1f}",
        )

    console.print(table)


def main(argv: list[str] | None = None) -> None:
    """Entry point for `xpyd-plan` command."""
    parser = argparse.ArgumentParser(
        prog="xpyd-plan",
        description="Analyze benchmark data to find optimal Prefill:Decode instance ratio",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze subcommand (primary)
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze benchmark data to find optimal P:D ratio",
    )
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
        help="Generate HTML report to the given path (e.g. report.html)",
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

    # plan subcommand (legacy, deprecated)
    plan_parser = subparsers.add_parser(
        "plan",
        help="[DEPRECATED] Estimate-based planning (use 'analyze' instead)",
    )
    plan_parser.add_argument("--config", type=str, required=True, help="YAML config path")
    plan_parser.add_argument("--dataset", type=str, default=None, help="Dataset JSON lines file")
    plan_parser.add_argument("--top", type=int, default=5, help="Top N candidates")
    plan_parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    # export subcommand
    export_parser = subparsers.add_parser(
        "export",
        help="Batch export benchmark analysis results from a directory",
    )
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
        "--total-instances", type=int, default=None,
        help="Total instances to optimize for",
    )

    # plan-capacity subcommand
    cap_parser = subparsers.add_parser(
        "plan-capacity",
        help="Recommend minimum instances and P:D ratio for a target QPS",
    )
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
        "--max-instances", type=int, default=64,
        help="Maximum instances to consider (default: 64)",
    )
    cap_parser.add_argument(
        "--output-format", type=str, choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )

    args = parser.parse_args(argv)

    if args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "plan":
        _cmd_plan_legacy(args)
    elif args.command == "export":
        _cmd_export(args)
    elif args.command == "plan-capacity":
        _cmd_plan_capacity(args)
    else:
        parser.print_help()
        sys.exit(1)

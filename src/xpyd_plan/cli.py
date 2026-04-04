"""CLI entry point for xpyd-plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.models import SLAConfig


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
        sla_percentile=args.sla_percentile,
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

        # Validate and filter outliers if requested
        if getattr(args, "validate", False):
            from xpyd_plan.validator import DataValidator, OutlierMethod

            method = OutlierMethod(getattr(args, "outlier_method", "iqr"))
            validator = DataValidator(method=method)
            vr = validator.validate(analyzer.data, filter_outliers=True)
            console.print(
                f"\n[bold]🔍 Validation:[/bold] {vr.outlier_count}/{vr.total_requests} "
                f"outliers detected (method={vr.method.value}, "
                f"quality={vr.quality.overall:.2f})"
            )
            if vr.filtered_data is not None:
                analyzer._data = vr.filtered_data
                console.print(
                    f"[dim]   Filtered to {len(vr.filtered_data.requests)} clean requests[/dim]"
                )

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


def _cmd_export(args: argparse.Namespace) -> None:
    """Handle the 'export' subcommand for batch export."""
    from xpyd_plan.export import export_batch

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
        sla_percentile=args.sla_percentile,
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
        sla_percentile=args.sla_percentile,
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


def _cmd_what_if(args: argparse.Namespace) -> None:
    """Handle the 'what-if' subcommand."""
    from xpyd_plan.bench_adapter import load_benchmark_auto
    from xpyd_plan.whatif import WhatIfSimulator

    console = Console()

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
        sla_percentile=args.sla_percentile,
    )

    data = load_benchmark_auto(args.benchmark)
    sim = WhatIfSimulator()
    sim.load(data)

    # Build scenario specs from CLI args
    scenarios: list[dict] = []
    if args.scale_qps:
        for part in args.scale_qps.split(","):
            part = part.strip().rstrip("xX")
            scenarios.append({"scale_qps": float(part)})

    if args.add_instances is not None:
        if scenarios:
            # Combine with each QPS scenario
            combined = []
            for s in scenarios:
                combined.append({**s, "add_instances": args.add_instances})
            # Also add instance-only scenario
            combined.append({"add_instances": args.add_instances})
            scenarios = combined
        else:
            scenarios.append({"add_instances": args.add_instances})

    if not scenarios:
        console.print("[red]Error: specify --scale-qps and/or --add-instances[/red]")
        sys.exit(1)

    comparison = sim.compare(scenarios, sla)

    if args.output_format == "json":
        print(comparison.model_dump_json(indent=2))
        return

    # Rich table output
    meta = data.metadata
    console.print("\n[bold]🔮 What-If Analysis[/bold]")
    console.print(
        f"   Baseline: {meta.num_prefill_instances}P:{meta.num_decode_instances}D"
        f"  ({meta.total_instances} instances, QPS: {meta.measured_qps:.1f})"
    )

    table = Table(title="\nScenario Comparison")
    table.add_column("Scenario", style="cyan")
    table.add_column("Instances", justify="right")
    table.add_column("Best P:D", style="green")
    table.add_column("Waste", justify="right")
    table.add_column("TTFT P95", justify="right")
    table.add_column("TPOT P95", justify="right")
    table.add_column("SLA", justify="center")

    # Baseline row
    b = comparison.baseline
    if b.best and b.best.sla_check:
        table.add_row(
            "[bold]Baseline[/bold]",
            str(b.total_instances),
            b.best.ratio_str,
            f"{b.best.waste_rate:.1%}",
            f"{b.best.sla_check.ttft_p95_ms:.1f}",
            f"{b.best.sla_check.tpot_p95_ms:.1f}",
            "✅" if b.best.meets_sla else "❌",
        )
    else:
        table.add_row(
            "[bold]Baseline[/bold]",
            str(b.total_instances),
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "❌",
        )

    # Scenario rows
    for s in comparison.scenarios:
        if s.best and s.best.sla_check:
            table.add_row(
                s.label,
                str(s.total_instances),
                s.best.ratio_str,
                f"{s.best.waste_rate:.1%}",
                f"{s.best.sla_check.ttft_p95_ms:.1f}",
                f"{s.best.sla_check.tpot_p95_ms:.1f}",
                "✅" if s.best.meets_sla else "❌",
            )
        else:
            table.add_row(
                s.label,
                str(s.total_instances),
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "❌",
            )

    console.print(table)


def _add_config_flag(parser: argparse.ArgumentParser) -> None:
    """Add --config flag to a subparser."""
    parser.add_argument(
        "--config", type=str, default=None, dest="config_file",
        help="Path to YAML config profile (default: auto-detect xpyd-plan.yaml)",
    )


def _apply_config_defaults(args: argparse.Namespace) -> None:
    """Apply config profile defaults to args where CLI didn't set a value."""
    from xpyd_plan.config import load_config

    config_path = getattr(args, "config_file", None)
    cfg = load_config(config_path)

    # SLA defaults
    if getattr(args, "sla_ttft", None) is None and cfg.sla.ttft_ms is not None:
        args.sla_ttft = cfg.sla.ttft_ms
    if getattr(args, "sla_tpot", None) is None and cfg.sla.tpot_ms is not None:
        args.sla_tpot = cfg.sla.tpot_ms
    if getattr(args, "sla_max_latency", None) is None and cfg.sla.max_latency_ms is not None:
        args.sla_max_latency = cfg.sla.max_latency_ms
    if getattr(args, "sla_percentile", None) == 95.0 and cfg.sla.percentile is not None:
        args.sla_percentile = cfg.sla.percentile

    # Output defaults
    if hasattr(args, "output_format") and args.output_format == "table":
        args.output_format = cfg.output.format
    if hasattr(args, "top") and args.top == 5:
        args.top = cfg.output.top

    # Defaults
    if getattr(args, "total_instances", None) is None and cfg.defaults.total_instances is not None:
        args.total_instances = cfg.defaults.total_instances
    if hasattr(args, "format") and args.format == "auto":
        args.format = cfg.defaults.benchmark_format

    # Cost defaults
    if getattr(args, "budget_ceiling", None) is None and cfg.cost.budget_ceiling is not None:
        args.budget_ceiling = cfg.cost.budget_ceiling

    # Store resolved config for potential use
    args._resolved_config = cfg


def _cmd_config(args: argparse.Namespace) -> None:
    """Handle the 'config' subcommand."""
    from xpyd_plan.config import init_config, load_config

    console = Console()

    if args.config_action == "init":
        path = init_config(args.output_path)
        console.print(f"[green]✅ Config file created: {path}[/green]")
    elif args.config_action == "show":
        config_path = getattr(args, "config_file", None)
        cfg = load_config(config_path)
        console.print("[bold]Resolved configuration:[/bold]\n")
        console.print(cfg.to_yaml())
    else:
        console.print("[red]Error: specify 'init' or 'show'[/red]")
        sys.exit(1)


def _cmd_validate(args: argparse.Namespace) -> None:
    """Handle the 'validate' subcommand."""

    from xpyd_plan.bench_adapter import load_benchmark_auto
    from xpyd_plan.validator import DataValidator, OutlierMethod

    console = Console()

    if not args.benchmark:
        console.print("[red]Error: --benchmark is required.[/red]")
        sys.exit(1)

    data = load_benchmark_auto(args.benchmark)
    method = OutlierMethod(args.outlier_method)
    validator = DataValidator(method=method)
    result = validator.validate(data, filter_outliers=True)

    output_format = getattr(args, "output_format", "table")

    if output_format == "json":
        console.print(result.model_dump_json(indent=2, exclude={"filtered_data"}))
        return

    # Table output
    console.print("\n[bold]📊 Benchmark Data Validation[/bold]")
    console.print(f"   File: {args.benchmark}")
    console.print(f"   Method: {result.method.value}")
    console.print(f"   Total requests: {result.total_requests}")
    console.print(f"   Outliers: {result.outlier_count}")
    console.print()

    # Quality scores
    q = result.quality
    quality_table = Table(title="Data Quality Scores")
    quality_table.add_column("Metric", style="cyan")
    quality_table.add_column("Score", justify="right")
    quality_table.add_row("Completeness", f"{q.completeness:.4f}")
    quality_table.add_row("Consistency", f"{q.consistency:.4f}")
    quality_table.add_row("Outlier Ratio", f"{q.outlier_ratio:.4f}")
    quality_table.add_row("Overall", f"[bold]{q.overall:.4f}[/bold]")
    console.print(quality_table)

    # Outlier details
    if result.outliers:
        console.print()
        outlier_table = Table(title=f"Outliers ({result.outlier_count} unique requests)")
        outlier_table.add_column("Request ID", style="cyan")
        outlier_table.add_column("Index", justify="right")
        outlier_table.add_column("Metric")
        outlier_table.add_column("Value", justify="right")
        outlier_table.add_column("Reason")
        for o in result.outliers[:50]:  # Cap display at 50
            outlier_table.add_row(o.request_id, str(o.index), o.metric, f"{o.value:.2f}", o.reason)
        console.print(outlier_table)
        if len(result.outliers) > 50:
            console.print(f"[dim]   ... and {len(result.outliers) - 50} more[/dim]")


def _cmd_trend(args: argparse.Namespace) -> None:
    """Handle the 'trend' subcommand."""
    from xpyd_plan.bench_adapter import load_benchmark_auto
    from xpyd_plan.trend import TrendTracker

    console = Console()
    db_path = args.db or "xpyd-plan-trend.db"
    tracker = TrendTracker(db_path=db_path)

    try:
        if args.trend_action == "add":
            data = load_benchmark_auto(args.benchmark)
            entry = tracker.add(data, label=args.label)
            if args.output_format == "json":
                print(entry.model_dump_json(indent=2))
            else:
                console.print(
                    f"[green]✅ Added entry #{entry.id}:[/green] "
                    f"{entry.label} (QPS={entry.measured_qps:.1f}, "
                    f"{entry.num_requests} requests)"
                )

        elif args.trend_action == "show":
            entries = tracker.list_entries(limit=args.limit)
            if args.output_format == "json":
                import json

                print(json.dumps([e.model_dump() for e in entries], indent=2))
                return

            if not entries:
                console.print("[dim]No trend entries found.[/dim]")
                return

            table = Table(title="Trend History")
            table.add_column("ID", justify="right")
            table.add_column("Label", style="cyan")
            table.add_column("QPS", justify="right")
            table.add_column("Config", style="green")
            table.add_column("TTFT P95", justify="right")
            table.add_column("TPOT P95", justify="right")
            table.add_column("Latency P95", justify="right")
            table.add_column("Requests", justify="right")

            for e in entries:
                table.add_row(
                    str(e.id),
                    e.label,
                    f"{e.measured_qps:.1f}",
                    f"{e.num_prefill}P:{e.num_decode}D",
                    f"{e.ttft_p95_ms:.1f}",
                    f"{e.tpot_p95_ms:.1f}",
                    f"{e.total_latency_p95_ms:.1f}",
                    str(e.num_requests),
                )
            console.print(table)

        elif args.trend_action == "check":
            report = tracker.check(
                lookback=args.lookback,
                threshold=args.threshold,
            )
            if args.output_format == "json":
                print(report.model_dump_json(indent=2))
                return

            console.print(f"\n[bold]📈 Trend Analysis ({report.lookback_count} entries)[/bold]")

            if report.lookback_count < 2:
                console.print("[dim]Not enough data for trend analysis (need >= 2 entries).[/dim]")
                return

            degrading = [a for a in report.alerts if a.is_degrading]
            if degrading:
                console.print(
                    f"\n[bold red]⚠️  {len(degrading)} metric(s) degrading![/bold red]"
                )
                table = Table(title="Degradation Alerts")
                table.add_column("Metric", style="cyan")
                table.add_column("Slope/Run", justify="right")
                table.add_column("Total Change", justify="right")

                for a in degrading:
                    table.add_row(
                        a.metric,
                        f"+{a.slope_per_run:.2f} ms",
                        f"[red]{a.total_change_pct:+.1%}[/red]",
                    )
                console.print(table)
            else:
                console.print("\n[bold green]✅ No degradation trends detected.[/bold green]")
        else:
            console.print("[red]Error: specify 'add', 'show', or 'check'[/red]")
            sys.exit(1)
    finally:
        tracker.close()


def _cmd_dashboard(args: argparse.Namespace) -> None:
    """Handle the 'dashboard' subcommand."""
    from xpyd_plan.benchmark_models import BenchmarkMetadata
    from xpyd_plan.dashboard import Dashboard
    from xpyd_plan.models import SLAConfig

    console = Console()

    if not args.benchmark and not args.stream:
        console.print("[red]Error: --benchmark or --stream is required.[/red]")
        sys.exit(1)

    sla = SLAConfig(
        ttft_ms=args.max_ttft_ms,
        tpot_ms=args.max_tpot_ms,
        max_latency_ms=args.max_total_latency_ms,
    )

    dashboard = Dashboard(
        sla=sla,
        total_instances=args.total_instances,
        refresh_interval=args.refresh_interval,
    )

    if args.benchmark:
        dashboard.load_file(args.benchmark)
        dashboard.run(console=console)
    else:
        # Streaming mode
        metadata = None
        if args.num_prefill and args.num_decode:
            total = args.total_instances or (args.num_prefill + args.num_decode)
            metadata = BenchmarkMetadata(
                num_prefill_instances=args.num_prefill,
                num_decode_instances=args.num_decode,
                total_instances=total,
                measured_qps=0.0,
            )
        dashboard.run(stream=sys.stdin, metadata=metadata, console=console)


def _cmd_interpolate(args: argparse.Namespace) -> None:
    """Run performance interpolation across P:D ratios."""
    import json as json_mod

    from xpyd_plan.interpolator import (
        InterpolationMethod,
        interpolate_performance,
    )

    console = Console()
    paths = args.benchmark
    if not paths or len(paths) < 2:
        console.print("[red]Need at least 2 benchmark files for interpolation.[/red]")
        sys.exit(1)

    from xpyd_plan.analyzer import BenchmarkAnalyzer
    analyzer = BenchmarkAnalyzer()
    datasets = analyzer.load_multi_data(paths)

    method = InterpolationMethod(args.method)

    target_ratios = None
    if args.ratios:
        target_ratios = []
        for r in args.ratios:
            parts = r.replace("P", "").replace("D", "").replace("p", "").replace("d", "").split(":")
            if len(parts) != 2:
                console.print(f"[red]Invalid ratio format '{r}'. Use 'P:D' e.g. '2:6'.[/red]")
                sys.exit(1)
            target_ratios.append((int(parts[0]), int(parts[1])))

    result = interpolate_performance(datasets, target_ratios=target_ratios, method=method)

    output_format = getattr(args, "output_format", "table")

    if output_format == "json":
        console.print(json_mod.dumps(result.model_dump(), indent=2))
        return

    # Table output
    table = Table(title="Performance Interpolation Results")
    table.add_column("P:D Ratio", style="cyan")
    table.add_column("TTFT P95 (ms)", justify="right")
    table.add_column("TPOT P95 (ms)", justify="right")
    table.add_column("Total Lat P95 (ms)", justify="right")
    table.add_column("QPS", justify="right")
    table.add_column("Confidence", justify="center")
    table.add_column("Measured", justify="center")

    for pred in result.predictions:
        conf_style = {
            "high": "green",
            "medium": "yellow",
            "low": "red",
        }.get(pred.confidence.value, "white")
        table.add_row(
            f"{pred.num_prefill}P:{pred.num_decode}D",
            f"{pred.ttft_p95_ms:.1f}",
            f"{pred.tpot_p95_ms:.1f}",
            f"{pred.total_latency_p95_ms:.1f}",
            f"{pred.throughput_qps:.1f}",
            f"[{conf_style}]{pred.confidence.value.upper()}[/{conf_style}]",
            "✓" if pred.is_measured else "",
        )

    console.print(table)

    if result.best_predicted:
        bp = result.best_predicted
        console.print(
            f"\n[bold green]Best predicted:[/bold green] "
            f"{bp.num_prefill}P:{bp.num_decode}D "
            f"(QPS={bp.throughput_qps:.1f}, confidence={bp.confidence.value})"
        )

    if result.notes:
        for note in result.notes:
            console.print(f"[dim]Note: {note}[/dim]")


def _cmd_compare(args: argparse.Namespace) -> None:
    """Execute the compare subcommand."""

    from xpyd_plan.comparator import compare_benchmarks

    console = Console()
    result = compare_benchmarks(args.baseline, args.current, threshold=args.threshold)

    if args.output_format == "json":
        console.print_json(result.model_dump_json(indent=2))
        return

    # Table output
    console.print("\n[bold]📊 Benchmark Comparison[/bold]")
    console.print(f"   Threshold: {result.threshold * 100:.0f}%")
    console.print(f"   Baseline QPS: {result.baseline_qps:.1f}")
    console.print(f"   Current QPS:  {result.current_qps:.1f}")

    qps = result.qps_delta
    qps_color = "red" if qps.is_regression else "green" if qps.relative_delta > 0 else "white"
    console.print(
        f"   QPS change: [{qps_color}]{qps.relative_delta:+.1%}[/{qps_color}]"
        f"{'  ⚠️  REGRESSION' if qps.is_regression else ''}"
    )

    table = Table(title="\nLatency Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Status", justify="center")

    for delta in result.latency_deltas:
        color = "red" if delta.is_regression else "green" if delta.relative_delta < 0 else "white"
        status = "⚠️  REGRESSED" if delta.is_regression else "✅"
        table.add_row(
            delta.metric,
            f"{delta.baseline:.1f}",
            f"{delta.current:.1f}",
            f"[{color}]{delta.relative_delta:+.1%}[/{color}]",
            status,
        )

    console.print(table)

    if result.has_regression:
        console.print(
            f"\n[bold red]⚠️  {result.regression_count} regression(s) detected![/bold red]"
        )
    else:
        console.print("\n[bold green]✅ No regressions detected.[/bold green]")


def _cmd_alert(args: argparse.Namespace) -> None:
    """Execute the alert subcommand."""
    import json as json_mod

    from xpyd_plan.alerting import AlertEngine
    from xpyd_plan.benchmark_models import BenchmarkData

    console = Console()

    # Load benchmark data
    with open(args.benchmark) as f:
        raw = json_mod.load(f)
    data = BenchmarkData(**raw)

    # Load rules and evaluate
    engine = AlertEngine.from_yaml(args.rules)
    report = engine.evaluate(data)

    output_format = getattr(args, "output_format", "table")

    if output_format == "json":
        console.print_json(report.model_dump_json(indent=2))
    else:
        # Table output
        table = Table(title="Alert Evaluation")
        table.add_column("Rule", style="bold")
        table.add_column("Metric")
        table.add_column("Actual")
        table.add_column("Threshold")
        table.add_column("Severity")
        table.add_column("Status")

        for result in report.results:
            severity_style = {
                "critical": "red bold",
                "warning": "yellow",
                "info": "blue",
            }.get(result.severity.value, "")
            status = "[red]TRIGGERED[/red]" if result.triggered else "[green]OK[/green]"
            table.add_row(
                result.rule_name,
                result.metric,
                f"{result.actual_value:.2f}",
                f"{result.threshold:.2f}",
                f"[{severity_style}]{result.severity.value.upper()}[/{severity_style}]",
                status,
            )

        console.print(table)
        console.print()
        if report.passed:
            console.print(f"[green bold]✓ {report.summary}[/green bold]")
        else:
            console.print(f"[red bold]✗ {report.summary}[/red bold]")

    if not report.passed:
        sys.exit(1)


def _parse_tags(tag_list: list[str]) -> dict[str, str]:
    """Parse KEY=VALUE tag arguments into a dict."""
    tags: dict[str, str] = {}
    for item in tag_list:
        if "=" not in item:
            Console().print(f"[red]Invalid tag format: {item} (expected KEY=VALUE)[/red]")
            sys.exit(1)
        key, value = item.split("=", 1)
        tags[key.strip()] = value.strip()
    return tags


def _cmd_annotate(args: argparse.Namespace) -> None:
    """Handle annotate subcommand."""
    from xpyd_plan.annotation import AnnotationManager

    manager = AnnotationManager()
    console = Console()

    if args.annotate_action == "add":
        tags = _parse_tags(args.tag)
        result = manager.add_tags(args.benchmark, tags)
        console.print(f"[green]Added {len(tags)} tag(s) to {args.benchmark}[/green]")
        for k, v in sorted(result.tags.items()):
            console.print(f"  {k} = {v}")

    elif args.annotate_action == "remove":
        result = manager.remove_tags(args.benchmark, args.key)
        console.print(f"[green]Removed tag(s) from {args.benchmark}[/green]")
        if result.tags:
            for k, v in sorted(result.tags.items()):
                console.print(f"  {k} = {v}")
        else:
            console.print("  (no tags remaining)")

    elif args.annotate_action == "clear":
        manager.clear_tags(args.benchmark)
        console.print(f"[green]Cleared all tags from {args.benchmark}[/green]")

    elif args.annotate_action == "list":
        result = manager.get_tags(args.benchmark)
        if getattr(args, "output_format", "table") == "json":
            import json as json_mod

            console.print(json_mod.dumps({"tags": result.tags}, indent=2))
        elif result.tags:
            table = Table(title=f"Tags: {args.benchmark}")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            for k, v in sorted(result.tags.items()):
                table.add_row(k, v)
            console.print(table)
        else:
            console.print(f"No tags on {args.benchmark}")

    elif args.annotate_action == "filter":
        tags = _parse_tags(args.tag)
        result = manager.filter_by_tags(args.dir, tags)
        if getattr(args, "output_format", "table") == "json":
            import json as json_mod

            console.print(
                json_mod.dumps(
                    {
                        "query_tags": result.query_tags,
                        "matched": [
                            {"path": m.benchmark_path, "tags": m.tags}
                            for m in result.matched
                        ],
                        "total_scanned": result.total_scanned,
                    },
                    indent=2,
                )
            )
        else:
            console.print(
                f"Scanned {result.total_scanned} file(s), "
                f"{len(result.matched)} matched"
            )
            if result.matched:
                table = Table(title="Matching Benchmarks")
                table.add_column("File", style="cyan")
                table.add_column("Tags", style="green")
                for m in result.matched:
                    tag_str = ", ".join(f"{k}={v}" for k, v in sorted(m.tags.items()))
                    table.add_row(m.benchmark_path, tag_str)
                console.print(table)

    else:
        console.print("[red]Usage: xpyd-plan annotate {add|list|remove|filter|clear}[/red]")
        sys.exit(1)


def _cmd_pareto(args: argparse.Namespace) -> None:
    """Handle 'pareto' subcommand."""
    import json as json_mod

    from xpyd_plan.bench_adapter import load_benchmark_auto
    from xpyd_plan.pareto import ParetoAnalyzer, ParetoObjective

    console = Console()

    # Load benchmark data
    data = load_benchmark_auto(args.benchmark[0])

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
        sla_percentile=args.sla_percentile,
    )
    total = args.total_instances or data.metadata.total_instances

    analyzer = BenchmarkAnalyzer(data)
    analysis = analyzer.find_optimal_ratio(total, sla)
    measured_qps = data.metadata.measured_qps

    # Cost config
    cost_config = None
    if args.cost_model:
        from xpyd_plan.cost import CostConfig

        cost_config = CostConfig.from_yaml(args.cost_model)

    # Parse objectives
    objectives = None
    if args.objectives:
        objectives = [ParetoObjective(o) for o in args.objectives]

    # Parse weights
    weights = None
    if args.weights:
        weights = {}
        for pair in args.weights.split(","):
            k, v = pair.strip().split("=")
            weights[k.strip()] = float(v.strip())

    pareto_analyzer = ParetoAnalyzer(cost_config=cost_config)
    frontier = pareto_analyzer.analyze(
        analysis,
        measured_qps=measured_qps,
        objectives=objectives,
        weights=weights,
    )

    if args.output_format == "json":
        console.print(json_mod.dumps(frontier.model_dump(), indent=2))
        return

    # Table output
    if not frontier.frontier:
        console.print("[yellow]No Pareto-optimal candidates found.[/yellow]")
        return

    count = len(frontier.frontier)
    console.print(f"\n[bold]🎯 Pareto Frontier ({count} optimal candidates)[/bold]")
    console.print(f"   Objectives: {', '.join(frontier.objectives_used)}")
    console.print(f"   Weights: {frontier.weights}")

    table = Table(title="Pareto-Optimal P:D Ratios")
    table.add_column("Ratio", style="cyan")
    table.add_column("Latency P95 (ms)", justify="right")
    if "cost" in frontier.objectives_used:
        table.add_column("Hourly Cost", justify="right")
    table.add_column("Waste Rate", justify="right")
    table.add_column("Score", justify="right", style="green")

    for c in frontier.frontier:
        row = [
            c.ratio_str,
            f"{c.latency_ms:.1f}",
        ]
        if "cost" in frontier.objectives_used:
            row.append(f"{c.hourly_cost:.2f}" if c.hourly_cost is not None else "N/A")
        row.extend([
            f"{c.waste_rate:.3f}",
            f"{c.weighted_score:.4f}" if c.weighted_score is not None else "N/A",
        ])
        table.add_row(*row)

    console.print(table)

    if frontier.best_weighted:
        console.print(
            f"\n   [bold green]✅ Best weighted:[/bold green] {frontier.best_weighted.ratio_str}"
            f" (score: {frontier.best_weighted.weighted_score:.4f})"
        )

    if args.include_dominated and frontier.dominated:
        console.print(f"\n[dim]Dominated candidates: {len(frontier.dominated)}[/dim]")
        for c in frontier.dominated:
            console.print(
                f"   [dim]{c.ratio_str}  latency={c.latency_ms:.1f}ms"
                f"  waste={c.waste_rate:.3f}[/dim]"
            )

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
    else:
        parser.print_help()
        sys.exit(1)

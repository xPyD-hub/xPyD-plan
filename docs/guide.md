# xPyD-plan User Guide

xPyD-plan is the benchmark data analysis toolkit for [xPyD-proxy](https://github.com/xPyD-hub/xPyD-proxy), designed to find the optimal **Prefill:Decode instance ratio** from real benchmark results.

> **Core principle:** No guessing, no modeling, no simulation — everything is based on actual benchmark data.

---

## Installation

```bash
# Basic installation
pip install xpyd-plan

# With HTML report generation
pip install "xpyd-plan[report]"

# Development environment
pip install "xpyd-plan[dev]"
```

---

## Core Subcommands

### Analysis

| Subcommand | Description |
|------------|-------------|
| `analyze` | SLA compliance check, utilization analysis, optimal P:D ratio search |
| `sensitivity` | P:D ratio vs. SLA satisfaction rate curve, with cliff detection |
| `confidence` | Bootstrap confidence intervals (latency percentiles) |
| `decompose` | Per-request latency decomposition into prefill/decode/overhead phases |
| `tail` | Extended percentile analysis (P99.9, P99.99) |

### Comparison & Testing

| Subcommand | Description |
|------------|-------------|
| `compare` | Two-benchmark comparison with regression detection |
| `ab-test` | Statistical A/B testing (Welch's t-test, Mann-Whitney U) |
| `model-compare` | Multi-model latency and cost-efficiency side-by-side comparison |
| `drift` | Distribution drift detection (Kolmogorov-Smirnov) |

### Planning & Optimization

| Subcommand | Description |
|------------|-------------|
| `recommend` | Ranked recommendations combining SLA, cost, Pareto, and trend data |
| `plan-capacity` | Capacity planning with linear scaling model |
| `what-if` | Scenario simulation — scale QPS or instance count and compare |
| `fleet` | Multi-GPU-type fleet sizing (with budget constraints) |
| `pareto` | Pareto frontier analysis across latency, cost, and waste |
| `interpolate` | Performance interpolation/extrapolation for untested P:D ratios |
| `forecast` | Capacity forecasting based on historical trends |
| `threshold-advisor` | SLA threshold tuning |

### Cost & Budget

| Subcommand | Description |
|------------|-------------|
| `budget` | SLA budget allocation across TTFT/TPOT phases |
| `scorecard` | Comprehensive efficiency scoring (SLA + utilization + waste) |
| `sla-tier` | Multi-SLA tier analysis |

### Data Management

| Subcommand | Description |
|------------|-------------|
| `validate` | Data quality scoring, outlier detection |
| `filter` | Filter and sample by token/latency/time window |
| `merge` | Merge multiple benchmark files |
| `discover` | Recursively scan directories for benchmark files |
| `generate` | Generate synthetic benchmark data (for testing) |
| `export` | Batch export analysis results (JSON/CSV/table) |

### Monitoring & Alerting

| Subcommand | Description |
|------------|-------------|
| `dashboard` | Rich TUI real-time dashboard |
| `alert` | YAML-defined alert rules with CI/CD-friendly exit codes |
| `trend` | Historical trend tracking (SQLite storage) |
| `metrics` | Prometheus/OpenMetrics format export |
| `timeline` | Time-window analysis with warmup detection |

---

## Typical Workflow

### Step 1: Collect Benchmark Data

Use [xpyd-bench](https://github.com/xPyD-hub/xPyD-bench) to run performance tests across different P:D ratio configurations:

```bash
# Run benchmarks under different P:D configurations
xpyd-bench run --config cluster-2p6d.yaml --output results/2p6d.json
xpyd-bench run --config cluster-3p5d.yaml --output results/3p5d.json
xpyd-bench run --config cluster-4p4d.yaml --output results/4p4d.json
```

### Step 2: Analyze the Optimal P:D Ratio

```bash
xpyd-plan analyze \
  --benchmark results/2p6d.json results/3p5d.json results/4p4d.json \
  --sla-ttft 200 --sla-tpot 50
```

This outputs the SLA compliance rate, utilization, and resource waste for each P:D configuration, highlighting the optimal ratio.

### Step 3: Get Deployment Recommendations

```bash
xpyd-plan recommend \
  --benchmark results/ \
  --sla-ttft 200 --sla-tpot 50 \
  --cost-model gpu-costs.yaml
```

`recommend` combines SLA compliance, cost efficiency, Pareto optimality, and trend data to produce ranked deployment recommendations.

### Step 4: Simulate Scale-Up Scenarios

```bash
# Assume QPS doubles — see which ratio can still hold up
xpyd-plan what-if \
  --benchmark results/3p5d.json \
  --scale-qps 2.0 \
  --sla-ttft 200 --sla-tpot 50
```

### Step 5: Export Results

```bash
# Export as JSON (for automation)
xpyd-plan export --benchmark results/ --format json --output plan-results.json

# Export as CSV (for spreadsheet analysis)
xpyd-plan export --benchmark results/ --format csv --output plan-results.csv

# Generate Markdown report (for PR/Wiki)
xpyd-plan report --format markdown --benchmark results/ --output report.md
```

---

## Interpreting Results

### P:D Ratio

The ratio of Prefill instances to Decode instances. For example, `2:6` means 2 Prefill instances + 6 Decode instances (8 total). Different ratios exhibit different performance characteristics for TTFT (time to first token) and TPOT (time per output token).

### Throughput

The measured throughput (QPS) for a given P:D configuration. Higher throughput means more requests processed on the same hardware.

### Cost

Per-request cost or total operating cost calculated based on GPU-hour rates. Provide GPU pricing via `--cost-model`.

### Pareto Frontier

The set of configurations that are not fully dominated by any other configuration across latency, cost, and resource waste dimensions. Points on the Pareto frontier represent "no free lunch" — improving one metric necessarily sacrifices another. The `pareto` subcommand visualizes this boundary to help with trade-off decisions.

### SLA Compliance

Whether TTFT and TPOT meet target thresholds at a given percentile (P95/P99). Compliance rate = fraction of requests meeting the SLA.

### Resource Waste

The gap between actual utilization and theoretical full capacity. The optimal ratio minimizes this waste while still meeting SLA requirements.

---

## More Information

- [README](../README.md) — Project overview and quick start
- [Design Principles](DESIGN_PRINCIPLES.md) — Architecture and design decisions
- [Development Loop](DEV_LOOP.md) — Development workflow guide
- [Roadmap](../ROADMAP.md) — Full milestone list

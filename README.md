# xPyD-plan

Benchmark-data analysis toolkit for [xPyD-proxy](https://github.com/xPyD-hub/xPyD-proxy) — find the optimal **Prefill:Decode instance ratio** from real benchmark results.

> **Core principle:** No guessing, no modeling, no simulation — everything is based on actual benchmark data.

## Install

```bash
# Base install
pip install xpyd-plan

# With HTML report generation
pip install "xpyd-plan[report]"

# Development
pip install "xpyd-plan[dev]"
```

## Quick Start

```bash
# Analyze a benchmark file — find the optimal P:D ratio
xpyd-plan analyze --benchmark results.json --sla-ttft 200 --sla-tpot 50

# Compare two benchmark runs
xpyd-plan compare --baseline baseline.json --current current.json

# Generate a Markdown report
xpyd-plan report --format markdown --benchmark results.json --output report.md

# Export results as JSON for automation
xpyd-plan analyze --benchmark results.json --output-format json
```

## Features

### Core Analysis

| Command | Description |
|---------|-------------|
| `analyze` | SLA compliance check, utilization analysis, optimal P:D ratio finder |
| `export` | Batch export analysis results as JSON/CSV/table |
| `confidence` | Bootstrap confidence intervals for latency percentiles |
| `sensitivity` | P:D ratio vs SLA satisfaction curves with cliff detection |
| `decompose` | Per-request latency decomposition into prefill/decode/overhead phases |
| `tail` | Extended percentile analysis (P99.9, P99.99) with tail classification |

### Comparison & Testing

| Command | Description |
|---------|-------------|
| `compare` | Benchmark comparison with regression detection |
| `ab-test` | Statistical A/B test analysis (Welch's t-test, Mann-Whitney U, effect size) |
| `drift` | Distribution drift detection via Kolmogorov-Smirnov test |
| `model-compare` | Multi-model side-by-side latency and cost-efficiency comparison |

### Planning & Optimization

| Command | Description |
|---------|-------------|
| `plan-capacity` | Capacity planning with linear scaling model and confidence levels |
| `plan-benchmarks` | Generate prioritized list of P:D ratios to benchmark |
| `what-if` | Scenario simulation — scale QPS or instances and compare |
| `fleet` | Multi-GPU-type fleet sizing with budget constraints |
| `pareto` | Pareto frontier analysis across latency, cost, and waste |
| `recommend` | Ranked recommendations combining SLA, cost, Pareto, and trend data |
| `interpolate` | Performance interpolation/extrapolation for untested P:D ratios |
| `threshold-advisor` | SLA threshold tuning — find thresholds for target pass rates |
| `forecast` | Capacity forecasting from historical trend data |

### Cost & Budgeting

| Command | Description |
|---------|-------------|
| `budget` | SLA budget allocation across TTFT/TPOT stages |
| `scorecard` | Composite efficiency score (SLA compliance + utilization + waste) |
| `sla-tier` | Multi-SLA tier analysis for different service levels |

### Data Management

| Command | Description |
|---------|-------------|
| `validate` | Data quality scoring, outlier detection (IQR/Z-score) |
| `filter` | Token/latency/time-window filtering and random sampling |
| `merge` | Merge multiple benchmark files (union/intersection strategies) |
| `annotate` | Tag benchmark files with metadata (non-destructive sidecar) |
| `discover` | Recursive directory scanning for benchmark files |
| `generate` | Generate synthetic benchmark data for testing |

### Monitoring & Alerting

| Command | Description |
|---------|-------------|
| `alert` | YAML-defined alert rules with CI/CD-friendly exit codes |
| `trend` | Historical trend tracking with SQLite storage |
| `saturation` | Saturation point detection across QPS levels |
| `metrics` | Prometheus/OpenMetrics text format export |
| `dashboard` | Real-time Rich TUI dashboard |
| `timeline` | Time-window analysis with warmup detection and trend regression |

### Workload Characterization

| Command | Description |
|---------|-------------|
| `workload` | Request clustering into categories (prefill-heavy, decode-heavy, etc.) |
| `correlation` | Pearson correlation between request characteristics and latency |
| `heatmap` | 2D latency heatmap (prompt × output tokens) with hotspot detection |
| `root-cause` | Anomaly root cause analysis — SLA-failing vs passing request comparison |
| `scaling` | Throughput scaling efficiency with knee-point detection |

### Reporting & Configuration

| Command | Description |
|---------|-------------|
| `report` | HTML report with inline SVG visualizations |
| `report --format markdown` | Markdown report for GitHub PRs/wikis |
| `config init` | Generate starter YAML config file |
| `config show` | Display resolved configuration |
| `pipeline` | YAML-defined multi-step batch pipeline runner |

## Benchmark Data Format

xPyD-plan consumes JSON benchmark files (native or [xpyd-bench](https://github.com/xPyD-hub/xPyD-bench) format) containing per-request measurements:

```json
{
  "metadata": {
    "num_prefill_instances": 2,
    "num_decode_instances": 6,
    "total_instances": 8,
    "measured_qps": 10.5
  },
  "requests": [
    {
      "request_id": "req-001",
      "prompt_tokens": 512,
      "output_tokens": 128,
      "ttft_ms": 45.2,
      "tpot_ms": 12.1,
      "total_latency_ms": 1593.0,
      "timestamp": 1700000000.0
    }
  ]
}
```

## Programmatic API

Every feature is available as a Python function:

```python
from xpyd_plan import analyze, compare_benchmarks, analyze_ab_test

# Analyze benchmark data
result = analyze("benchmark.json", sla_ttft_ms=200, sla_tpot_ms=50)

# Compare two runs
comparison = compare_benchmarks("baseline.json", "current.json")
```

## Configuration

Create `xpyd-plan.yaml` (or `~/.config/xpyd-plan/config.yaml`) for persistent defaults:

```bash
xpyd-plan config init    # generates a commented starter file
xpyd-plan config show    # shows resolved config
```

CLI flags always override config file values.

## How It Works

1. **Input**: Benchmark results from `xpyd-bench` (real measured data)
2. **Analyze**: Compute latency distributions, SLA compliance, utilization per P:D ratio
3. **Optimize**: Find the ratio with minimum resource waste while meeting SLA constraints
4. **Report**: Output tables, JSON/CSV, Markdown/HTML reports, or Prometheus metrics

## License

TBD

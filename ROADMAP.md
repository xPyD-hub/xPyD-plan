# xPyD-plan Roadmap

## Vision

Help users find the **optimal Prefill:Decode instance ratio** based on **real benchmark data** — satisfy SLA constraints with minimum resource waste.

> **Core principle:** We are a **measured-data analysis tool**, not a performance simulator.
> No guessing, no modeling, no simulation — everything is based on actual benchmark results.

---

## Milestones

### M1 ✅ Core Data Models + Brute-force Search

*Completed — PR #2*

- Pydantic data models (`SLAConfig`, `DatasetStats`, `GPUProfile`, `PDConfig`, etc.)
- Linear performance estimator (simple throughput-proportional model)
- Brute-force enumeration of all P:D splits
- CLI with YAML config input and Rich table output
- 17 tests

> ⚠️ Direction was off (estimator-based), but the project skeleton is reusable.

### M2 ✅ Queuing-Theory Estimator

*Completed — PR #3*

- M/M/c (Erlang-C) queuing model for latency estimation
- GPU profile library (A100-80G, H100-80G)
- Batching degradation model
- Backward-compatible CLI and config format

> ⚠️ Direction was off — this is a simulation model, not what we need.
> May be deprecated or removed in future milestones.

### M3 ✅ Core Refactor — Benchmark Data Analyzer

*Completed — PR #6*

- **Benchmark data format** — define JSON schema for xpyd-bench output
  - Per-request: request_id, prompt_tokens, output_tokens, ttft_ms, tpot_ms, total_latency_ms, timestamp
  - Cluster config: num_prefill_instances, num_decode_instances, total_instances
  - Measured QPS
- **BenchmarkAnalyzer** — load, validate, and analyze benchmark data
- **SLA compliance check** — based on measured latency distributions (P95/P99), not estimates
- **Utilization analysis** — compute P and D instance utilization from measured data
- **Optimal P:D finder** — enumerate ratios, find the one with minimum waste while meeting SLA
- **CLI rewrite** — support the new analysis workflow
- **Tests** — comprehensive test suite with fixture-generated benchmark datasets

### M4 ✅ Multi-Scenario Analysis

*Completed — PR #8*

- Support loading multiple benchmark files (different QPS levels)
- Per-scenario independent analysis with `find_optimal_ratio_multi()`
- Unified P:D ratio recommendation across all QPS scenarios (min worst-case waste)
- CLI `--benchmark` accepts multiple files, auto-detects multi-scenario mode
- 15 new tests

### M5 ✅ Sensitivity Analysis

*Completed — PR #11*

- P:D ratio vs SLA satisfaction rate curves with margin computation
- Cliff detection: identify pass→fail transitions with failing metric
- Safety-margin recommendations with cliff distance awareness
- CLI `--sensitivity` flag for analyze subcommand
- 26 new tests

### M6 ✅ xpyd-bench Integration

*Completed — PR #13*

- XpydBenchAdapter for direct ingestion of xpyd-bench output format
- StreamingAnalyzer for live analysis during benchmark execution
- Schema version auto-detection with clear error on unsupported versions
- CLI `--format auto|native|xpyd-bench` and `--stream` flags
- 20 new tests

### M7 ✅ Report Generation

*Completed — PR #15*

- HTML report generation with inline SVG visualizations
- Utilization heatmaps, latency distributions
- Comparison tables across different P:D configurations

### M8 ✅ Cost-Aware Optimization

*Completed — PR #17*

- `CostConfig` Pydantic model with GPU hourly rate and currency
- `CostAnalyzer` for cost-per-request and total hourly cost calculations
- Budget constraint filtering (exclude ratios above cost ceiling)
- Cost-optimal vs SLA-optimal ratio comparison
- CLI `--cost-model` flag accepting YAML cost config
- 22 new tests

### M9 ✅ JSON/CSV Export & Programmatic API

*Completed — PR #20*

- Machine-readable output: `--output-format json|csv|table` (default: table)
- JSON export with full analysis results, cost data, sensitivity data
- CSV export: one row per ratio candidate for spreadsheet integration
- Programmatic Python API: `analyze()` returns structured results without CLI
- `xpyd-plan export` subcommand for batch export of multiple benchmark sets
- 20 new tests

### M10 ✅ Remove Deprecated Legacy Estimator & Planner

*Completed — PR #26*

- Remove `estimator.py` (M/M/c queuing model) and `planner.py` (brute-force estimator-based planning)
- Remove legacy `plan` CLI subcommand
- Remove `PerformanceEstimate`, `CandidateResult`, `PlanResult` model classes
- Remove associated tests (`test_estimator.py`, `test_planner.py`, `test_queueing_estimator.py`)
- Clean up unused imports in `cli.py` and `__init__.py`
- 203 tests remain, all passing

### M10 ✅ Capacity Planning Mode

*Completed — PR #23*

- `CapacityPlanner` class with `fit()` and `recommend()` methods
- Linear scaling model: estimates QPS-per-instance from measured benchmarks
- Confidence levels: HIGH (interpolation), MEDIUM (slight extrapolation), LOW (far extrapolation)
- CLI `plan-capacity` subcommand with `--target-qps`, `--benchmark`, table/JSON output
- Programmatic API: `plan_capacity()` returns structured dict
- 20% headroom built into recommendations, `max_instances` cap
- 24 new tests

### M11 ✅ What-If Scenario Simulation

*Completed — PR #28*

- `WhatIfSimulator` class with `scale_qps()`, `scale_instances()`, `compare()` methods
- `WhatIfScenario` and `WhatIfComparison` Pydantic models
- CLI `what-if` subcommand with `--benchmark`, `--scale-qps`, `--add-instances`
- Side-by-side comparison table output, JSON output format support
- Programmatic API: `what_if()` function
- 20 new tests (223 total)

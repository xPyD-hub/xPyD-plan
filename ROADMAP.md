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

### M12 ✅ Configuration Profile Support

*Completed — PR #31*

- `ConfigProfile` Pydantic model with `SLAProfile`, `CostProfile`, `OutputProfile`, `DefaultsProfile`
- YAML config file loading with fallback chain: `--config` → `./xpyd-plan.yaml` → `~/.config/xpyd-plan/config.yaml`
- CLI flags override config file values
- `xpyd-plan config init` generates commented starter YAML
- `xpyd-plan config show` displays resolved configuration
- All subcommands (`analyze`, `export`, `plan-capacity`, `what-if`) respect config defaults
- Backward compatible — works exactly as before when no config file exists
- 31 new tests (254 total)

### M13 ✅ Configurable SLA Percentile Threshold

*Completed — PR #33*

- `SLAConfig.sla_percentile` field (default: 95.0, valid range: 1-100)
- `SLACheck.evaluated_percentile` reports which percentile was used for pass/fail
- `SLACheck.{ttft,tpot,total_latency}_evaluated_ms` — values at the evaluated percentile
- `check_sla()` and `_estimate_ratio_performance()` use the configured percentile
- CLI `--sla-percentile` flag on analyze, export, plan-capacity, what-if subcommands
- Config profile `sla.percentile` YAML key support
- Sensitivity and streaming analysis respect the configured percentile
- Backward compatible — default P95 behavior unchanged
- 34 new tests (288 total)

### M14 ✅ Benchmark Comparison & Regression Detection

*Completed — PR #35*

- `BenchmarkComparator` class in `comparator.py`
- `MetricDelta` and `ComparisonResult` Pydantic models
- Latency deltas at P50/P95/P99 for TTFT, TPOT, total latency
- QPS delta (higher-is-better semantics)
- Configurable regression threshold (default 10%)
- CLI `compare` subcommand with table and JSON output
- Programmatic `compare_benchmarks()` API
- 23 new tests (311 total)

### M15 ✅ Historical Trend Tracking

*Completed — PR #39*

- `TrendTracker` class in `trend.py`
- `TrendEntry` and `TrendReport` Pydantic models
- SQLite-backed storage for analysis results over time
- Detect gradual performance degradation across benchmark runs
- CLI `trend` subcommand: `trend add`, `trend show`, `trend check`
- Configurable lookback window and degradation threshold
- Programmatic `track_trend()` API
- ~20 new tests

### M16 ✅ Benchmark Data Validation & Outlier Detection

*Completed — PR #41*

- `DataValidator` class in `validator.py`
- `ValidationResult` and `DataQualityScore` Pydantic models
- Statistical outlier detection (IQR and Z-score methods)
- Data quality scoring: completeness, consistency, outlier ratio
- Automatic filtering of anomalous requests with reporting
- CLI `validate` subcommand with table and JSON output
- Integration with `analyze` subcommand via `--validate` flag
- Programmatic `validate_benchmark()` API
- ~22 new tests (351 total)

### M17 ✅ Interactive CLI Dashboard

*Completed — PR #43*

- Rich Live-based TUI dashboard for real-time monitoring
- Auto-refreshing panels: latency distribution, utilization, SLA status
- Support both file-based and streaming input
- Keyboard shortcuts for switching views
- CLI `dashboard` subcommand with `--refresh-interval`
- ~34 new tests (385 total)

### M18 ✅ Performance Interpolation Model

*Completed — PR #45*

- `PerformanceInterpolator` class in `interpolator.py`
- `PredictedPerformance` and `InterpolationResult` Pydantic models
- Linear and cubic spline interpolation methods
- Confidence classification: HIGH (interpolation), MEDIUM (near extrapolation ≤20%), LOW (far extrapolation)
- CLI `interpolate` subcommand with `--benchmark`, `--method`, `--ratios`, table + JSON output
- Programmatic `interpolate_performance()` API
- scipy dependency for spline interpolation
- 25 new tests (410 total)

### M19 ✅ Alert Rules Engine

*Completed — PR #47*

- `AlertEngine` class in `alerting.py`
- `AlertRule`, `AlertResult`, `AlertReport`, `AlertSeverity`, `Comparator` Pydantic models
- YAML rule file loading with metric + threshold + comparator + severity
- 11 supported metrics (TTFT/TPOT/total_latency at P50/P95/P99, QPS, request count)
- CLI `alert` subcommand with table and JSON output
- Non-zero exit code on critical alerts (CI/CD pipeline friendly)
- Programmatic `evaluate_alerts()` API
- 20 new tests (430 total)

### M20 ✅ Benchmark Annotation & Tagging

*Completed — PR #49*

- `AnnotationManager` class in `annotation.py`
- `Annotation`, `AnnotatedBenchmark`, `FilterResult` Pydantic models
- Sidecar `.tags.yaml` files (non-destructive, preserves original benchmark JSON)
- Add, remove, clear, list, and filter tags on benchmark files
- CLI `annotate` subcommand: `add`, `list`, `remove`, `filter`, `clear`
- Programmatic `annotate_benchmark()` API
- 26 new tests (456 total)

### M21 ✅ Pareto Frontier Analysis

*Completed — PR #52*

- `ParetoAnalyzer` class in `pareto.py`
- `ParetoCandidate`, `ParetoFrontier`, `ParetoObjective` Pydantic models
- Pareto dominance algorithm: identifies non-dominated P:D ratios across latency, cost, waste
- Weighted scoring for ranking Pareto-optimal candidates
- CLI `pareto` subcommand with table and JSON output
- Programmatic `find_pareto_frontier()` API
- ~26 new tests (482 total)

### M22 ✅ Recommendation Engine

*Completed — PR #54*

- `RecommendationEngine` class in `recommender.py`
- `Recommendation`, `RecommendationReport`, `RecommendationPriority` Pydantic models
- Combine SLA compliance, cost analysis, Pareto frontier, and trend data into ranked recommendations
- Priority levels: CRITICAL, HIGH, MEDIUM, LOW
- Action categories: SCALE_UP, SCALE_DOWN, REBALANCE, INVESTIGATE, NO_ACTION
- CLI `recommend` subcommand with table and JSON output
- Programmatic `get_recommendations()` API
- 22 new tests (504 total)

### M23 — Fleet Sizing Calculator

- `FleetCalculator` class in `fleet.py`
- `FleetConfig`, `FleetOption`, `FleetReport` Pydantic models
- Multi-GPU-type fleet optimization: given target QPS, available GPU types with costs, find cheapest fleet
- Per-GPU-type P:D ratio optimization using benchmark data
- Budget-constrained fleet sizing with cost ceiling
- Availability-zone aware instance distribution
- CLI `fleet` subcommand with `--target-qps`, `--gpu-configs`, table + JSON output
- Programmatic `calculate_fleet()` API
- ~24 new tests

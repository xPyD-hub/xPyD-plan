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

### M23 ✅ Fleet Sizing Calculator

*Completed — PR #56*

- `FleetCalculator` class in `fleet.py`
- `GPUTypeConfig`, `FleetAllocation`, `FleetOption`, `FleetReport` Pydantic models
- Multi-GPU-type fleet optimization: given target QPS, available GPU types with costs, find cheapest fleet
- Per-GPU-type P:D ratio optimization using benchmark data
- Budget-constrained fleet sizing with cost ceiling
- CLI `fleet` subcommand with `--target-qps`, `--gpu-configs`, table + JSON output
- Programmatic `calculate_fleet()` API
- 22 new tests (526 total)

### M24 ✅ Batch Pipeline Runner

*Completed — PR #59*

- `PipelineRunner` class in `pipeline.py`
- `PipelineConfig`, `PipelineStep`, `PipelineResult` Pydantic models
- YAML-defined pipeline: chain validate → analyze → compare → alert → report in one run
- Step dependency resolution (later steps consume earlier outputs)
- CLI `pipeline` subcommand with `--config pipeline.yaml`
- Dry-run mode (`--dry-run`) to preview pipeline without execution
- Programmatic `run_pipeline()` API
- 28 new tests

### M25 ✅ Markdown Report Generation

*Completed — PR #61*

- `MarkdownReporter` class in `md_report.py`
- `MarkdownReportConfig` Pydantic model
- Generate standalone `.md` reports (complement existing HTML reports)
- Sections: executive summary, SLA compliance, cost analysis, recommendations
- Embeddable in GitHub PRs and wiki pages
- CLI `report --format markdown` option alongside existing HTML
- Programmatic `generate_markdown_report()` API
- 33 new tests (587 total)

### M26 ✅ Benchmark Simulation Generator

*Completed — PR #64*

- `BenchmarkGenerator` class in `generator.py`
- `GeneratorConfig`, `LatencyProfile`, `AnomalyConfig` Pydantic models
- Generate synthetic benchmark data for testing and demos
- Configurable: QPS, instance counts, latency distributions (normal, log-normal, bimodal)
- Inject anomalies (spikes, cold starts) for testing validator/alerting
- CLI `generate` subcommand with `--config gen.yaml` and `--output benchmark.json`
- Programmatic `generate_benchmark()` API
- 33 new tests (620 total)

### M27 ✅ SLA Budget Allocation

*Completed — PR #66*

- `BudgetAllocator` class in `budget.py`
- `BudgetAllocation`, `StageBudget`, `AllocationStrategy` Pydantic models
- Analyze TTFT/TPOT contribution ratios from benchmark data at configurable percentiles
- 4 allocation strategies: proportional, balanced, ttft-priority, tpot-priority
- Feasibility check with headroom calculation
- CLI `budget` subcommand with `--benchmark`, `--total-budget-ms`, `--strategy`, table + JSON output
- Programmatic `allocate_budget()` API
- 23 new tests (643 total)

### M28 ✅ Benchmark Merge & Aggregation

*Completed — PR #68*

- `BenchmarkMerger` class in `merger.py`
- `MergeConfig`, `MergeResult`, `MergeStrategy` Pydantic models
- Union strategy: keep all unique requests, first occurrence wins for duplicates
- Intersection strategy: keep only requests present in all files
- Validates compatible cluster configurations (can be disabled)
- Aggregated metadata: combined QPS from all sources
- CLI `merge` subcommand with `--benchmark`, `--output`, `--strategy`, `--output-format`
- Programmatic `merge_benchmarks()` API
- 20 new tests (663 total)

### M29 ✅ Benchmark Filtering & Slicing

*Completed — PR #70*

- `BenchmarkFilter` class in `filter.py`
- `FilterConfig`, `BenchmarkFilterResult` Pydantic models
- Token count filters: min/max prompt_tokens, min/max output_tokens
- Latency filters: min/max ttft_ms, tpot_ms, total_latency_ms
- Time window: start_time / end_time (epoch seconds)
- Random sampling: sample_count or sample_fraction with reproducible seed
- QPS automatically adjusted proportionally to retention rate
- CLI `filter` subcommand with all filter flags, table + JSON output
- Programmatic `filter_benchmark()` API
- 28 new tests (691 total)

### M30 ✅ Refactor Monolithic cli.py into cli/ Package

*Completed — PR #72*

- Split 2504-line `cli.py` into `cli/` package with 23 focused modules
- `cli/__init__.py` re-exports `main()` and `_apply_config_defaults`
- `cli/_main.py` contains `main()` with argument parser setup and command dispatch
- `cli/_helpers.py` contains shared CLI helper functions
- `cli/_config.py` contains config helpers (`_add_config_flag`, `_apply_config_defaults`)
- Each subcommand in its own module (e.g., `_analyze.py`, `_export.py`, `_fleet.py`)
- Pure refactor — no behavioral changes
- All 691 tests pass unchanged

### M31 ✅ Bootstrap Confidence Intervals

*Completed — PR #74*

- `ConfidenceAnalyzer` class in `confidence.py`
- `ConfidenceInterval`, `MetricConfidence`, `ConfidenceReport`, `Adequacy` Pydantic models
- Bootstrap resampling (configurable iterations, default 1000) for percentile CI estimation
- Configurable confidence level (default 95%) and target percentile
- Sample size adequacy classification: SUFFICIENT, MARGINAL, INSUFFICIENT
- Adequacy thresholds based on relative CI width (>10% marginal, >25% insufficient)
- CLI `confidence` subcommand with `--benchmark`, `--percentile`, `--confidence-level`, `--iterations`, `--seed`, table + JSON output
- Programmatic `analyze_confidence()` API
- Reproducible results via `--seed` parameter
- 21 new tests (712 total)

### M32 ✅ Multi-Model Comparison Matrix

*Completed — PR #78*

- `ModelComparator` class in `model_compare.py`
- `ModelProfile`, `ModelComparison`, `ComparisonMatrix`, `ModelRanking` Pydantic models
- Load benchmark files tagged with model name (via CLI flag `--models name1,name2,...`)
- Side-by-side latency comparison across models at P50/P95/P99
- Rank models by cost-efficiency (cost per token at SLA compliance)
- Best-model recommendation per scenario with weighted scoring
- CLI `model-compare` subcommand with `--benchmarks`, `--models`, table + JSON output
- Programmatic `compare_models()` API
- 22 new tests (734 total)

### M33 ✅ A/B Test Analysis

*Completed — PR #81*

- `ABTestAnalyzer` class in `ab_test.py`
- `ABTestConfig`, `ABTestResult`, `StatisticalTest`, `EffectSize` Pydantic models
- Compare two benchmark files as control vs treatment with statistical rigor
- Welch's t-test for latency metric differences (TTFT, TPOT, total latency)
- Mann-Whitney U test as non-parametric alternative
- Effect size (Cohen's d) with magnitude classification (negligible/small/medium/large)
- Confidence intervals for mean difference
- Power analysis: warn when sample size is insufficient for detecting meaningful differences
- CLI `ab-test` subcommand with `--control`, `--treatment`, `--alpha`, `--metric`, table + JSON output
- Programmatic `analyze_ab_test()` API
- 41 new tests (775 total)

### M34 ✅ Workload Characterization & Clustering

*Completed — PR #83*

- `WorkloadClassifier` class in `workload.py`
- `WorkloadCategory`, `WorkloadClass`, `WorkloadProfile`, `WorkloadReport` Pydantic models
- 5 workload categories: PREFILL_HEAVY, DECODE_HEAVY, BALANCED, SHORT, LONG
- Per-class latency statistics (TTFT, TPOT, total) at P50/P95/P99
- Per-class SLA compliance check with margin calculation
- Bottleneck class identification (worst SLA margin)
- CLI `workload` subcommand with table + JSON output
- Programmatic `classify_workload()` API
- 25 new tests (800 total)

### M35 ✅ Throughput Scaling Analysis

*Completed — PR #85*

- `ScalingAnalyzer` class in `scaling.py`
- `ScalingPoint`, `ScalingCurve`, `ScalingReport` Pydantic models
- Per-instance throughput and scaling efficiency calculation (relative to baseline)
- Knee-point detection: first configuration where efficiency drops below threshold (default 80%)
- Optimal scaling point recommendation (highest QPS while above threshold)
- CLI `scaling` subcommand with `--benchmark` (multiple files), `--knee-threshold`, table + JSON output
- Programmatic `analyze_scaling()` API
- 21 new tests (821 total)

### M36 ✅ Prometheus/OpenMetrics Export

*Completed — PR #TBD*

- `MetricsExporter` class in `metrics_export.py`
- `MetricLine`, `MetricsReport` Pydantic models
- Export analysis results as Prometheus/OpenMetrics text format
- Metrics: latency percentiles (P50/P95/P99) for TTFT/TPOT/total, QPS, instance counts, request count
- Labels encode P:D ratio, instance configuration
- HELP and TYPE annotations per OpenMetrics spec
- CLI `metrics` subcommand with `--benchmark`, `--output`, `--output-format text|json|table`
- Programmatic `export_metrics()` API
- ~20 new tests

### M37 ✅ Correlation Analysis

*Completed — PR #89*

- `CorrelationAnalyzer` class in `correlation.py`
- `CorrelationPair`, `CorrelationReport`, `CorrelationStrength` Pydantic models
- Pearson r computation between request characteristics and latency metrics (6 pairs)
- Strength classification: strong (|r|≥0.7), moderate (0.4–0.7), weak (0.2–0.4), negligible (<0.2)
- CLI `correlation` subcommand with `--benchmark`, `--output-format table|json`
- Programmatic `analyze_correlation()` API
- 21 new tests (861 total)

### M38 ✅ Batch Benchmark Discovery & Auto-Loading

*Completed — PR #91*

- `BenchmarkDiscovery` class in `discovery.py`
- `DiscoveredBenchmark`, `DiscoveryReport`, `ConfigGroup`, `ValidationStatus` Pydantic models
- Recursive directory scanning with configurable depth limit
- Glob pattern filtering (`--pattern '**/*h100*.json'`)
- Quick validation without full deserialization
- Group-by-config summary
- CLI `discover` subcommand with `--dir`, `--pattern`, `--max-depth`, table + JSON output
- Programmatic `discover_benchmarks()` API
- 25 new tests (886 total)

### M39 ✅ Latency Heatmap Data Generation

*Completed — PR #93*

- `HeatmapGenerator` class in `heatmap.py`
- `HeatmapConfig`, `HeatmapCell`, `HeatmapGrid`, `HeatmapReport` Pydantic models
- 2D grid mapping (prompt_tokens bins × output_tokens bins) to aggregated latency metrics
- Configurable bin count, aggregation metric (mean, P50, P95, P99), and target latency field (TTFT, TPOT, total)
- Hotspot detection: identify cells exceeding SLA thresholds
- CLI `heatmap` subcommand with `--benchmark`, `--bins`, `--metric`, `--field`, table + JSON output
- Programmatic `generate_heatmap()` API
- 25 new tests (911 total)

### M40 ✅ Request Timeline Analysis

*Completed — PR #96*

- `TimelineAnalyzer` class in `timeline.py`
- `TimeWindow`, `WarmupAnalysis`, `LatencyTrend`, `TimelineReport` Pydantic models
- Time-window segmentation with configurable window size (seconds)
- Per-window latency statistics (TTFT/TPOT/total at P50/P95)
- Warmup detection: flag initial windows where P95 latency exceeds N× steady-state median
- Trend detection via linear regression: IMPROVING, DEGRADING, STABLE classification
- CLI `timeline` subcommand with `--benchmark`, `--window-size`, `--warmup-factor`, table + JSON output
- Programmatic `analyze_timeline()` API
- 29 new tests (940 total)

### M41 — Distribution Drift Detection

*In progress*

- `DriftDetector` class in `drift.py`
- `DriftResult`, `DriftReport`, `DriftSeverity` Pydantic models
- Kolmogorov-Smirnov two-sample test for TTFT, TPOT, and total_latency distributions
- Severity classification: NONE (p>0.05), MINOR (p≤0.05, D<0.2), MODERATE (D≥0.2), MAJOR (D≥0.4)
- Summary with which metrics drifted and by how much
- CLI `drift` subcommand with `--baseline`, `--current`, table + JSON output
- Programmatic `detect_drift()` API
- ~22 new tests

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

*Completed — PR #87*

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

### M41 ✅ Distribution Drift Detection

*Completed — PR #98*

- `DriftDetector` class in `drift.py`
- `DriftResult`, `DriftReport`, `DriftSeverity` Pydantic models
- Kolmogorov-Smirnov two-sample test for TTFT, TPOT, and total_latency distributions
- Severity classification: NONE (p>0.05), MINOR (p≤0.05, D<0.2), MODERATE (D≥0.2), MAJOR (D≥0.4)
- Summary with which metrics drifted and by how much
- CLI `drift` subcommand with `--baseline`, `--current`, table + JSON output
- Programmatic `detect_drift()` API
- ~22 new tests

### M42 ✅ Anomaly Root Cause Analysis

*Completed — PR #100*

- `RootCauseAnalyzer` class in `root_cause.py`
- `RootCause`, `CauseFactor`, `RootCauseReport`, `FactorSignificance` Pydantic models
- Segment requests into SLA-passing vs SLA-failing groups
- Compare prompt_tokens, output_tokens distributions between groups (Mann-Whitney U)
- Compare temporal patterns between groups
- Rank factors by effect size and p-value
- Significance: HIGH (p<0.001), MEDIUM (p<0.01), LOW (p<0.05), NONE (p>=0.05)
- CLI `root-cause` subcommand with `--benchmark`, `--sla-ttft`, `--sla-tpot`, table + JSON output
- Programmatic `analyze_root_cause()` API
- ~25 new tests

### M43 ✅ Tail Latency Analysis

*Completed — PR #103*

- `TailAnalyzer` class in `tail.py`
- `TailMetric`, `TailReport`, `TailClassification`, `LongTailProfile` Pydantic models
- Extended percentile computation: P50, P90, P95, P99, P99.9, P99.99 for TTFT, TPOT, total_latency
- Tail ratio metrics: P99/P50, P99.9/P50 — quantify tail heaviness
- Tail classification: LIGHT (<2x), MODERATE (2-5x), HEAVY (5-10x), EXTREME (>10x)
- Long-tail request characterization: token distribution stats for P99+ requests
- CLI `tail` subcommand with table + JSON output
- Programmatic `analyze_tail()` API
- 17 new tests (1015 total)

### M44 ✅ Efficiency Scorecard

*Completed — PR #105*

- `ScorecardCalculator` class in `scorecard.py`
- `ConfigScorecard`, `DimensionScore`, `ScoreGrade`, `ScorecardReport` Pydantic models
- Composite 0-100 score combining SLA compliance (40%), utilization (30%), and waste (30%)
- Configurable dimension weights
- Letter grading: A (90+), B (75+), C (60+), D (40+), F (<40)
- CLI `scorecard` subcommand with `--benchmark`, `--sla-*`, `--*-weight`, table + JSON output
- Programmatic `calculate_scorecard()` API
- 31 new tests (1046 total)

### M45 ✅ Benchmark Plan Generator

*Completed — PR #107*

- `BenchmarkPlanGenerator` class in `plan_generator.py`
- `PlannedRatio`, `BenchmarkPlan`, `RatioPriority` Pydantic models
- Prioritized ratio enumeration: balanced (critical), boundaries (high), quarters (high), rest (medium)
- Refinement mode: given existing `AnalysisResult`, focus on neighbors of current best
- Max-runs cap to limit recommended configurations
- CLI `plan-benchmarks` subcommand with `--total-instances`, `--max-runs`, table + JSON output
- Programmatic `generate_benchmark_plan()` API
- 19 new tests (1065 total)

### M46 ✅ SLA Threshold Tuning Advisor

*Completed — PR #109*

- `ThresholdAdvisor` class in `threshold_advisor.py`
- `ThresholdSuggestion`, `AdvisorReport`, `SweetSpot`, `PassRateTarget` Pydantic models
- For each latency metric, compute the threshold needed to achieve target pass rates
- Sweet-spot detection: identify inflection points where small threshold relaxation yields large compliance gains
- CLI `threshold-advisor` subcommand with `--benchmark`, `--pass-rates`, table + JSON output
- Programmatic `advise_thresholds()` API
- 19 new tests (1084 total)

### M47 ✅ Capacity Forecasting

*Completed — PR #111*

- `CapacityForecaster` class in `forecaster.py`
- `ForecastPoint`, `ForecastReport`, `ForecastMethod`, `CapacityExhaustion` Pydantic models
- Given historical trend data (from TrendTracker), project future latency and QPS trajectories
- Linear and exponential extrapolation methods
- Estimate time-to-SLA-breach: when will latency percentiles exceed SLA thresholds at current growth rate
- Capacity exhaustion warning with configurable planning horizon (default 30 days)
- CLI `forecast` subcommand with `--trend-db`, `--horizon-days`, `--method`, table + JSON output
- Programmatic `forecast_capacity()` API
- ~20 new tests

### M48 ✅ Multi-SLA Tier Analysis

*Completed — PR #113*

- `SLATierAnalyzer` class in `sla_tier.py`
- `SLATier`, `TierResult`, `MultiTierReport` Pydantic models
- Analyze benchmark data against multiple named SLA policies simultaneously
- Per-tier optimal P:D ratio finding
- Unified ratio identification (single P:D ratio satisfying all tiers)
- YAML tier definition loading (`load_tiers_from_yaml()`)
- CLI `sla-tier` subcommand with `--benchmark`, `--tiers`, table + JSON output
- Programmatic `analyze_sla_tiers()` API
- 18 new tests (1120 total)

### M49 ✅ Saturation Point Detection

*Completed — PR #115*

- `SaturationDetector` class in `saturation.py`
- `SaturationPoint`, `SaturationThreshold`, `SaturationReport` Pydantic models
- Analyze multiple benchmarks at increasing QPS levels to find saturation point
- Per-metric saturation detection (TTFT/TPOT/total at P95/P99)
- Configurable increase threshold (default 50% relative increase)
- Conservative overall safe QPS (minimum across all saturated metrics)
- CLI `saturation` subcommand with `--benchmark`, `--increase-threshold`, table + JSON output
- Programmatic `detect_saturation()` API
- 21 new tests (1141 total)

### M50 ✅ Latency Decomposition Analysis

*Completed — PR #118*

- `LatencyDecomposer` class in `decomposer.py`
- `DecomposedRequest`, `DecompositionReport`, `BottleneckType`, `PhaseStats` Pydantic models
- Per-request decomposition: prefill_fraction, decode_fraction, overhead_fraction
- Aggregate statistics at P50/P95 for each phase
- Bottleneck classification: PREFILL_BOUND, DECODE_BOUND, OVERHEAD_BOUND, BALANCED (configurable threshold, default 50%)
- Actionable recommendations based on bottleneck type
- CLI `decompose` subcommand with `--benchmark`, `--bottleneck-threshold`, table + JSON output
- Programmatic `decompose_latency()` API
- 23 new tests (1164 total)

### M51 ✅ Comprehensive README Rewrite

*Completed — PR #120*

- Rewrite README.md to reflect all 50 milestones of current capabilities
- Accurate quick-start examples using current CLI subcommands
- Feature matrix grouped by category (analysis, comparison, planning, etc.)
- Installation instructions with optional dependencies
- Remove all references to deprecated estimation-based workflow

### M52 ✅ Plugin Architecture

*Completed — PR #122*

- `PluginSpec` ABC defining the interface plugins must implement
- `PluginMetadata` Pydantic model (name, version, description, author, type)
- `PluginRegistry` class: discover and load plugins via `importlib.metadata.entry_points()`
- Plugins can register analysis functions and CLI subcommands
- CLI `plugins` subcommand to list installed plugins (table + JSON)
- Built-in plugin validation (interface compliance check)
- Programmatic `list_plugins()`, `get_plugin()`, `get_registry()` API
- 23 new tests (1187 total)

### M53 ✅ End-to-End Integration Tests

*Completed — PR #124*

- 54 new integration tests exercising full CLI workflows
- Multi-file benchmark fixtures (different QPS levels, P:D ratios, instance counts)
- Pipeline integration test (validate → analyze chained via YAML)
- JSON output schema stability assertions
- 3 xfail tests documenting pre-existing CLI bugs (scorecard, pareto, threshold-advisor)
- CI matrix: Python 3.10, 3.11, 3.12
- 1241 passed, 3 xfailed

### M54 ✅ Fix Pre-Existing CLI Bugs (scorecard, pareto, threshold-advisor)

*Completed — PR #126*

- Fix `scorecard` subcommand: add missing `total_instances` argument to `find_optimal_ratio()`
- Fix `pareto` subcommand: use `analyzer._data = data` instead of passing to constructor
- Fix `threshold-advisor` subcommand: load data via `load_benchmark_auto()` first
- Convert 3 xfail integration tests to passing
- 1244 passed, 0 xfail

### M55 ✅ Request Fairness Analysis

*Completed — PR #128*

- `FairnessAnalyzer` class in `fairness.py`
- `FairnessReport`, `BucketStats`, `FairnessIndex`, `FairnessClassification` Pydantic models
- Bucket requests by prompt_tokens into configurable quantile-based bins (default 4)
- Per-bucket P50/P95 latency stats for TTFT, TPOT, total_latency
- Jain's fairness index (J = (Σxi)² / (n·Σxi²)) computed on per-bucket P95 latencies
- Fairness classification: FAIR (J≥0.9), MODERATE (0.7-0.9), UNFAIR (<0.7)
- CLI `fairness` subcommand with `--benchmark`, `--buckets`, table + JSON output
- Programmatic `analyze_fairness()` API
- 26 new tests (1270 total)

### M56 ✅ Benchmark Quick Summary

*Completed — PR #130*

- `SummaryGenerator` class in `summary.py`
- `SummaryReport`, `TokenStats`, `LatencyOverview` Pydantic models
- Compact overview: request count, duration, measured QPS, P:D ratio, instance counts
- Token distribution stats (min, mean, P50, P95, max) for prompt and output tokens
- Latency overview (min, mean, P50, P95, P99, max) for TTFT, TPOT, total latency
- CLI `summary` subcommand with `--benchmark`, table + JSON output
- Programmatic `summarize_benchmark()` API
- 19 new tests (1289 total)

### M57 ✅ Outlier Impact Analysis

*Completed — PR #132*

- `OutlierImpactAnalyzer` class in `outlier_impact.py`
- `ImpactReport`, `MetricImpact`, `SLAComplianceComparison`, `ImpactRecommendation` Pydantic models
- IQR-based outlier identification (configurable multiplier, default 1.5)
- Per-metric (TTFT, TPOT, total_latency) impact at P50/P95/P99
- SLA compliance comparison (before/after outlier removal)
- Recommendation logic: FILTER if SLA flips or >5% P95 shift, KEEP otherwise
- CLI `outlier-impact` subcommand with `--benchmark`, `--sla-*`, `--iqr-multiplier`, table + JSON output
- Programmatic `analyze_outlier_impact()` API
- 21 new tests (1331 total)

### M58 ✅ Percentile Convergence Analysis

*Completed — PR #134*

- `ConvergenceAnalyzer` class in `convergence.py`
- `ConvergencePoint`, `ConvergenceReport`, `MetricConvergence`, `StabilityStatus` Pydantic models
- Running P50/P95/P99 computation at cumulative sample windows (configurable steps)
- CV-based stability classification: STABLE (≤threshold), MARGINAL (≤2×threshold), UNSTABLE
- Minimum stable sample size detection per metric
- CLI `convergence` subcommand with `--benchmark`, `--steps`, `--threshold`, table + JSON output
- Programmatic `analyze_convergence()` API
- 21 new tests (1331 total)

### M59 ✅ Load Profile Classification

*Completed — PR #136*

- `LoadProfileClassifier` class in `load_profile.py`
- `LoadProfile`, `LoadProfileReport`, `ProfileType`, `RateWindow` Pydantic models
- Time-windowed request rate computation with configurable window size
- 6 profile types: STEADY_STATE, RAMP_UP, RAMP_DOWN, BURST, CYCLIC, UNKNOWN
- Rate CV analysis for steady-state detection
- Linear regression on rate for ramp classification
- Peak-to-trough ratio for burst detection
- Direction change counting for cyclic detection
- CLI `load-profile` subcommand with `--benchmark`, `--window-size`, table + JSON output
- Programmatic `classify_load_profile()` API
- ~22 new tests

### M60 ✅ Throughput Percentile Analysis

*Completed — PR #138*

- `ThroughputAnalyzer` class in `throughput.py`
- `ThroughputReport`, `ThroughputBucket`, `ThroughputStats`, `ThroughputStability` Pydantic models
- Compute per-second completed request counts from timestamp data
- Throughput distribution: min, mean, P50, P95, P99, max requests/sec
- Stability classification: STABLE (CV≤0.15), VARIABLE (0.15–0.40), UNSTABLE (>0.40)
- Bottleneck second detection: identify seconds where throughput drops below a threshold
- Sustainable throughput estimation (P5 of per-second counts)
- CLI `throughput` subcommand with `--benchmark`, `--bucket-size`, table + JSON output
- Programmatic `analyze_throughput()` API
- ~22 new tests

### M61 ✅ Token Efficiency Analysis

*Completed — PR #140*

- `TokenEfficiencyAnalyzer` class in `token_efficiency.py`
- `TokenEfficiencyReport`, `AggregateEfficiency`, `InstanceEfficiency`, `PerRequestEfficiency`, `EfficiencyGrade` Pydantic models
- Per-request output tokens/sec, total tokens/sec, and decode tokens/sec
- Aggregate statistics: mean, P50, P95, P99, min, max for output and decode TPS
- Instance efficiency: output tokens per instance, per decode instance, prompt tokens per prefill instance
- Efficiency grade (EXCELLENT/GOOD/FAIR/POOR) based on P95/P50 throughput ratio
- CLI `token-efficiency` subcommand with `--benchmark`, `--details`, table + JSON output
- Programmatic `analyze_token_efficiency()` API
- 18 new tests (1390 total)

---

## Phase 2: Continuous Evolution

### M62 ✅ Request Queuing Time Analysis

*Completed — PR #144*

- `QueueAnalyzer` class in `queue_analysis.py`
- `QueueReport`, `QueueStats`, `ConcurrencyProfile` Pydantic models
- Estimate per-request queuing delay from timestamp gaps and concurrency overlap
- Concurrency profile: concurrent in-flight requests over time
- Peak concurrency detection and queue depth estimation
- CLI `queue` subcommand with `--benchmark`, table + JSON output
- Programmatic `analyze_queue()` API
- ~20 new tests

### M63 ✅ Batch Size Impact Analysis

*Completed — PR #146*

- `BatchAnalyzer` class in `batch_analysis.py`
- `BatchReport`, `BatchBucket`, `BatchEfficiency` Pydantic models
- Group requests by temporal proximity into inferred batches
- Per-batch latency and throughput statistics
- Optimal batch size recommendation based on throughput/latency tradeoff
- CLI `batch-analysis` subcommand with `--benchmark`, `--window-ms`, table + JSON output
- Programmatic `analyze_batch_impact()` API
- ~20 new tests

### M64 ✅ Multi-Benchmark Statistical Summary

*Completed — PR #148*

- `StatSummaryAnalyzer` class in `stat_summary.py`
- `StatSummaryReport`, `RunSummary`, `AggregatedStats`, `LatencyAggStats` Pydantic models
- Load N benchmark files and compute cross-run statistics (mean of means, std of P95s, etc.)
- Identify most/least stable runs via normalized P95 deviation
- Coefficient of variation across runs for repeatability assessment
- CLI `stat-summary` subcommand with `--benchmark` (multiple), table + JSON output
- Programmatic `summarize_stats()` API
- 19 new tests (1450 total)

### M65 ✅ Cost Projection & ROI Calculator

*Completed — PR #150*

- `ROICalculator` class in `roi.py`
- `ROIReport`, `CostProjection`, `SavingsEstimate` Pydantic models
- Given current P:D ratio cost and optimal ratio cost, project monthly/yearly savings
- Break-even analysis: how many hours until migration cost is recovered
- CLI `roi` subcommand with `--benchmark`, `--cost-model`, `--migration-cost`, table + JSON output
- Programmatic `calculate_roi()` API
- 20 new tests (1470 total)

### M66 ✅ Benchmark Schema Versioning & Migration

*Completed — PR #152*

- `SchemaMigrator` class in `schema_migrate.py`
- `SchemaVersion`, `MigrationResult` Pydantic models
- Detect benchmark JSON schema version from file content
- Migrate older schema versions to current format (v1 → v2 transformations)
- v2 adds: top-level `schema_version`, `metadata.run_id`, `metadata.schema_version`
- Dry-run mode to preview changes without writing
- CLI `migrate` subcommand with `--benchmark`, `--target-version`, `--dry-run`, `--output`, `--output-format`
- Programmatic `migrate_schema()` API
- 23 new tests (1493 total)

### M67 ✅ Request Replay Schedule Generator

*Completed — PR #154*

- `ReplayGenerator` class in `replay.py`
- `ReplaySchedule`, `ReplayEntry`, `ReplayConfig` Pydantic models
- Extract request arrival times and token counts from benchmark data
- Generate reproducible replay schedule (JSON) with relative timestamps
- Time scaling: speed up or slow down schedule (`--time-scale` factor)
- QPS override: redistribute arrivals uniformly (`--target-qps`)
- Support both native and xpyd-bench benchmark formats
- CLI `replay` subcommand with `--benchmark`, `--time-scale`, `--target-qps`, `--output`, `--output-format`
- Programmatic `generate_replay()` API
- 25 new tests (1518 total)

### M68 ✅ Latency-Token Regression Analysis

*Completed — PR #156*

- `RegressionAnalyzer` class in `regression.py`
- `RegressionFit`, `PredictedLatency`, `RegressionReport` Pydantic models
- Linear regression fits: TTFT~prompt_tokens, TPOT~output_tokens, total_latency~total_tokens
- R², slope, intercept, slope standard error for each fit
- Optional prediction with 95% confidence interval
- CLI `regression` subcommand with table and JSON output
- Programmatic `analyze_regression()` API
- 18 new tests (1536 total)

### M69 ✅ Latency CDF Export

*Completed — PR #158*

- `CDFGenerator` class in `cdf.py`
- `CDFPoint`, `CDFCurve`, `SLAMarker`, `CDFReport` Pydantic models
- Generate CDF data points for TTFT, TPOT, total_latency distributions
- Multi-benchmark CDF overlay for visual comparison
- SLA threshold markers with pass-rate at each threshold
- CLI `cdf` subcommand with `--benchmark` (multiple), `--sla-*`, `--points`, table + JSON + CSV output
- Programmatic `generate_cdf()` API
- 20 new tests (1556 total)

### M70 ✅ Benchmark Health Check

*Completed — PR #160*

- `HealthChecker` class in `health_check.py`
- `CheckResult`, `HealthReport`, `HealthStatus` Pydantic models
- Composite readiness check: data validation, percentile convergence, load profile, outlier impact
- Single PASS/WARN/FAIL verdict with per-check details
- CLI `health-check` subcommand with `--benchmark`, table + JSON output
- Programmatic `check_health()` API
- 18 new tests (1574 total)

### M72 ✅ Request Arrival Pattern Analysis

*Completed — PR #170*

- `ArrivalPatternAnalyzer` class in `arrival_pattern.py`
- `ArrivalPattern`, `ArrivalPatternReport`, `InterArrivalStats`, `BurstInfo` Pydantic models
- Inter-arrival time statistics: mean, std, CV, percentiles (P50/P95/P99)
- Pattern classification: POISSON, BURSTY, PERIODIC, UNIFORM, UNKNOWN
- Burst detection with configurable threshold (fraction of mean IAT)
- Poisson fit assessment via CV-based exponentiality test
- Periodicity detection via lag-1 autocorrelation
- CLI `arrival-pattern` subcommand with `--benchmark`, `--burst-threshold`, table + JSON output
- Programmatic `analyze_arrival_pattern()` API
- 19 new tests (1593 total)

### M73 ✅ Benchmark Sampling & Downsampling

*Completed — PR #166*

- `BenchmarkSampler` class in `sampler.py`
- `SampleResult`, `SampleConfig`, `SampleValidation`, `SamplingMethod`, `MetricDeviation` Pydantic models
- Random sampling with reproducible seed
- Stratified sampling by prompt_tokens quantile bins to preserve distribution shape
- Reservoir sampling (Algorithm R) for streaming/unknown-size inputs
- Statistical validation: P50/P95/P99 deviation between original and sample with tolerance check
- QPS adjustment proportional to sample fraction
- CLI `sample` subcommand with `--benchmark`, `--method`, `--size`, `--seed`, `--bins`, `--tolerance`, `--output`, table + JSON output
- Programmatic `sample_benchmark()` API
- 21 new tests (1636 total)

### M74 ✅ Request Size Distribution Analysis

*Completed — PR #168*

- `SizeDistributionAnalyzer` class in `size_distribution.py`
- `SizeDistributionReport`, `Histogram`, `SizeBin`, `DistributionShape`, `SizeLatencyCorrelation` Pydantic models
- Configurable bin count for prompt and output token histograms
- Distribution shape classification: UNIFORM, RIGHT_SKEWED, LEFT_SKEWED, BIMODAL, NORMAL
- Per-size-bin latency statistics (P50/P95 TTFT, TPOT, total_latency)
- Size-latency Pearson correlation for prompt and output tokens
- CLI `size-distribution` subcommand with `--benchmark`, `--bins`, table + JSON output
- Programmatic `analyze_size_distribution()` API
- 19 new tests (1655 total)

### M75 ✅ Concurrency Utilization Analysis

*Completed — PR #170*

- `ConcurrencyUtilizationAnalyzer` class in `concurrency_util.py`
- `UtilizationWindow`, `UtilizationReport`, `UtilizationLevel`, `RightSizingRecommendation` Pydantic models
- Time-windowed concurrent request count estimation via temporal sampling
- Per-window utilization classification: IDLE (<20%), LOW (20-50%), MODERATE (50-80%), HIGH (>80%)
- Idle window and high-utilization window detection and counting
- Right-sizing recommendation: min/target instances based on average/P95 concurrency with 20% headroom
- Over-provisioned and under-provisioned detection
- CLI `concurrency-util` subcommand with `--benchmark`, `--window-size`, table + JSON output
- Programmatic `analyze_concurrency_util()` API
- 19 new tests (1674 total)

### M76 ✅ Benchmark Reproducibility Score

*Completed — PR #TBD*

- `ReproducibilityAnalyzer` class in `reproducibility.py`
- `ReproducibilityReport`, `MetricReproducibility`, `ReproducibilityGrade`, `RunPairTest` Pydantic models
- Per-metric coefficient of variation (CV) across repeated runs
- KS two-sample test between all run pairs for distribution consistency
- Composite 0-100 reproducibility score with grade classification (EXCELLENT/GOOD/FAIR/POOR)
- Unreliable metric flagging when CV exceeds configurable threshold
- Recommended minimum runs based on observed variance
- CLI `reproducibility` subcommand with `--benchmark` (multiple), `--cv-threshold`, table + JSON output
- Programmatic `analyze_reproducibility()` API
- ~20 new tests

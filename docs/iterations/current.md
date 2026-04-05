# xPyD-plan Current Iteration Status

> Last updated: 2026-04-05

---

## Current Milestone: M104 — Latency Baseline Manager ✅

The project has completed **104 milestones**, covering the full feature chain from core analysis to advanced optimization.

---

## Feature List

### Core Analysis
- Benchmark data loading, validation, and SLA compliance analysis
- Optimal P:D ratio search (multi-scenario support)
- Sensitivity analysis and cliff detection
- Bootstrap confidence intervals
- Latency decomposition (prefill/decode/overhead)
- Tail latency analysis (P99.9, P99.99)

### Comparison & Testing
- Benchmark comparison and regression detection
- A/B statistical testing (Welch's t-test, Mann-Whitney U)
- Multi-model comparison
- Distribution drift detection (KS test)

### Planning & Optimization
- Capacity planning (linear scaling model)
- What-if scenario simulation
- Multi-GPU-type fleet sizing
- Pareto frontier analysis
- Comprehensive deployment recommendations (recommend)
- Performance interpolation/extrapolation
- Capacity forecasting (historical trends)
- SLA threshold tuning

### Cost & Budget
- Cost-aware optimization
- SLA budget allocation
- Comprehensive efficiency scorecard
- Multi-SLA tier analysis

### Data Management
- Data quality scoring and outlier detection
- Filtering, merging, annotation, and discovery
- Synthetic data generation
- Batch export (JSON/CSV/table)
- SQLite export
- Schema migration
- Benchmark session management

### Monitoring & Alerting
- Rich TUI real-time dashboard
- YAML alert rules
- Historical trend tracking
- Prometheus metrics export
- Timeline analysis
- Saturation point detection
- Health checks

### Advanced Analysis
- Workload clustering
- Correlation analysis
- Latency heatmap
- Root cause analysis
- Scaling efficiency and inflection point detection
- QPS curve fitting
- Latency variance decomposition
- Latency prediction integration
- Latency anomaly classification
- SLA margin calculation
- Latency baseline management

### Reporting & Configuration
- HTML/Markdown report generation
- YAML configuration system
- Multi-step pipeline executor
- Replay scheduling

---

## Known Limitations

1. **Offline analysis only** — Does not support real-time streaming ingestion of production traffic data
2. **Linear scaling assumption** — `plan-capacity` uses a linear model, which may deviate from reality under high concurrency
3. **Single-cluster perspective** — Cross-cluster/cross-region joint optimization is not yet supported
4. **GPU model coverage** — Built-in profiles only include A100/H100; other models require manual configuration
5. **Benchmark format** — Only supports xpyd-bench native format and compatible JSON; direct import from other benchmark tools is not supported

---

## Next Steps

- Expand GPU profile library (L40S, B200, and other new hardware)
- Explore online/streaming analysis mode
- Cross-cluster joint optimization
- Closed-loop integration with xPyD-proxy auto-tuning
- Web UI dashboard (replacing TUI)
- Richer visualizations (interactive charts)

# xPyD-plan Roadmap

## Vision

Help users find the **optimal Prefill:Decode GPU ratio** for their workloads — satisfy SLA constraints at the lowest cost.

---

## Milestones

### M1 ✅ Core Data Models + Brute-force Search

*Completed — PR #2*

- Pydantic data models (`SLAConfig`, `DatasetStats`, `GPUProfile`, `PDConfig`, etc.)
- Linear performance estimator (simple throughput-proportional model)
- Brute-force enumeration of all P:D splits
- CLI with YAML config input and Rich table output
- 17 tests

### M2: Realistic Performance Model

Replace the naïve linear estimator with a queuing-theory model that accounts for real-world effects.

- **Queuing model** — M/M/c (Erlang-C) to estimate waiting time under concurrent load
- **GPU profile library** — Built-in profiles for A100-80G, H100-80G (prefill/decode throughput, cost)
- **Batching effect** — Model how continuous batching improves per-GPU throughput at higher concurrency
- **Backward compatible** — Existing CLI and config format continue to work

### M3: Advanced SLA Constraint Engine

- Support percentile-based latency constraints (P50 / P95 / P99)
- Throughput floor constraint (min requests/s)
- Multi-dimensional SLA (combine latency + throughput + cost ceiling)
- Weighted scoring with user-defined priorities

### M4: Smart Search Optimization

- Replace brute-force enumeration with binary / heuristic search
- Support heterogeneous GPU pools (mix of GPU types)
- Scale to hundreds of nodes without blowup

### M5: Benchmark Data Integration

- Import real benchmark results from `xpyd-bench`
- Calibrate / override model parameters with measured data
- Regression fitting for throughput-vs-concurrency curves

### M6: Report Generation

- HTML report with charts (throughput vs. cost Pareto front, latency breakdown)
- PDF export
- Exportable comparison tables

### M7: Configuration Template Library

- Pre-built GPU profiles: A100-40G, A100-80G, H100-80G, H200
- Pre-built model profiles: Llama-3-70B, Qwen-2.5-72B, DeepSeek-V3
- `xpyd-plan init` to scaffold a config from templates

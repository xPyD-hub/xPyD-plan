# xPyD-plan Design Principles

## Core Positioning
Offline planning tool: analyze vLLM benchmark data from real hardware to recommend the optimal P:D configuration.

## Methodology
1. Users run vLLM benchmarks on real hardware with several P:D configurations we specify
2. Collect multiple measured data points (TTFT, TPOT/ITL, throughput, etc.)
3. Analyze data, fit/interpolate, build performance model
4. Extrapolate predicted performance for all possible P:D ratios
5. Find the P:D combination that meets SLA with minimum waste (best cost-efficiency)

## Principles

### Data-Driven
- Everything is based on real hardware benchmark data
- No pure theoretical simulation, no guessing hardware performance
- Data format: vLLM benchmark standard output

### Independent Thinking
- Reference industry solutions (e.g., dynamo planner) for ideas, but never copy
- Every technical decision must have its own analysis and reasoning
- Algorithm choices based on actual data characteristics, not "because others do it"

### User-Friendly
- Minimize the number of benchmark runs users need to perform
- Clearly tell users which configurations to benchmark
- Recommendations must include confidence/reliability assessment
- Be honest about uncertainty when sample points are few

### Rigorous Definitions
- "Optimal" = minimum idle waste on P and D instances while meeting SLA
- "Waste" must have a strict mathematical definition, no ambiguity
- SLA checks use measured percentiles (P95/P99), not averages

### Granularity
- Measured in instances, not individual GPU cards
- QPS is not optimized — it is a given from the benchmark

## References
- ai-dynamo/dynamo planner module: reference its ideas on profiling data interpolation and SLA checking
- However, dynamo is an online autoscaler; we are an offline planning tool — different scenarios
- Do not copy dynamo's code or data formats

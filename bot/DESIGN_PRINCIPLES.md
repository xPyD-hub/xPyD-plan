<!-- ⚠️ DO NOT COMPRESS, SUMMARIZE, OR SKIP ANY PART OF THIS FILE ⚠️ -->

# xPyD-plan Design Principles

## Core Positioning

Offline planning tool: analyze benchmark data from real hardware to recommend the optimal P:D configuration.

## Methodology

1. Users run benchmarks on real hardware with specified P:D configurations
2. Collect measured data points (TTFT, TPOT/ITL, throughput, etc.)
3. Analyze data, fit/interpolate, build performance model
4. Extrapolate predicted performance for all possible P:D ratios
5. Find the P:D combination that meets SLA with minimum waste

## Principles

### Data-Driven
- Everything based on real hardware benchmark data
- No pure theoretical simulation, no guessing
- Supports vLLM, SGLang, TensorRT-LLM formats

### Independent Thinking
- Reference industry solutions for ideas, never copy blindly
- Every technical decision must have its own analysis

### User-Friendly
- Minimize required benchmark runs
- Clearly specify which configurations to benchmark
- Include confidence/reliability assessment
- Be honest about uncertainty

### Rigorous Definitions
- "Optimal" = minimum idle waste while meeting SLA
- SLA checks use measured percentiles (P95/P99), not averages
- Granularity: instances, not individual GPU cards

## References
- ai-dynamo planner: reference its ideas on profiling data interpolation
- dynamo is online autoscaler; we are offline planner — different scenarios

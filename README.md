# xPyD-plan

PD ratio planner for [xPyD-proxy](https://github.com/xPyD-hub/xPyD-proxy) — recommend optimal Prefill:Decode node allocation based on benchmark data or dataset characteristics.

## Features

- Analyze [xpyd-bench](https://github.com/xPyD-hub/xPyD-bench) results to recommend P:D ratio
- Offline estimation from dataset (prompt length distribution + expected output length)
- Model prefill/decode cost and find optimal split for a given GPU budget

## Install

```bash
pip install xpyd-plan
```

## Quick Start

```bash
# From benchmark results
xpyd-plan --data bench_results.json --budget 8

# From dataset (offline, no benchmark needed)
xpyd-plan --dataset prompts.json --budget 8

# Output to file
xpyd-plan --data results.json --budget 8 --output recommendation.json
```

## How It Works

1. **Input**: benchmark results (from `xpyd-bench`) or a dataset of prompts
2. **Model**: estimates prefill cost ≈ f(input_tokens) and decode cost ≈ g(output_tokens)
3. **Optimize**: given N total nodes, finds the P:D split that maximizes throughput
4. **Output**: recommended ratio with reasoning and confidence

## License

TBD

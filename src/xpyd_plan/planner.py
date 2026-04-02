"""Planner core — cost modeling and ratio optimization."""

from __future__ import annotations

# TODO: implement
# - load benchmark results (from xpyd-bench output JSON)
# - load dataset for offline estimation (prompt length distribution)
# - model prefill cost: f(input_tokens)
# - model decode cost: g(output_tokens)
# - given budget N nodes, recommend P:D split that maximizes throughput
# - output recommendation with reasoning and confidence interval

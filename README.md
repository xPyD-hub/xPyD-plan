# xPyD-plan

PD ratio planner — recommend optimal **Prefill:Decode** allocation from real benchmark data.

## Install

```bash
pip install xpyd-plan
# With HTML reports
pip install "xpyd-plan[report]"
# Development
pip install "xpyd-plan[dev]"
```

## Quick Start

```bash
# Find optimal P:D ratio from benchmark results
xpyd-plan analyze --benchmark results.json --sla-ttft 200 --sla-tpot 50

# Compare two benchmark runs
xpyd-plan compare --baseline baseline.json --current current.json

# Generate report
xpyd-plan report --format markdown --benchmark results.json --output report.md
```

## License

[Apache 2.0](LICENSE)

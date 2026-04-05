# xPyD-plan Current Iteration Status

> Last updated: 2026-04-05

---

## Current Milestone: M107 — Request Rate Limiter Recommender ✅

Recommend safe request rate limits based on multi-QPS benchmark data. Interpolates latency vs QPS curves to find the maximum sustainable QPS within SLA constraints, then suggests rate limits with configurable safety margins.

### Changes
- **`src/xpyd_plan/rate_recommender.py`**: `RateRecommender` class with interpolation-based max QPS finder and safety margin
- **`tests/test_rate_recommender.py`**: 19 new tests
- **`src/xpyd_plan/cli/_rate_recommend.py`**: CLI `rate-recommend` subcommand
- **`src/xpyd_plan/__init__.py`**: Added exports (preserved M106 `load_gpu_profiles`, `get_all_profiles`)

---

## Previous Milestones

The project has completed **105 milestones**, covering the full feature chain from core analysis to advanced optimization.

---

## Known Limitations

1. **Offline analysis only** — Does not support real-time streaming ingestion of production traffic data
2. **Linear scaling assumption** — `plan-capacity` uses a linear model, which may deviate from reality under high concurrency
3. **Single-cluster perspective** — Cross-cluster/cross-region joint optimization is not yet supported
4. **Benchmark format** — Only supports xpyd-bench native format and compatible JSON; direct import from other benchmark tools is not supported

---

## Next Steps

- Explore online/streaming analysis mode
- Cross-cluster joint optimization
- Closed-loop integration with xPyD-proxy auto-tuning
- Web UI dashboard (replacing TUI)
- Richer visualizations (interactive charts)
- Custom GPU profile loading from YAML config

## Iteration History

| # | Date | Task | Result | Reviewer Comments |
|---|------|------|--------|-------------------|
| 1 | 2026-04-06 | M107 Rate Limiter Recommender | 🔄 changes requested | Both bots: __init__.py dropped M106 exports, current.md missing M107 info |
| 2 | 2026-04-06 | M107 fix review comments | ⏳ pending re-review | Restored M106 exports, added M107 iteration record |

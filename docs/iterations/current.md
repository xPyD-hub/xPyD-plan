# xPyD-plan Current Iteration Status

> Last updated: 2026-04-06

---

## Current Milestone: M109 — vLLM Benchmark Command Generator 🔄

Generate ready-to-run vLLM server and benchmark_serving.py commands for P:D ratio exploration.

### Changes
- **`src/xpyd_plan/vllm_import.py`**: `VLLMImporter` class, `VLLMBenchmarkData`, `VLLMRequest`, `ImportResult`, `ImportConfig` models, `import_vllm()` API
- **`src/xpyd_plan/cli/_import.py`**: CLI `import` subcommand
- **`src/xpyd_plan/cli/_main.py`**: Register import subcommand
- **`src/xpyd_plan/__init__.py`**: Added exports (preserved all prior exports)
- **`tests/test_vllm_import.py`**: 32 new tests
- **`ROADMAP.md`**: Added M105, M106, M107, M108 entries

---

## Previous Milestones

The project has completed **107 milestones**, covering the full feature chain from core analysis to advanced optimization.

---

## Known Limitations

1. **Offline analysis only** — Does not support real-time streaming ingestion of production traffic data
2. **Linear scaling assumption** — `plan-capacity` uses a linear model, which may deviate from reality under high concurrency
3. **Single-cluster perspective** — Cross-cluster/cross-region joint optimization is not yet supported
4. **Benchmark format** — Now supports vLLM format import in addition to native format

---

## Next Steps

- Explore online/streaming analysis mode
- Cross-cluster joint optimization
- Closed-loop integration with xPyD-proxy auto-tuning
- Web UI dashboard (replacing TUI)
- Richer visualizations (interactive charts)
- Support additional benchmark tool formats (TensorRT-LLM, SGLang)

## Iteration History

| # | Date | Task | Result | Reviewer Comments |
|---|------|------|--------|-------------------|
| 1 | 2026-04-06 | M107 Rate Limiter Recommender | ✅ merged | PR #238 |
| 2 | 2026-04-06 | M108 vLLM Benchmark Importer | ⏳ pending review | PR #TBD |
| 3 | 2026-04-06 | M109 vLLM Benchmark Command Generator | ⏳ awaiting review | Issue #241, PR #TBD |

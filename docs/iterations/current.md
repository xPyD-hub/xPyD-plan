# xPyD-plan Current Iteration Status

> Last updated: 2026-04-06

---

## Current Milestone: M110 — SGLang Benchmark Format Importer ✅

Import SGLang `bench_serving` output into native xpyd-plan format for downstream analysis.

### What was done

- `SGLangImporter` logic in `sglang_import.py`
- Pydantic models: `SGLangRequest`, `SGLangBenchmarkData`, `SGLangImportConfig`, `SGLangImportResult`
- Auto-detect SGLang format (presence of `itl` as list + `latency` field, absence of `request_latency`)
- Failed request filtering with count in warnings
- TPOT computed as mean of `itl` list, with fallback estimation
- CLI `import` subcommand updated with `--format sglang|auto` support
- Programmatic `import_sglang()` and `import_sglang_data()` APIs
- 23 new tests (2495 total)

### Changes

| File | Change |
|------|--------|
| `src/xpyd_plan/sglang_import.py` | New — SGLang import logic |
| `src/xpyd_plan/cli/_import.py` | Updated — SGLang format support in import CLI |
| `src/xpyd_plan/__init__.py` | Updated — export SGLang symbols |
| `tests/test_sglang_import.py` | New — 23 tests |
| `docs/iterations/current.md` | Updated |

---

## Project Status

The project has completed **110 milestones**, covering the full feature chain from core analysis to advanced optimization.

---

## Known Limitations

1. **Offline analysis only** — Does not support real-time streaming ingestion of production traffic data
2. **Linear scaling assumption** — `plan-capacity` uses a linear model, which may deviate from reality under high concurrency
3. **Single-cluster perspective** — Cross-cluster/cross-region joint optimization is not yet supported
4. **Benchmark format** — Now supports vLLM and SGLang format import in addition to native format

---

## Next Steps

- Explore online/streaming analysis mode
- Cross-cluster joint optimization
- Closed-loop integration with xPyD-proxy auto-tuning
- Web UI dashboard (replacing TUI)
- Richer visualizations (interactive charts)
- Support additional benchmark tool formats (TensorRT-LLM)

## Iteration History

| # | Date | Task | Result | Reviewer Comments |
|---|------|------|--------|-------------------|
| 1 | 2026-04-06 | M107 Rate Limiter Recommender | ✅ merged | PR #238 |
| 2 | 2026-04-06 | M108 vLLM Benchmark Importer | ✅ merged | PR #240 |
| 3 | 2026-04-06 | M109 vLLM Benchmark Command Generator | ✅ merged | PR #242 |
| 4 | 2026-04-06 | M110 SGLang Benchmark Format Importer | ✅ merged | PR #244 |
| 5 | 2026-04-06 | M111 SGLang Benchmark Command Generator | ⏳ pending review | Issue #245 |

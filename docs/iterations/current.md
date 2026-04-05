# xPyD-plan Current Iteration Status

> Last updated: 2026-04-06

---

## Current: M106 — Custom GPU Profile Loading from YAML

### What

Add `load_gpu_profiles(path)` to load custom GPU profiles from YAML files, merging with built-in profiles. Enables users with custom/proprietary GPUs to use fleet calculator, normalizer, and capacity planner.

### Changes

- `gpu_profiles.py`: added `load_gpu_profiles()`, `get_all_profiles()`, updated `get_gpu_profile()` and `list_gpu_profiles()` with optional `custom_profiles_path`
- `__init__.py`: export new functions
- 17 new tests in `test_custom_gpu_profiles.py`

### Status

- [x] Implementation
- [x] Tests (17 passing)
- [x] Lint passing
- [ ] PR created
- [ ] Review

## Current Milestone: M105 — Expand GPU Profile Library ✅

Added 5 new GPU profiles to the built-in library: A100-40G, A10G-24G, L40S-48G, H200-141G, and B200-192G. This addresses the known limitation that only A100/H100 profiles were available.

### Changes
- **`src/xpyd_plan/gpu_profiles.py`**: Added profiles for A100-40G, A10G-24G, L40S-48G, H200-141G, B200-192G (total: 7 profiles, up from 2)
- **`tests/test_gpu_profiles.py`**: Extended test coverage for all new profiles including throughput ordering and cost comparisons

### New GPU Profiles

| Profile | Memory | Prefill tok/s | Decode tok/s | $/hr |
|---------|--------|--------------|-------------|------|
| A10G-24G | 24 GB | 18,000 | 800 | 0.75 |
| A100-40G | 40 GB | 40,000 | 1,600 | 1.80 |
| L40S-48G | 48 GB | 35,000 | 1,400 | 1.50 |
| A100-80G | 80 GB | 50,000 | 2,000 | 2.50 |
| H100-80G | 80 GB | 120,000 | 5,000 | 4.50 |
| H200-141G | 141 GB | 150,000 | 7,500 | 5.50 |
| B200-192G | 192 GB | 250,000 | 12,000 | 8.00 |

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

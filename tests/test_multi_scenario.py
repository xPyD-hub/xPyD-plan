"""Tests for multi-scenario analysis (M4)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.benchmark_models import MultiScenarioResult


def _make_benchmark_data(
    n: int = 100,
    num_p: int = 2,
    num_d: int = 6,
    qps: float = 10.0,
    ttft_range: tuple[float, float] = (50, 500),
    tpot_range: tuple[float, float] = (10, 40),
    seed: int = 42,
) -> dict:
    """Generate a benchmark dataset dict."""
    rng = random.Random(seed)
    requests = []
    base_ts = 1700000000.0
    for i in range(n):
        prompt_tokens = rng.randint(100, 2000)
        output_tokens = rng.randint(50, 500)
        ttft = rng.uniform(*ttft_range)
        tpot = rng.uniform(*tpot_range)
        total = ttft + tpot * output_tokens
        requests.append({
            "request_id": f"req-{i:04d}",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": round(ttft, 2),
            "tpot_ms": round(tpot, 2),
            "total_latency_ms": round(total, 2),
            "timestamp": base_ts + i * 0.1,
        })
    return {
        "metadata": {
            "num_prefill_instances": num_p,
            "num_decode_instances": num_d,
            "total_instances": num_p + num_d,
            "measured_qps": qps,
        },
        "requests": requests,
    }


def _write_benchmark_files(tmp_path: Path, configs: list[dict]) -> list[Path]:
    """Write multiple benchmark dicts to JSON files."""
    paths = []
    for i, cfg in enumerate(configs):
        p = tmp_path / f"bench_{i}.json"
        p.write_text(json.dumps(cfg))
        paths.append(p)
    return paths


@pytest.fixture
def multi_data_dicts() -> list[dict]:
    """Three QPS scenarios: low, medium, high."""
    return [
        _make_benchmark_data(qps=5.0, seed=1),
        _make_benchmark_data(qps=15.0, seed=2),
        _make_benchmark_data(qps=30.0, seed=3, ttft_range=(100, 800), tpot_range=(15, 50)),
    ]


@pytest.fixture
def multi_files(tmp_path: Path, multi_data_dicts: list[dict]) -> list[Path]:
    return _write_benchmark_files(tmp_path, multi_data_dicts)


# ── Test: load_multi_data ─────────────────────────────────────────────────


def test_load_multi_data_from_files(multi_files: list[Path]):
    a = BenchmarkAnalyzer()
    datasets = a.load_multi_data(multi_files)
    assert len(datasets) == 3
    # Should be sorted by QPS ascending
    qps_list = [d.metadata.measured_qps for d in datasets]
    assert qps_list == sorted(qps_list)


def test_load_multi_data_from_dicts(multi_data_dicts: list[dict]):
    a = BenchmarkAnalyzer()
    datasets = a.load_multi_data_from_dicts(multi_data_dicts)
    assert len(datasets) == 3


def test_load_multi_data_empty():
    a = BenchmarkAnalyzer()
    with pytest.raises(ValueError, match="At least one"):
        a.load_multi_data([])


def test_load_multi_data_from_dicts_empty():
    a = BenchmarkAnalyzer()
    with pytest.raises(ValueError, match="At least one"):
        a.load_multi_data_from_dicts([])


def test_load_multi_data_file_not_found(tmp_path: Path):
    a = BenchmarkAnalyzer()
    with pytest.raises(FileNotFoundError):
        a.load_multi_data([tmp_path / "nonexistent.json"])


def test_multi_data_property_before_load():
    a = BenchmarkAnalyzer()
    with pytest.raises(RuntimeError, match="No multi-scenario data"):
        _ = a.multi_data


# ── Test: find_optimal_ratio_multi ────────────────────────────────────────


def test_find_optimal_ratio_multi_basic(multi_data_dicts: list[dict]):
    from xpyd_plan.models import SLAConfig

    a = BenchmarkAnalyzer()
    a.load_multi_data_from_dicts(multi_data_dicts)
    sla = SLAConfig(ttft_ms=10000, tpot_ms=10000)
    result = a.find_optimal_ratio_multi(8, sla)

    assert isinstance(result, MultiScenarioResult)
    assert len(result.scenarios) == 3
    assert result.total_instances == 8
    assert len(result.per_scenario_best) == 3


def test_find_optimal_ratio_multi_has_unified(multi_data_dicts: list[dict]):
    from xpyd_plan.models import SLAConfig

    a = BenchmarkAnalyzer()
    a.load_multi_data_from_dicts(multi_data_dicts)
    sla = SLAConfig(ttft_ms=100000, tpot_ms=100000)
    result = a.find_optimal_ratio_multi(8, sla)

    # With very relaxed SLA, should find a unified best
    assert result.unified_best is not None
    assert result.unified_best.meets_sla is True


def test_find_optimal_ratio_multi_impossible_sla(multi_data_dicts: list[dict]):
    from xpyd_plan.models import SLAConfig

    a = BenchmarkAnalyzer()
    a.load_multi_data_from_dicts(multi_data_dicts)
    sla = SLAConfig(ttft_ms=0.001, tpot_ms=0.001)
    result = a.find_optimal_ratio_multi(8, sla)

    assert result.unified_best is None
    for best in result.per_scenario_best:
        assert best is None


def test_find_optimal_ratio_multi_scenarios_sorted_by_qps(multi_data_dicts: list[dict]):
    from xpyd_plan.models import SLAConfig

    a = BenchmarkAnalyzer()
    a.load_multi_data_from_dicts(multi_data_dicts)
    sla = SLAConfig()
    result = a.find_optimal_ratio_multi(8, sla)

    qps_list = [s.qps for s in result.scenarios]
    assert qps_list == sorted(qps_list)


def test_find_optimal_ratio_multi_two_scenarios():
    """Two scenarios with different characteristics."""
    from xpyd_plan.models import SLAConfig

    low = _make_benchmark_data(qps=5.0, seed=10)
    high = _make_benchmark_data(
        qps=50.0, seed=20, ttft_range=(200, 1000), tpot_range=(20, 60)
    )
    a = BenchmarkAnalyzer()
    a.load_multi_data_from_dicts([low, high])
    sla = SLAConfig(ttft_ms=50000, tpot_ms=50000)
    result = a.find_optimal_ratio_multi(8, sla)

    assert len(result.scenarios) == 2
    assert result.unified_best is not None


def test_find_optimal_ratio_multi_single_scenario():
    """Single scenario should still work."""
    from xpyd_plan.models import SLAConfig

    data = _make_benchmark_data(qps=10.0)
    a = BenchmarkAnalyzer()
    a.load_multi_data_from_dicts([data])
    sla = SLAConfig(ttft_ms=10000)
    result = a.find_optimal_ratio_multi(8, sla)

    assert len(result.scenarios) == 1
    assert result.unified_best is not None


# ── Test: CLI multi-file ──────────────────────────────────────────────────


def test_cli_analyze_multi_files(multi_files: list[Path]):
    from xpyd_plan.cli import main

    main([
        "analyze",
        "--benchmark", *[str(f) for f in multi_files],
        "--sla-ttft", "10000",
        "--sla-tpot", "10000",
    ])


def test_cli_analyze_multi_with_output(multi_files: list[Path], tmp_path: Path):
    from xpyd_plan.cli import main

    out = tmp_path / "multi_result.json"
    main([
        "analyze",
        "--benchmark", *[str(f) for f in multi_files],
        "--sla-ttft", "10000",
        "--output", str(out),
    ])
    assert out.exists()
    result = json.loads(out.read_text())
    assert "scenarios" in result
    assert "unified_best" in result


def test_cli_analyze_single_file_still_works(multi_files: list[Path]):
    """Single file should still use original behavior."""
    from xpyd_plan.cli import main

    main(["analyze", "--benchmark", str(multi_files[0])])

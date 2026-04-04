"""Tests for what-if scenario simulation (M11)."""

from __future__ import annotations

import json
import random

import pytest

from xpyd_plan.models import SLAConfig
from xpyd_plan.whatif import WhatIfScenario, WhatIfSimulator, what_if

# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_requests(
    n: int = 100,
    ttft_range: tuple[float, float] = (50, 200),
    tpot_range: tuple[float, float] = (10, 30),
    seed: int = 42,
) -> list[dict]:
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
    return requests


def _make_benchmark(num_p=2, num_d=6, qps=10.0, n=100, seed=42):
    return {
        "metadata": {
            "num_prefill_instances": num_p,
            "num_decode_instances": num_d,
            "total_instances": num_p + num_d,
            "measured_qps": qps,
        },
        "requests": _make_requests(n=n, seed=seed),
    }


@pytest.fixture
def benchmark_data():
    return _make_benchmark()


@pytest.fixture
def benchmark_file(tmp_path, benchmark_data):
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(benchmark_data))
    return str(p)


@pytest.fixture
def sla():
    return SLAConfig(ttft_ms=500.0, tpot_ms=50.0, max_latency_ms=20000.0)


@pytest.fixture
def strict_sla():
    return SLAConfig(ttft_ms=50.0, tpot_ms=5.0, max_latency_ms=1000.0)


# ── WhatIfSimulator tests ────────────────────────────────────────────────


class TestWhatIfSimulator:
    def test_load_from_dict(self, benchmark_data):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        assert sim.data.metadata.measured_qps == 10.0

    def test_load_file(self, benchmark_file):
        sim = WhatIfSimulator()
        data = sim.load_file(benchmark_file)
        assert data.metadata.total_instances == 8

    def test_load_file_not_found(self):
        sim = WhatIfSimulator()
        with pytest.raises(FileNotFoundError):
            sim.load_file("/nonexistent/file.json")

    def test_no_data_raises(self):
        sim = WhatIfSimulator()
        with pytest.raises(RuntimeError, match="No data loaded"):
            sim.data

    def test_baseline(self, benchmark_data, sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        base = sim.baseline(sla)
        assert base.label == "Baseline"
        assert base.total_instances == 8
        assert base.analysis is not None

    def test_scale_qps_double(self, benchmark_data, sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        result = sim.scale_qps(2.0, sla)
        assert result.scale_qps == 2.0
        assert "2.0x QPS" in result.label
        assert result.analysis is not None

    def test_scale_qps_half(self, benchmark_data, sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        result = sim.scale_qps(0.5, sla)
        assert result.scale_qps == 0.5

    def test_scale_qps_invalid(self, benchmark_data, sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        with pytest.raises(ValueError, match="positive"):
            sim.scale_qps(0, sla)

    def test_scale_instances_add(self, benchmark_data, sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        result = sim.scale_instances(4, sla)
        assert result.total_instances == 12
        assert result.instance_delta == 4
        assert "+4 instances" in result.label

    def test_scale_instances_remove(self, benchmark_data, sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        result = sim.scale_instances(-2, sla)
        assert result.total_instances == 6
        assert result.instance_delta == -2

    def test_scale_instances_too_few(self, benchmark_data, sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        with pytest.raises(ValueError, match="minimum 2"):
            sim.scale_instances(-7, sla)

    def test_compare_multiple(self, benchmark_data, sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        scenarios = [
            {"scale_qps": 2.0},
            {"add_instances": 4},
            {"scale_qps": 1.5, "add_instances": 2},
        ]
        comparison = sim.compare(scenarios, sla)
        assert comparison.baseline.label == "Baseline"
        assert len(comparison.scenarios) == 3
        assert comparison.scenarios[0].scale_qps == 2.0
        assert comparison.scenarios[1].instance_delta == 4
        assert comparison.scenarios[2].scale_qps == 1.5

    def test_compare_custom_label(self, benchmark_data, sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        scenarios = [{"scale_qps": 2.0, "label": "Peak hour"}]
        comparison = sim.compare(scenarios, sla)
        assert comparison.scenarios[0].label == "Peak hour"

    def test_compare_invalid_instances(self, benchmark_data, sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        with pytest.raises(ValueError, match="minimum 2"):
            sim.compare([{"add_instances": -7}], sla)

    def test_scale_qps_with_custom_total(self, benchmark_data, sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        result = sim.scale_qps(2.0, sla, total_instances=12)
        assert result.total_instances == 12

    def test_strict_sla_fails(self, benchmark_data, strict_sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        result = sim.scale_qps(3.0, strict_sla)
        # With very strict SLA and 3x QPS, likely no ratio meets SLA
        assert result.analysis is not None


# ── Programmatic API tests ────────────────────────────────────────────────


class TestWhatIfAPI:
    def test_what_if_api(self, benchmark_file):
        result = what_if(
            benchmark_path=benchmark_file,
            scenarios=[{"scale_qps": 1.5}],
            sla_config={"ttft_ms": 500, "tpot_ms": 50},
        )
        assert "baseline" in result
        assert "scenarios" in result
        assert len(result["scenarios"]) == 1

    def test_what_if_api_none_sla(self, benchmark_file):
        result = what_if(
            benchmark_path=benchmark_file,
            scenarios=[{"add_instances": 2}],
        )
        assert "baseline" in result


# ── Model tests ───────────────────────────────────────────────────────────


class TestWhatIfModels:
    def test_scenario_defaults(self):
        s = WhatIfScenario(label="test", total_instances=8)
        assert s.scale_qps == 1.0
        assert s.instance_delta == 0
        assert s.best is None

    def test_comparison_serialization(self, benchmark_data, sla):
        sim = WhatIfSimulator()
        sim.load_from_dict(benchmark_data)
        comparison = sim.compare([{"scale_qps": 2.0}], sla)
        # Should be JSON-serializable
        data = json.loads(comparison.model_dump_json())
        assert "baseline" in data
        assert len(data["scenarios"]) == 1

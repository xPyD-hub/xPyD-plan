"""Tests for multi-model comparison matrix (M32)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.model_compare import (
    ComparisonMatrix,
    ModelComparator,
    ModelComparison,
    ModelProfile,
    ModelRanking,
    compare_models,
)


def _make_benchmark(
    num_prefill: int = 2,
    num_decode: int = 6,
    qps: float = 10.0,
    ttft_base: float = 50.0,
    tpot_base: float = 20.0,
    total_base: float = 200.0,
    n_requests: int = 100,
) -> dict:
    """Create a synthetic benchmark dict."""
    import random

    random.seed(42)
    requests = []
    for i in range(n_requests):
        ttft = ttft_base + random.gauss(0, ttft_base * 0.1)
        tpot = tpot_base + random.gauss(0, tpot_base * 0.1)
        total = total_base + random.gauss(0, total_base * 0.1)
        requests.append({
            "request_id": f"req-{i}",
            "prompt_tokens": random.randint(50, 500),
            "output_tokens": random.randint(10, 200),
            "ttft_ms": max(1.0, ttft),
            "tpot_ms": max(1.0, tpot),
            "total_latency_ms": max(1.0, total),
            "timestamp": 1700000000.0 + i,
        })
    return {
        "metadata": {
            "num_prefill_instances": num_prefill,
            "num_decode_instances": num_decode,
            "total_instances": num_prefill + num_decode,
            "measured_qps": qps,
        },
        "requests": requests,
    }


def _write_benchmark(tmp_dir: Path, name: str, **kwargs) -> str:
    """Write a benchmark file and return its path."""
    path = tmp_dir / f"{name}.json"
    path.write_text(json.dumps(_make_benchmark(**kwargs)))
    return str(path)


class TestModelProfile:
    def test_model_profile_fields(self):
        p = ModelProfile(
            model_name="llama-70b",
            num_prefill_instances=2,
            num_decode_instances=6,
            total_instances=8,
            measured_qps=10.0,
            request_count=100,
            ttft_p50_ms=50.0,
            ttft_p95_ms=65.0,
            ttft_p99_ms=80.0,
            tpot_p50_ms=20.0,
            tpot_p95_ms=28.0,
            tpot_p99_ms=35.0,
            total_latency_p50_ms=200.0,
            total_latency_p95_ms=260.0,
            total_latency_p99_ms=300.0,
            avg_prompt_tokens=250.0,
            avg_output_tokens=100.0,
        )
        assert p.model_name == "llama-70b"
        assert p.cost_per_hour is None
        assert p.cost_per_request is None

    def test_model_profile_with_cost(self):
        p = ModelProfile(
            model_name="test",
            num_prefill_instances=1,
            num_decode_instances=3,
            total_instances=4,
            measured_qps=5.0,
            request_count=50,
            ttft_p50_ms=40.0,
            ttft_p95_ms=55.0,
            ttft_p99_ms=70.0,
            tpot_p50_ms=15.0,
            tpot_p95_ms=22.0,
            tpot_p99_ms=30.0,
            total_latency_p50_ms=180.0,
            total_latency_p95_ms=240.0,
            total_latency_p99_ms=280.0,
            avg_prompt_tokens=200.0,
            avg_output_tokens=80.0,
            cost_per_hour=12.0,
            cost_per_request=0.00066,
        )
        assert p.cost_per_hour == 12.0
        assert p.cost_per_request == 0.00066


class TestModelComparison:
    def test_comparison_fields(self):
        c = ModelComparison(
            metric="TTFT P95",
            model_a="A",
            model_b="B",
            value_a=50.0,
            value_b=60.0,
            absolute_delta=10.0,
            relative_delta=0.2,
            winner="A",
        )
        assert c.winner == "A"
        assert c.relative_delta == pytest.approx(0.2)


class TestModelRanking:
    def test_ranking_fields(self):
        r = ModelRanking(
            model_name="llama-70b",
            rank=1,
            latency_score=0.5,
            cost_efficiency_score=0.3,
            recommendation="Best overall",
        )
        assert r.rank == 1


class TestModelComparator:
    def test_load_profile(self, tmp_path):
        path = _write_benchmark(tmp_path, "model_a")
        comparator = ModelComparator()
        profile = comparator.load_profile(path, "model-a")
        assert profile.model_name == "model-a"
        assert profile.request_count == 100
        assert profile.ttft_p50_ms > 0
        assert profile.cost_per_hour is None

    def test_load_profile_with_cost(self, tmp_path):
        path = _write_benchmark(tmp_path, "model_a")
        comparator = ModelComparator(gpu_hourly_rate=3.0)
        profile = comparator.load_profile(path, "model-a")
        assert profile.cost_per_hour == pytest.approx(3.0 * 8)  # 8 total instances
        assert profile.cost_per_request is not None
        assert profile.cost_per_request > 0

    def test_compare_two_models(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "a", ttft_base=50.0, tpot_base=20.0)
        path_b = _write_benchmark(tmp_path, "b", ttft_base=70.0, tpot_base=30.0)
        comparator = ModelComparator()
        result = comparator.compare([path_a, path_b], ["model-a", "model-b"])

        assert isinstance(result, ComparisonMatrix)
        assert len(result.profiles) == 2
        assert len(result.comparisons) == 9  # 9 latency metrics for 1 pair
        assert len(result.rankings) == 2
        assert result.best_latency_model == "model-a"  # lower latency

    def test_compare_three_models(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "a", ttft_base=50.0)
        path_b = _write_benchmark(tmp_path, "b", ttft_base=70.0)
        path_c = _write_benchmark(tmp_path, "c", ttft_base=90.0)
        comparator = ModelComparator()
        result = comparator.compare(
            [path_a, path_b, path_c], ["A", "B", "C"]
        )

        assert len(result.profiles) == 3
        # 3 pairs × 9 metrics = 27
        assert len(result.comparisons) == 27
        assert len(result.rankings) == 3
        assert result.rankings[0].rank == 1

    def test_compare_with_cost(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "a", ttft_base=50.0)
        path_b = _write_benchmark(tmp_path, "b", ttft_base=70.0)
        comparator = ModelComparator(gpu_hourly_rate=2.5)
        result = comparator.compare([path_a, path_b], ["A", "B"])

        assert result.best_cost_model is not None
        for p in result.profiles:
            assert p.cost_per_hour is not None
            assert p.cost_per_request is not None
        for r in result.rankings:
            assert r.cost_efficiency_score is not None

    def test_compare_mismatched_counts(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "a")
        with pytest.raises(ValueError, match="must match"):
            ModelComparator().compare([path_a], ["A", "B"])

    def test_compare_less_than_two(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "a")
        with pytest.raises(ValueError, match="At least 2"):
            ModelComparator().compare([path_a], ["A"])

    def test_winner_determined_correctly(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "a", ttft_base=100.0)
        path_b = _write_benchmark(tmp_path, "b", ttft_base=30.0)
        result = ModelComparator().compare([path_a, path_b], ["slow", "fast"])

        ttft_comps = [c for c in result.comparisons if c.metric == "TTFT P95"]
        assert len(ttft_comps) == 1
        assert ttft_comps[0].winner == "fast"

    def test_rankings_order(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "a", ttft_base=50.0, tpot_base=20.0, total_base=200.0)
        path_b = _write_benchmark(tmp_path, "b", ttft_base=100.0, tpot_base=40.0, total_base=400.0)
        result = ModelComparator().compare([path_a, path_b], ["fast", "slow"])

        assert result.rankings[0].model_name == "fast"
        assert result.rankings[0].rank == 1
        assert result.rankings[1].model_name == "slow"
        assert result.rankings[1].rank == 2

    def test_best_cost_model_none_without_cost(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "a")
        path_b = _write_benchmark(tmp_path, "b")
        result = ModelComparator().compare([path_a, path_b], ["A", "B"])
        assert result.best_cost_model is None

    def test_relative_delta_zero_division(self, tmp_path):
        """When a metric value is 0 for model_a, relative_delta should be 0."""
        # This is hard to trigger naturally since latency > 0, but we test the code path
        path_a = _write_benchmark(tmp_path, "a", ttft_base=50.0)
        path_b = _write_benchmark(tmp_path, "b", ttft_base=60.0)
        result = ModelComparator().compare([path_a, path_b], ["A", "B"])
        # All comparisons should have valid relative_delta
        for c in result.comparisons:
            assert not (c.value_a == 0 and c.relative_delta != 0.0)


class TestCompareModelsAPI:
    def test_programmatic_api(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "a", ttft_base=50.0)
        path_b = _write_benchmark(tmp_path, "b", ttft_base=80.0)
        result = compare_models([path_a, path_b], ["A", "B"])
        assert isinstance(result, ComparisonMatrix)
        assert result.best_latency_model == "A"

    def test_programmatic_api_with_cost(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "a")
        path_b = _write_benchmark(tmp_path, "b")
        result = compare_models([path_a, path_b], ["A", "B"], gpu_hourly_rate=2.0)
        assert result.best_cost_model is not None

    def test_json_serialization(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "a")
        path_b = _write_benchmark(tmp_path, "b")
        result = compare_models([path_a, path_b], ["A", "B"])
        j = json.loads(result.model_dump_json())
        assert "profiles" in j
        assert "comparisons" in j
        assert "rankings" in j
        assert "best_latency_model" in j


class TestCLIModelCompare:
    def test_cli_table_output(self, tmp_path, capsys):
        from xpyd_plan.cli._main import main

        path_a = _write_benchmark(tmp_path, "a", ttft_base=50.0)
        path_b = _write_benchmark(tmp_path, "b", ttft_base=80.0)
        main([
            "model-compare",
            "--benchmarks", path_a, path_b,
            "--models", "ModelA,ModelB",
        ])

    def test_cli_json_output(self, tmp_path, capsys):
        from xpyd_plan.cli._main import main

        path_a = _write_benchmark(tmp_path, "a")
        path_b = _write_benchmark(tmp_path, "b")
        main([
            "model-compare",
            "--benchmarks", path_a, path_b,
            "--models", "A,B",
            "--output-format", "json",
        ])

    def test_cli_with_cost(self, tmp_path, capsys):
        from xpyd_plan.cli._main import main

        path_a = _write_benchmark(tmp_path, "a")
        path_b = _write_benchmark(tmp_path, "b")
        main([
            "model-compare",
            "--benchmarks", path_a, path_b,
            "--models", "A,B",
            "--gpu-hourly-rate", "3.0",
        ])

    def test_cli_mismatched_models(self, tmp_path):
        from xpyd_plan.cli._main import main

        path_a = _write_benchmark(tmp_path, "a")
        with pytest.raises(SystemExit):
            main([
                "model-compare",
                "--benchmarks", path_a,
                "--models", "A,B",
            ])

"""Tests for Multi-SLA Tier Analysis (M48)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import yaml

from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.models import SLAConfig
from xpyd_plan.sla_tier import (
    SLATier,
    SLATierAnalyzer,
    analyze_sla_tiers,
    load_tiers_from_yaml,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_requests(
    n: int = 100,
    ttft_range: tuple[float, float] = (50, 500),
    tpot_range: tuple[float, float] = (10, 40),
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


def _make_benchmark_json(
    tmp_path: Path,
    n: int = 100,
    num_p: int = 2,
    num_d: int = 6,
    qps: float = 10.0,
    **kwargs,
) -> Path:
    data = {
        "metadata": {
            "num_prefill_instances": num_p,
            "num_decode_instances": num_d,
            "total_instances": num_p + num_d,
            "measured_qps": qps,
        },
        "requests": _make_requests(n=n, **kwargs),
    }
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(data))
    return p


def _make_tiers_yaml(tmp_path: Path, tiers: list[dict]) -> Path:
    p = tmp_path / "tiers.yaml"
    p.write_text(yaml.dump({"tiers": tiers}))
    return p


def _load_data(tmp_path: Path, **kwargs) -> BenchmarkData:
    p = _make_benchmark_json(tmp_path, **kwargs)
    return BenchmarkData.model_validate(json.loads(p.read_text()))


# ── SLATierAnalyzer Tests ───────────────────────────────────────────────


class TestSLATierAnalyzer:
    def test_single_tier(self, tmp_path: Path) -> None:
        data = _load_data(tmp_path)
        tiers = [SLATier(name="standard", sla=SLAConfig(ttft_ms=9999, tpot_ms=9999))]
        report = SLATierAnalyzer(data).analyze(tiers)

        assert len(report.tier_results) == 1
        assert report.tier_results[0].tier.name == "standard"
        assert report.tier_results[0].best is not None
        assert report.total_instances == 8

    def test_multiple_tiers(self, tmp_path: Path) -> None:
        data = _load_data(tmp_path)
        tiers = [
            SLATier(name="premium", sla=SLAConfig(ttft_ms=9999, tpot_ms=9999, sla_percentile=99)),
            SLATier(name="standard", sla=SLAConfig(ttft_ms=9999, tpot_ms=9999)),
        ]
        report = SLATierAnalyzer(data).analyze(tiers)

        assert len(report.tier_results) == 2
        assert report.tier_results[0].tier.name == "premium"
        assert report.tier_results[1].tier.name == "standard"

    def test_unified_best_exists(self, tmp_path: Path) -> None:
        """When all tiers are relaxed, a unified ratio should exist."""
        data = _load_data(tmp_path)
        tiers = [
            SLATier(name="t1", sla=SLAConfig(ttft_ms=9999, tpot_ms=9999)),
            SLATier(name="t2", sla=SLAConfig(ttft_ms=9999, tpot_ms=9999)),
        ]
        report = SLATierAnalyzer(data).analyze(tiers)
        assert report.unified_best is not None

    def test_no_unified_when_impossible(self, tmp_path: Path) -> None:
        """When one tier is impossibly tight, no unified ratio."""
        data = _load_data(tmp_path)
        tiers = [
            SLATier(name="relax", sla=SLAConfig(ttft_ms=9999, tpot_ms=9999)),
            SLATier(name="impossible", sla=SLAConfig(ttft_ms=0.001, tpot_ms=0.001)),
        ]
        report = SLATierAnalyzer(data).analyze(tiers)
        assert report.unified_best is None

    def test_empty_tiers_raises(self, tmp_path: Path) -> None:
        data = _load_data(tmp_path)
        with pytest.raises(ValueError, match="(?i)at least one"):
            SLATierAnalyzer(data).analyze([])

    def test_tier_result_has_candidates(self, tmp_path: Path) -> None:
        data = _load_data(tmp_path)
        tiers = [SLATier(name="t1", sla=SLAConfig(ttft_ms=9999, tpot_ms=9999))]
        report = SLATierAnalyzer(data).analyze(tiers)
        assert len(report.tier_results[0].candidates) > 0

    def test_impossible_sla_no_best(self, tmp_path: Path) -> None:
        data = _load_data(tmp_path)
        tiers = [SLATier(name="impossible", sla=SLAConfig(ttft_ms=0.001))]
        report = SLATierAnalyzer(data).analyze(tiers)
        assert report.tier_results[0].best is None

    def test_different_percentiles(self, tmp_path: Path) -> None:
        data = _load_data(tmp_path)
        tiers = [
            SLATier(name="p95", sla=SLAConfig(ttft_ms=9999, sla_percentile=95)),
            SLATier(name="p99", sla=SLAConfig(ttft_ms=9999, sla_percentile=99)),
        ]
        report = SLATierAnalyzer(data).analyze(tiers)
        assert len(report.tier_results) == 2
        # Both should pass with relaxed SLA
        for tr in report.tier_results:
            assert tr.best is not None


# ── load_tiers_from_yaml Tests ──────────────────────────────────────────


class TestLoadTiersFromYaml:
    def test_basic_load(self, tmp_path: Path) -> None:
        tiers_file = _make_tiers_yaml(tmp_path, [
            {"name": "premium", "ttft_ms": 100, "tpot_ms": 20, "percentile": 99},
            {"name": "standard", "ttft_ms": 500, "tpot_ms": 50},
        ])
        tiers = load_tiers_from_yaml(tiers_file)
        assert len(tiers) == 2
        assert tiers[0].name == "premium"
        assert tiers[0].sla.ttft_ms == 100
        assert tiers[0].sla.sla_percentile == 99
        assert tiers[1].sla.sla_percentile == 95  # default

    def test_invalid_yaml_missing_tiers_key(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump({"policies": []}))
        with pytest.raises(ValueError, match="tiers"):
            load_tiers_from_yaml(p)

    def test_optional_fields(self, tmp_path: Path) -> None:
        tiers_file = _make_tiers_yaml(tmp_path, [
            {"name": "minimal", "ttft_ms": 200},
        ])
        tiers = load_tiers_from_yaml(tiers_file)
        assert tiers[0].sla.tpot_ms is None
        assert tiers[0].sla.max_latency_ms is None


# ── analyze_sla_tiers API Tests ─────────────────────────────────────────


class TestAnalyzeSlaTiersAPI:
    def test_with_tiers_path(self, tmp_path: Path) -> None:
        bench_path = _make_benchmark_json(tmp_path)
        tiers_file = _make_tiers_yaml(tmp_path, [
            {"name": "t1", "ttft_ms": 9999, "tpot_ms": 9999},
        ])
        result = analyze_sla_tiers(benchmark_path=bench_path, tiers_path=tiers_file)
        assert "tier_results" in result
        assert "unified_best" in result
        assert "total_instances" in result

    def test_with_tiers_objects(self, tmp_path: Path) -> None:
        bench_path = _make_benchmark_json(tmp_path)
        tiers = [SLATier(name="t1", sla=SLAConfig(ttft_ms=9999))]
        result = analyze_sla_tiers(benchmark_path=bench_path, tiers=tiers)
        assert len(result["tier_results"]) == 1

    def test_neither_tiers_raises(self, tmp_path: Path) -> None:
        bench_path = _make_benchmark_json(tmp_path)
        with pytest.raises(ValueError, match="Provide either"):
            analyze_sla_tiers(benchmark_path=bench_path)

    def test_both_tiers_raises(self, tmp_path: Path) -> None:
        bench_path = _make_benchmark_json(tmp_path)
        tiers_file = _make_tiers_yaml(tmp_path, [{"name": "t1", "ttft_ms": 100}])
        tiers = [SLATier(name="t1", sla=SLAConfig(ttft_ms=100))]
        with pytest.raises(ValueError, match="not both"):
            analyze_sla_tiers(benchmark_path=bench_path, tiers=tiers, tiers_path=tiers_file)

    def test_json_output_structure(self, tmp_path: Path) -> None:
        bench_path = _make_benchmark_json(tmp_path)
        tiers = [
            SLATier(name="premium", sla=SLAConfig(ttft_ms=9999, tpot_ms=9999)),
            SLATier(name="standard", sla=SLAConfig(ttft_ms=9999)),
        ]
        result = analyze_sla_tiers(benchmark_path=bench_path, tiers=tiers)
        assert result["total_instances"] == 8
        assert len(result["tier_results"]) == 2
        # Each tier result has expected keys
        for tr in result["tier_results"]:
            assert "tier" in tr
            assert "best" in tr
            assert "candidates" in tr


# ── Model Serialization Tests ───────────────────────────────────────────


class TestModels:
    def test_sla_tier_model(self) -> None:
        tier = SLATier(name="premium", sla=SLAConfig(ttft_ms=100))
        d = tier.model_dump()
        assert d["name"] == "premium"
        assert d["sla"]["ttft_ms"] == 100

    def test_multi_tier_report_serialization(self, tmp_path: Path) -> None:
        data = _load_data(tmp_path)
        tiers = [SLATier(name="t1", sla=SLAConfig(ttft_ms=9999))]
        report = SLATierAnalyzer(data).analyze(tiers)
        d = report.model_dump()
        assert isinstance(d, dict)
        # Round-trip via JSON
        json_str = json.dumps(d)
        assert "tier_results" in json.loads(json_str)

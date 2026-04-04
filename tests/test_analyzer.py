"""Tests for BenchmarkAnalyzer — measured-data analysis."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.benchmark_models import (
    AnalysisResult,
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
    RatioCandidate,
    SLACheck,
    UtilizationResult,
)
from xpyd_plan.models import SLAConfig

# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_requests(
    n: int = 100,
    ttft_range: tuple[float, float] = (50, 500),
    tpot_range: tuple[float, float] = (10, 40),
    prompt_range: tuple[int, int] = (100, 2000),
    output_range: tuple[int, int] = (50, 500),
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic benchmark request records."""
    rng = random.Random(seed)
    requests = []
    base_ts = 1700000000.0
    for i in range(n):
        prompt_tokens = rng.randint(*prompt_range)
        output_tokens = rng.randint(*output_range)
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


def _make_benchmark_data(
    n: int = 100,
    num_p: int = 2,
    num_d: int = 6,
    qps: float = 10.0,
    **kwargs,
) -> dict:
    """Generate a complete benchmark data dict."""
    return {
        "metadata": {
            "num_prefill_instances": num_p,
            "num_decode_instances": num_d,
            "total_instances": num_p + num_d,
            "measured_qps": qps,
        },
        "requests": _make_requests(n=n, **kwargs),
    }


@pytest.fixture
def benchmark_data() -> dict:
    return _make_benchmark_data()


@pytest.fixture
def benchmark_file(benchmark_data: dict, tmp_path: Path) -> Path:
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(benchmark_data))
    return p


@pytest.fixture
def analyzer(benchmark_file: Path) -> BenchmarkAnalyzer:
    a = BenchmarkAnalyzer()
    a.load_data(benchmark_file)
    return a


# ── Test: load_data ───────────────────────────────────────────────────────


def test_load_data_from_file(benchmark_file: Path):
    a = BenchmarkAnalyzer()
    data = a.load_data(benchmark_file)
    assert isinstance(data, BenchmarkData)
    assert len(data.requests) == 100


def test_load_data_from_dict(benchmark_data: dict):
    a = BenchmarkAnalyzer()
    data = a.load_data_from_dict(benchmark_data)
    assert len(data.requests) == 100
    assert data.metadata.num_prefill_instances == 2


def test_load_data_file_not_found():
    a = BenchmarkAnalyzer()
    with pytest.raises(FileNotFoundError):
        a.load_data("/nonexistent/path.json")


def test_load_data_invalid_json(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    a = BenchmarkAnalyzer()
    with pytest.raises(Exception):
        a.load_data(p)


def test_load_data_missing_fields(tmp_path: Path):
    p = tmp_path / "incomplete.json"
    p.write_text(json.dumps({"metadata": {"num_prefill_instances": 1}}))
    a = BenchmarkAnalyzer()
    with pytest.raises(Exception):
        a.load_data(p)


def test_data_property_before_load():
    a = BenchmarkAnalyzer()
    with pytest.raises(RuntimeError, match="No benchmark data loaded"):
        _ = a.data


# ── Test: check_sla ──────────────────────────────────────────────────────


def test_check_sla_all_pass(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig(ttft_ms=10000, tpot_ms=10000)
    result = analyzer.check_sla(sla)
    assert isinstance(result, SLACheck)
    assert result.meets_all is True


def test_check_sla_ttft_fail(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig(ttft_ms=1.0)  # impossibly tight
    result = analyzer.check_sla(sla)
    assert result.meets_ttft is False
    assert result.meets_all is False


def test_check_sla_tpot_fail(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig(tpot_ms=0.1)
    result = analyzer.check_sla(sla)
    assert result.meets_tpot is False


def test_check_sla_no_constraints(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig()
    result = analyzer.check_sla(sla)
    assert result.meets_all is True


def test_check_sla_max_latency_fail(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig(max_latency_ms=1.0)
    result = analyzer.check_sla(sla)
    assert result.meets_total_latency is False


def test_check_sla_percentiles_are_positive(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig()
    result = analyzer.check_sla(sla)
    assert result.ttft_p95_ms > 0
    assert result.ttft_p99_ms >= result.ttft_p95_ms
    assert result.tpot_p95_ms > 0
    assert result.tpot_p99_ms >= result.tpot_p95_ms


# ── Test: compute_utilization ─────────────────────────────────────────────


def test_compute_utilization(analyzer: BenchmarkAnalyzer):
    result = analyzer.compute_utilization()
    assert isinstance(result, UtilizationResult)
    assert 0 <= result.prefill_utilization <= 1
    assert 0 <= result.decode_utilization <= 1
    assert 0 <= result.waste_rate <= 1


def test_utilization_waste_rate(analyzer: BenchmarkAnalyzer):
    result = analyzer.compute_utilization()
    expected_waste = 1.0 - min(result.prefill_utilization, result.decode_utilization)
    assert abs(result.waste_rate - expected_waste) < 0.01


# ── Test: find_optimal_ratio ──────────────────────────────────────────────


def test_find_optimal_ratio_basic(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig(ttft_ms=10000, tpot_ms=10000)
    result = analyzer.find_optimal_ratio(8, sla)
    assert isinstance(result, AnalysisResult)
    assert result.best is not None
    assert result.best.meets_sla is True
    assert len(result.candidates) == 7  # 1P:7D through 7P:1D


def test_find_optimal_ratio_impossible_sla(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig(ttft_ms=0.001, tpot_ms=0.001)
    result = analyzer.find_optimal_ratio(8, sla)
    assert result.best is None


def test_find_optimal_ratio_budget_too_small(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig()
    result = analyzer.find_optimal_ratio(1, sla)
    assert result.best is None
    assert len(result.candidates) == 0


def test_find_optimal_ratio_sorted_by_waste(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig(ttft_ms=100000, tpot_ms=100000)
    result = analyzer.find_optimal_ratio(8, sla)
    sla_meeting = [c for c in result.candidates if c.meets_sla]
    for i in range(len(sla_meeting) - 1):
        assert sla_meeting[i].waste_rate <= sla_meeting[i + 1].waste_rate


def test_find_optimal_ratio_best_has_lowest_waste(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig(ttft_ms=100000, tpot_ms=100000)
    result = analyzer.find_optimal_ratio(8, sla)
    assert result.best is not None
    for c in result.candidates:
        if c.meets_sla:
            assert result.best.waste_rate <= c.waste_rate


def test_find_optimal_ratio_includes_current_config(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig()
    result = analyzer.find_optimal_ratio(8, sla)
    assert result.current_config is not None
    assert result.current_sla_check is not None


def test_find_optimal_ratio_different_totals(analyzer: BenchmarkAnalyzer):
    sla = SLAConfig(ttft_ms=100000, tpot_ms=100000)
    r4 = analyzer.find_optimal_ratio(4, sla)
    r16 = analyzer.find_optimal_ratio(16, sla)
    assert len(r4.candidates) == 3  # 1:3, 2:2, 3:1
    assert len(r16.candidates) == 15


# ── Test: ratio scaling logic ─────────────────────────────────────────────


def test_more_prefill_reduces_ttft(analyzer: BenchmarkAnalyzer):
    """More prefill instances should reduce TTFT."""
    sla = SLAConfig()
    result = analyzer.find_optimal_ratio(8, sla)
    # Find 1P:7D and 7P:1D
    c_1p = next(c for c in result.candidates if c.num_prefill == 1)
    c_7p = next(c for c in result.candidates if c.num_prefill == 7)
    assert c_7p.sla_check.ttft_p95_ms < c_1p.sla_check.ttft_p95_ms


def test_more_decode_reduces_tpot(analyzer: BenchmarkAnalyzer):
    """More decode instances should reduce TPOT."""
    sla = SLAConfig()
    result = analyzer.find_optimal_ratio(8, sla)
    c_1d = next(c for c in result.candidates if c.num_decode == 1)
    c_7d = next(c for c in result.candidates if c.num_decode == 7)
    assert c_7d.sla_check.tpot_p95_ms < c_1d.sla_check.tpot_p95_ms


# ── Test: BenchmarkData model validation ──────────────────────────────────


def test_benchmark_request_validation():
    with pytest.raises(Exception):
        BenchmarkRequest(
            request_id="r1",
            prompt_tokens=0,  # must be >= 1
            output_tokens=10,
            ttft_ms=50,
            tpot_ms=10,
            total_latency_ms=150,
            timestamp=1700000000,
        )


def test_benchmark_metadata_validation():
    with pytest.raises(Exception):
        BenchmarkMetadata(
            num_prefill_instances=0,  # must be >= 1
            num_decode_instances=2,
            total_instances=2,
            measured_qps=10,
        )


def test_benchmark_data_empty_requests():
    with pytest.raises(Exception):
        BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=10,
            ),
            requests=[],
        )


# ── Test: edge cases ─────────────────────────────────────────────────────


def test_single_request():
    data = _make_benchmark_data(n=1)
    a = BenchmarkAnalyzer()
    a.load_data_from_dict(data)
    sla = SLAConfig(ttft_ms=10000)
    result = a.find_optimal_ratio(4, sla)
    assert result.best is not None


def test_large_dataset():
    data = _make_benchmark_data(n=1000)
    a = BenchmarkAnalyzer()
    a.load_data_from_dict(data)
    sla = SLAConfig(ttft_ms=10000, tpot_ms=10000)
    result = a.find_optimal_ratio(8, sla)
    assert result.best is not None


def test_ratio_candidate_properties():
    c = RatioCandidate(
        num_prefill=3,
        num_decode=5,
        prefill_utilization=0.7,
        decode_utilization=0.8,
        waste_rate=0.3,
        meets_sla=True,
    )
    assert c.total == 8
    assert c.ratio_str == "3P:5D"

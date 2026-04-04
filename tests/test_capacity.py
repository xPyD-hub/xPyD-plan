"""Tests for capacity planning module (M10)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from xpyd_plan.capacity import (
    CapacityPlanner,
    CapacityRecommendation,
    ConfidenceLevel,
    ScalingDataPoint,
    plan_capacity,
)
from xpyd_plan.models import SLAConfig

# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_requests(
    n: int = 100,
    ttft_range: tuple[float, float] = (50, 300),
    tpot_range: tuple[float, float] = (10, 35),
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


def _make_benchmark(
    num_p: int = 2,
    num_d: int = 6,
    qps: float = 10.0,
    n: int = 100,
    seed: int = 42,
    ttft_range: tuple[float, float] = (50, 300),
    tpot_range: tuple[float, float] = (10, 35),
) -> dict:
    return {
        "metadata": {
            "num_prefill_instances": num_p,
            "num_decode_instances": num_d,
            "total_instances": num_p + num_d,
            "measured_qps": qps,
        },
        "requests": _make_requests(
            n=n, seed=seed, ttft_range=ttft_range, tpot_range=tpot_range
        ),
    }


def _make_datasets():
    """Create benchmark datasets at different scales."""
    from xpyd_plan.benchmark_models import BenchmarkData

    configs = [
        (1, 3, 5.0, 10),
        (2, 6, 10.0, 20),
        (3, 9, 15.0, 30),
        (4, 12, 20.0, 40),
    ]
    datasets = []
    for num_p, num_d, qps, seed in configs:
        data = _make_benchmark(num_p=num_p, num_d=num_d, qps=qps, seed=seed)
        datasets.append(BenchmarkData(**data))
    return datasets


@pytest.fixture
def datasets():
    return _make_datasets()


@pytest.fixture
def sla() -> SLAConfig:
    return SLAConfig(ttft_ms=500.0, tpot_ms=50.0)


@pytest.fixture
def benchmark_files(tmp_path: Path) -> list[str]:
    configs = [
        (1, 3, 5.0, 10),
        (2, 6, 10.0, 20),
        (3, 9, 15.0, 30),
    ]
    paths = []
    for num_p, num_d, qps, seed in configs:
        data = _make_benchmark(num_p=num_p, num_d=num_d, qps=qps, seed=seed)
        p = tmp_path / f"bench_{qps:.0f}.json"
        p.write_text(json.dumps(data))
        paths.append(str(p))
    return paths


# ── CapacityPlanner.fit ───────────────────────────────────────────────────


class TestFit:
    def test_fit_stores_points(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        assert len(planner.points) == 4

    def test_fit_points_sorted(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        totals = [p.total_instances for p in planner.points]
        assert totals == sorted(totals)

    def test_fit_empty_raises(self) -> None:
        planner = CapacityPlanner()
        with pytest.raises(ValueError, match="At least one"):
            planner.fit([])

    def test_fit_no_sla_defaults(self, datasets) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets)
        assert len(planner.points) == 4

    def test_fit_point_fields(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        p = planner.points[0]
        assert p.total_instances > 0
        assert p.measured_qps > 0
        assert p.ttft_p95_ms > 0
        assert p.tpot_p95_ms > 0


# ── CapacityPlanner.recommend ─────────────────────────────────────────────


class TestRecommend:
    def test_recommend_basic(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        rec = planner.recommend(target_qps=10.0, sla=sla)
        assert isinstance(rec, CapacityRecommendation)
        assert rec.recommended_instances >= 2
        assert rec.num_prefill >= 1
        assert rec.num_decode >= 1
        assert rec.num_prefill + rec.num_decode == rec.recommended_instances

    def test_recommend_higher_qps_needs_more(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        rec_low = planner.recommend(target_qps=5.0, sla=sla)
        rec_high = planner.recommend(target_qps=30.0, sla=sla)
        assert rec_high.recommended_instances >= rec_low.recommended_instances

    def test_recommend_ratio_string(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        rec = planner.recommend(target_qps=10.0, sla=sla)
        assert "P:" in rec.recommended_ratio
        assert "D" in rec.recommended_ratio

    def test_recommend_headroom_positive(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        rec = planner.recommend(target_qps=5.0, sla=sla)
        # With 20% headroom built in, should be positive
        assert rec.estimated_headroom_pct > 0

    def test_recommend_not_fitted_raises(self) -> None:
        planner = CapacityPlanner()
        with pytest.raises(RuntimeError, match="fit"):
            planner.recommend(target_qps=10.0)

    def test_recommend_negative_qps_raises(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        with pytest.raises(ValueError, match="positive"):
            planner.recommend(target_qps=-1.0)

    def test_recommend_zero_qps_raises(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        with pytest.raises(ValueError, match="positive"):
            planner.recommend(target_qps=0.0)

    def test_recommend_max_instances_cap(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        rec = planner.recommend(target_qps=1000.0, sla=sla, max_instances=10)
        assert rec.recommended_instances <= 10
        assert any("max_instances" in n for n in rec.notes)

    def test_recommend_scaling_points_included(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        rec = planner.recommend(target_qps=10.0, sla=sla)
        assert len(rec.scaling_points) == 4


# ── Confidence levels ─────────────────────────────────────────────────────


class TestConfidence:
    def test_interpolation_high(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        rec = planner.recommend(target_qps=12.0, sla=sla)
        assert rec.confidence == ConfidenceLevel.HIGH

    def test_below_range_high(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        rec = planner.recommend(target_qps=2.0, sla=sla)
        assert rec.confidence == ConfidenceLevel.HIGH

    def test_far_extrapolation_low(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        rec = planner.recommend(target_qps=200.0, sla=sla, max_instances=200)
        assert rec.confidence == ConfidenceLevel.LOW

    def test_slight_extrapolation_medium(self, datasets, sla) -> None:
        planner = CapacityPlanner()
        planner.fit(datasets, sla)
        # QPS range is 5-20; slight overshoot
        rec = planner.recommend(target_qps=25.0, sla=sla)
        assert rec.confidence in (ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH)


# ── Programmatic API ──────────────────────────────────────────────────────


class TestPlanCapacityApi:
    def test_basic(self, benchmark_files) -> None:
        result = plan_capacity(benchmark_files, target_qps=10.0)
        assert "recommended_instances" in result
        assert "recommended_ratio" in result
        assert "confidence" in result

    def test_with_dict_sla(self, benchmark_files) -> None:
        result = plan_capacity(
            benchmark_files, target_qps=10.0, sla_config={"ttft_ms": 500.0}
        )
        assert result["target_qps"] == 10.0

    def test_with_none_sla(self, benchmark_files) -> None:
        result = plan_capacity(benchmark_files, target_qps=10.0, sla_config=None)
        assert "recommended_instances" in result

    def test_max_instances(self, benchmark_files) -> None:
        result = plan_capacity(benchmark_files, target_qps=500.0, max_instances=8)
        assert result["recommended_instances"] <= 8


# ── ScalingDataPoint model ────────────────────────────────────────────────


class TestScalingDataPoint:
    def test_create(self) -> None:
        p = ScalingDataPoint(
            total_instances=8,
            num_prefill=2,
            num_decode=6,
            measured_qps=10.0,
            ttft_p95_ms=200.0,
            tpot_p95_ms=30.0,
            waste_rate=0.1,
        )
        assert p.max_qps_meeting_sla is None

    def test_with_max_qps(self) -> None:
        p = ScalingDataPoint(
            total_instances=8,
            num_prefill=2,
            num_decode=6,
            measured_qps=10.0,
            max_qps_meeting_sla=10.0,
            ttft_p95_ms=200.0,
            tpot_p95_ms=30.0,
            waste_rate=0.1,
        )
        assert p.max_qps_meeting_sla == 10.0

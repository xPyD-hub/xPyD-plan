"""Tests for planner core logic."""

from __future__ import annotations

from xpyd_plan.models import DatasetStats, GPUProfile, SLAConfig
from xpyd_plan.planner import plan


def _default_dataset() -> DatasetStats:
    return DatasetStats(
        prompt_len_mean=500,
        prompt_len_p95=2000,
        output_len_mean=200,
        output_len_p95=800,
        num_requests=500,
    )


def _default_gpu() -> GPUProfile:
    return GPUProfile(
        name="A100-80G",
        prefill_tokens_per_sec=50000,
        decode_tokens_per_sec=2000,
        memory_gb=80,
        cost_per_hour=2.50,
    )


class TestPlanner:
    def test_finds_best_within_sla(self):
        sla = SLAConfig(ttft_ms=1000, tpot_ms=100)
        result = plan(sla=sla, dataset=_default_dataset(), gpu=_default_gpu(), budget=8)
        assert result.best is not None
        assert result.best.meets_sla
        assert result.best.performance.ttft_ms <= 1000
        assert result.best.performance.tpot_ms <= 100

    def test_impossible_sla_returns_none(self):
        # Extremely tight SLA that no config can meet
        sla = SLAConfig(ttft_ms=0.001, tpot_ms=0.001)
        result = plan(sla=sla, dataset=_default_dataset(), gpu=_default_gpu(), budget=4)
        assert result.best is None
        assert all(not c.meets_sla for c in result.candidates)

    def test_budget_too_small(self):
        sla = SLAConfig(tpot_ms=100)
        result = plan(sla=sla, dataset=_default_dataset(), gpu=_default_gpu(), budget=1)
        assert result.best is None
        assert result.candidates == []

    def test_all_candidates_enumerated(self):
        sla = SLAConfig()
        result = plan(sla=sla, dataset=_default_dataset(), gpu=_default_gpu(), budget=6)
        # Should have budget-1 candidates (P=1..5)
        assert len(result.candidates) == 5

    def test_best_has_highest_score(self):
        sla = SLAConfig(ttft_ms=5000, tpot_ms=500)  # relaxed SLA
        result = plan(sla=sla, dataset=_default_dataset(), gpu=_default_gpu(), budget=8)
        assert result.best is not None
        sla_meeting = [c for c in result.candidates if c.meets_sla]
        for c in sla_meeting:
            assert c.score <= result.best.score

    def test_no_sla_all_meet(self):
        """When no SLA constraints are set, all candidates should meet SLA."""
        sla = SLAConfig()
        result = plan(sla=sla, dataset=_default_dataset(), gpu=_default_gpu(), budget=4)
        assert all(c.meets_sla for c in result.candidates)
        assert result.best is not None

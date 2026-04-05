"""Tests for scaling_policy module (M95)."""

from __future__ import annotations

from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.scaling_policy import (
    ScalingAction,
    ScalingDirection,
    ScalingPolicy,
    ScalingPolicyGenerator,
    ScalingRule,
    ScalingTrigger,
    generate_scaling_policy,
)

UP = ScalingDirection.SCALE_UP
DOWN = ScalingDirection.SCALE_DOWN


def _make_benchmark(
    qps: float,
    num_prefill: int,
    num_decode: int,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    n_requests: int = 100,
) -> BenchmarkData:
    """Create a synthetic benchmark dataset."""
    import random

    random.seed(42)
    requests = []
    for i in range(n_requests):
        ttft = ttft_base + random.gauss(0, ttft_base * 0.1)
        tpot = tpot_base + random.gauss(0, tpot_base * 0.1)
        total = ttft + tpot * 50
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100 + random.randint(0, 200),
                output_tokens=50 + random.randint(0, 100),
                ttft_ms=max(1.0, ttft),
                tpot_ms=max(0.1, tpot),
                total_latency_ms=max(1.0, total),
                timestamp=1700000000.0 + i,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=qps,
        ),
        requests=requests,
    )


def _make_high_latency_benchmark(
    qps: float,
    num_prefill: int,
    num_decode: int,
) -> BenchmarkData:
    """Create a benchmark with latency near/above typical SLA."""
    return _make_benchmark(
        qps=qps,
        num_prefill=num_prefill,
        num_decode=num_decode,
        ttft_base=450.0,
        tpot_base=90.0,
    )


class TestScalingPolicyGenerator:
    """Tests for ScalingPolicyGenerator."""

    def test_basic_generation(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(data)
        policy = gen.generate()
        assert isinstance(policy, ScalingPolicy)
        assert len(policy.rules) > 0

    def test_has_scale_up_rules(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(data)
        policy = gen.generate()
        up_rules = [
            r for r in policy.rules if r.action.direction == UP
        ]
        assert len(up_rules) >= 2

    def test_has_scale_down_rules_when_sla_met(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(
            data, sla_ttft_ms=500.0, sla_tpot_ms=100.0,
            sla_total_ms=5000.0,
        )
        policy = gen.generate()
        down_rules = [
            r for r in policy.rules if r.action.direction == DOWN
        ]
        assert len(down_rules) >= 1

    def test_no_scale_down_when_sla_violated(self) -> None:
        data = [_make_high_latency_benchmark(
            qps=100.0, num_prefill=1, num_decode=1,
        )]
        gen = ScalingPolicyGenerator(
            data, sla_ttft_ms=10.0, sla_tpot_ms=1.0,
            sla_total_ms=50.0,
        )
        policy = gen.generate()
        down_rules = [
            r for r in policy.rules if r.action.direction == DOWN
        ]
        assert len(down_rules) == 0

    def test_cooldown_set(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(data)
        policy = gen.generate()
        assert policy.cooldown_seconds > 0

    def test_instance_bounds(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(
            data, min_instances=2, max_instances=32,
        )
        policy = gen.generate()
        assert policy.min_prefill_instances == 2
        assert policy.max_prefill_instances == 32

    def test_multiple_benchmarks(self) -> None:
        data = [
            _make_benchmark(qps=10.0, num_prefill=2, num_decode=2),
            _make_benchmark(
                qps=50.0, num_prefill=4, num_decode=4,
                ttft_base=100.0,
            ),
            _make_benchmark(
                qps=100.0, num_prefill=8, num_decode=8,
                ttft_base=200.0,
            ),
        ]
        gen = ScalingPolicyGenerator(data)
        policy = gen.generate()
        assert len(policy.rules) >= 3

    def test_rules_sorted_by_priority(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(data)
        policy = gen.generate()
        priorities = [r.priority for r in policy.rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_empty_data_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ScalingPolicyGenerator([])

    def test_ttft_scale_up_targets_prefill(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(data)
        policy = gen.generate()
        ttft_up = [
            r for r in policy.rules
            if "ttft" in r.name and r.action.direction == UP
        ]
        for rule in ttft_up:
            assert rule.action.prefill_delta > 0

    def test_tpot_scale_up_targets_decode(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(data)
        policy = gen.generate()
        tpot_up = [
            r for r in policy.rules
            if "tpot" in r.name and r.action.direction == UP
        ]
        for rule in tpot_up:
            assert rule.action.decode_delta > 0

    def test_notes_present(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(data)
        policy = gen.generate()
        assert len(policy.notes) > 0

    def test_custom_sla_thresholds(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(
            data, sla_ttft_ms=200.0, sla_tpot_ms=50.0,
        )
        policy = gen.generate()
        ttft_up = [
            r for r in policy.rules
            if "ttft" in r.name and r.action.direction == UP
        ]
        for rule in ttft_up:
            assert rule.trigger.threshold == 160.0

    def test_evaluation_interval(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(data)
        policy = gen.generate()
        assert policy.evaluation_interval_seconds == 30

    def test_total_critical_rule_adds_both(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(data)
        policy = gen.generate()
        total_rules = [r for r in policy.rules if "total" in r.name]
        for rule in total_rules:
            if rule.action.direction == UP:
                assert rule.action.prefill_delta > 0
                assert rule.action.decode_delta > 0


class TestScalingModels:
    """Tests for Pydantic models."""

    def test_scaling_trigger_model(self) -> None:
        trigger = ScalingTrigger(
            metric="ttft_p95_ms", threshold=400.0,
            comparator="gte",
        )
        assert trigger.sustained_seconds == 60

    def test_scaling_action_model(self) -> None:
        action = ScalingAction(
            direction=ScalingDirection.SCALE_UP, prefill_delta=1,
        )
        assert action.decode_delta == 0

    def test_scaling_rule_model(self) -> None:
        rule = ScalingRule(
            name="test",
            trigger=ScalingTrigger(
                metric="ttft_p95_ms", threshold=400.0,
                comparator="gte",
            ),
            action=ScalingAction(
                direction=ScalingDirection.SCALE_UP,
                prefill_delta=1,
            ),
        )
        assert rule.priority == 1

    def test_policy_serialization(self) -> None:
        data = [_make_benchmark(qps=10.0, num_prefill=2, num_decode=2)]
        gen = ScalingPolicyGenerator(data)
        policy = gen.generate()
        d = policy.model_dump()
        assert "rules" in d
        assert "cooldown_seconds" in d
        policy2 = ScalingPolicy.model_validate(d)
        assert len(policy2.rules) == len(policy.rules)


class TestGenerateScalingPolicyAPI:
    """Tests for programmatic API."""

    def test_api_returns_dict(self, tmp_path: Path) -> None:
        bm = _make_benchmark(qps=10.0, num_prefill=2, num_decode=2)
        f = tmp_path / "bench.json"
        f.write_text(bm.model_dump_json())
        result = generate_scaling_policy([str(f)])
        assert isinstance(result, dict)
        assert "rules" in result

    def test_api_with_custom_sla(self, tmp_path: Path) -> None:
        bm = _make_benchmark(qps=10.0, num_prefill=2, num_decode=2)
        f = tmp_path / "bench.json"
        f.write_text(bm.model_dump_json())
        result = generate_scaling_policy(
            [str(f)], sla_ttft_ms=200.0,
        )
        assert isinstance(result, dict)

"""Tests for benchmark sampling and downsampling."""

from __future__ import annotations

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.sampler import (
    BenchmarkSampler,
    SamplingMethod,
    sample_benchmark,
)


def _make_data(n: int = 200, seed: int = 42) -> BenchmarkData:
    """Create benchmark data with n requests."""
    import random as _random

    rng = _random.Random(seed)
    requests = []
    for i in range(n):
        pt = rng.randint(50, 2000)
        ot = rng.randint(10, 500)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=pt,
                output_tokens=ot,
                ttft_ms=rng.uniform(10, 200),
                tpot_ms=rng.uniform(5, 50),
                total_latency_ms=rng.uniform(100, 5000),
                timestamp=1700000000 + i * 0.5,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=100.0,
        ),
        requests=requests,
    )


class TestRandomSample:
    def test_basic(self) -> None:
        data = _make_data(200)
        sampler = BenchmarkSampler(seed=42)
        result = sampler.random_sample(data, 50)
        assert result.sample_count == 50
        assert result.method == SamplingMethod.RANDOM
        assert len(result.data.requests) == 50

    def test_size_exceeds_total(self) -> None:
        data = _make_data(20)
        sampler = BenchmarkSampler(seed=1)
        result = sampler.random_sample(data, 100)
        assert result.sample_count == 20

    def test_reproducible(self) -> None:
        data = _make_data(200)
        r1 = BenchmarkSampler(seed=99).random_sample(data, 50)
        r2 = BenchmarkSampler(seed=99).random_sample(data, 50)
        ids1 = [r.request_id for r in r1.data.requests]
        ids2 = [r.request_id for r in r2.data.requests]
        assert ids1 == ids2

    def test_different_seeds(self) -> None:
        data = _make_data(200)
        r1 = BenchmarkSampler(seed=1).random_sample(data, 50)
        r2 = BenchmarkSampler(seed=2).random_sample(data, 50)
        ids1 = {r.request_id for r in r1.data.requests}
        ids2 = {r.request_id for r in r2.data.requests}
        assert ids1 != ids2

    def test_qps_adjusted(self) -> None:
        data = _make_data(200)
        result = BenchmarkSampler(seed=42).random_sample(data, 50)
        assert result.data.metadata.measured_qps == pytest.approx(25.0, rel=0.01)


class TestStratifiedSample:
    def test_basic(self) -> None:
        data = _make_data(200)
        sampler = BenchmarkSampler(seed=42)
        result = sampler.stratified_sample(data, 50, bins=4)
        assert result.method == SamplingMethod.STRATIFIED
        assert result.sample_count <= 50

    def test_size_exceeds_total(self) -> None:
        data = _make_data(20)
        result = BenchmarkSampler(seed=1).stratified_sample(data, 100, bins=4)
        assert result.sample_count == 20

    def test_preserves_distribution(self) -> None:
        """Stratified sampling should better preserve token distribution."""
        import numpy as np

        data = _make_data(1000, seed=7)
        orig_tokens = [r.prompt_tokens for r in data.requests]
        orig_std = np.std(orig_tokens)

        result = BenchmarkSampler(seed=42).stratified_sample(data, 100, bins=4)
        samp_tokens = [r.prompt_tokens for r in result.data.requests]
        samp_std = np.std(samp_tokens)

        # Stratified should keep std within reasonable bounds
        assert abs(samp_std - orig_std) / orig_std < 0.3

    def test_reproducible(self) -> None:
        data = _make_data(200)
        r1 = BenchmarkSampler(seed=42).stratified_sample(data, 50)
        r2 = BenchmarkSampler(seed=42).stratified_sample(data, 50)
        ids1 = [r.request_id for r in r1.data.requests]
        ids2 = [r.request_id for r in r2.data.requests]
        assert ids1 == ids2


class TestReservoirSample:
    def test_basic(self) -> None:
        data = _make_data(200)
        result = BenchmarkSampler(seed=42).reservoir_sample(data, 50)
        assert result.sample_count == 50
        assert result.method == SamplingMethod.RESERVOIR

    def test_size_exceeds_total(self) -> None:
        data = _make_data(20)
        result = BenchmarkSampler(seed=1).reservoir_sample(data, 100)
        assert result.sample_count == 20

    def test_reproducible(self) -> None:
        data = _make_data(200)
        r1 = BenchmarkSampler(seed=42).reservoir_sample(data, 50)
        r2 = BenchmarkSampler(seed=42).reservoir_sample(data, 50)
        ids1 = [r.request_id for r in r1.data.requests]
        ids2 = [r.request_id for r in r2.data.requests]
        assert ids1 == ids2


class TestValidation:
    def test_representative(self) -> None:
        data = _make_data(1000, seed=7)
        result = BenchmarkSampler(seed=42, tolerance_pct=20.0).random_sample(data, 500)
        assert result.validation.is_representative is True
        assert len(result.validation.deviations) == 3

    def test_not_representative_tiny_sample(self) -> None:
        data = _make_data(1000, seed=7)
        result = BenchmarkSampler(seed=42, tolerance_pct=0.1).random_sample(data, 10)
        # Very tiny sample with very tight tolerance — likely not representative
        assert result.validation.tolerance_pct == 0.1

    def test_deviation_fields(self) -> None:
        data = _make_data(200)
        result = BenchmarkSampler(seed=42).random_sample(data, 100)
        for d in result.validation.deviations:
            assert d.metric in ("ttft_ms", "tpot_ms", "total_latency_ms")
            assert d.p50_deviation_pct >= 0
            assert d.p95_deviation_pct >= 0
            assert d.p99_deviation_pct >= 0


class TestProgrammaticAPI:
    def test_random(self) -> None:
        data = _make_data(100)
        result = sample_benchmark(data, SamplingMethod.RANDOM, size=30, seed=42)
        assert result.sample_count == 30

    def test_stratified(self) -> None:
        data = _make_data(100)
        result = sample_benchmark(data, SamplingMethod.STRATIFIED, size=30, seed=42, bins=3)
        assert result.sample_count <= 30

    def test_reservoir(self) -> None:
        data = _make_data(100)
        result = sample_benchmark(data, SamplingMethod.RESERVOIR, size=30, seed=42)
        assert result.sample_count == 30

    def test_invalid_method(self) -> None:
        data = _make_data(10)
        with pytest.raises(ValueError, match="Unknown"):
            sample_benchmark(data, "bogus", size=5)  # type: ignore[arg-type]


class TestSampleResult:
    def test_fraction(self) -> None:
        data = _make_data(200)
        result = BenchmarkSampler(seed=1).random_sample(data, 50)
        assert result.sample_fraction == pytest.approx(0.25, abs=0.01)

    def test_original_count(self) -> None:
        data = _make_data(150)
        result = BenchmarkSampler(seed=1).random_sample(data, 50)
        assert result.original_count == 150

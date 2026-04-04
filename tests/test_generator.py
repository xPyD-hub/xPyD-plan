"""Tests for benchmark simulation generator (M26)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.generator import (
    AnomalyConfig,
    AnomalyType,
    BenchmarkGenerator,
    DistributionType,
    GeneratorConfig,
    LatencyProfile,
    generate_benchmark,
    load_generator_config,
)


class TestGeneratorConfig:
    """Tests for GeneratorConfig model."""

    def test_default_config(self):
        cfg = GeneratorConfig()
        assert cfg.num_requests == 1000
        assert cfg.num_prefill_instances == 2
        assert cfg.num_decode_instances == 2
        assert cfg.measured_qps == 100.0
        assert cfg.seed is None

    def test_custom_config(self):
        cfg = GeneratorConfig(
            num_requests=500,
            num_prefill_instances=4,
            num_decode_instances=8,
            measured_qps=200.0,
            seed=42,
        )
        assert cfg.num_requests == 500
        assert cfg.num_prefill_instances == 4
        assert cfg.seed == 42

    def test_invalid_num_requests(self):
        with pytest.raises(Exception):
            GeneratorConfig(num_requests=0)

    def test_invalid_qps(self):
        with pytest.raises(Exception):
            GeneratorConfig(measured_qps=-1.0)

    def test_invalid_instances(self):
        with pytest.raises(Exception):
            GeneratorConfig(num_prefill_instances=0)


class TestLatencyProfile:
    """Tests for LatencyProfile model."""

    def test_default_profile(self):
        p = LatencyProfile()
        assert p.distribution == DistributionType.LOGNORMAL
        assert p.ttft_mean_ms == 30.0
        assert p.bimodal_split == 0.7

    def test_normal_distribution(self):
        p = LatencyProfile(distribution=DistributionType.NORMAL)
        assert p.distribution == DistributionType.NORMAL

    def test_bimodal_distribution(self):
        p = LatencyProfile(
            distribution=DistributionType.BIMODAL,
            bimodal_split=0.6,
            bimodal_second_factor=2.5,
        )
        assert p.bimodal_split == 0.6
        assert p.bimodal_second_factor == 2.5


class TestAnomalyConfig:
    """Tests for AnomalyConfig model."""

    def test_spike_config(self):
        a = AnomalyConfig(type=AnomalyType.SPIKE, fraction=0.1, multiplier=10.0)
        assert a.type == AnomalyType.SPIKE
        assert a.fraction == 0.1

    def test_cold_start_config(self):
        a = AnomalyConfig(
            type=AnomalyType.COLD_START,
            cold_start_count=5,
            cold_start_multiplier=4.0,
        )
        assert a.cold_start_count == 5


class TestBenchmarkGenerator:
    """Tests for BenchmarkGenerator."""

    def test_generate_default(self):
        cfg = GeneratorConfig(num_requests=100, seed=42)
        gen = BenchmarkGenerator(cfg)
        data = gen.generate()
        assert isinstance(data, BenchmarkData)
        assert len(data.requests) == 100
        assert data.metadata.num_prefill_instances == 2
        assert data.metadata.num_decode_instances == 2
        assert data.metadata.total_instances == 4

    def test_generate_reproducible(self):
        cfg = GeneratorConfig(num_requests=50, seed=123)
        data1 = BenchmarkGenerator(cfg).generate()
        data2 = BenchmarkGenerator(cfg).generate()
        assert data1.requests[0].ttft_ms == data2.requests[0].ttft_ms
        assert data1.requests[-1].tpot_ms == data2.requests[-1].tpot_ms

    def test_generate_normal_distribution(self):
        cfg = GeneratorConfig(
            num_requests=200,
            seed=42,
            latency=LatencyProfile(distribution=DistributionType.NORMAL),
        )
        data = BenchmarkGenerator(cfg).generate()
        assert len(data.requests) == 200
        assert all(r.ttft_ms > 0 for r in data.requests)

    def test_generate_lognormal_distribution(self):
        cfg = GeneratorConfig(
            num_requests=200,
            seed=42,
            latency=LatencyProfile(distribution=DistributionType.LOGNORMAL),
        )
        data = BenchmarkGenerator(cfg).generate()
        assert all(r.tpot_ms > 0 for r in data.requests)

    def test_generate_bimodal_distribution(self):
        cfg = GeneratorConfig(
            num_requests=500,
            seed=42,
            latency=LatencyProfile(distribution=DistributionType.BIMODAL),
        )
        data = BenchmarkGenerator(cfg).generate()
        ttfts = [r.ttft_ms for r in data.requests]
        # Bimodal should have wider spread
        assert max(ttfts) > min(ttfts)

    def test_generate_with_spike_anomalies(self):
        cfg_clean = GeneratorConfig(num_requests=100, seed=42)
        cfg_spike = GeneratorConfig(
            num_requests=100,
            seed=42,
            anomalies=[
                AnomalyConfig(type=AnomalyType.SPIKE, fraction=0.2, multiplier=5.0)
            ],
        )
        data_clean = BenchmarkGenerator(cfg_clean).generate()
        data_spike = BenchmarkGenerator(cfg_spike).generate()
        # Spike data should have higher max latency
        max_clean = max(r.total_latency_ms for r in data_clean.requests)
        max_spike = max(r.total_latency_ms for r in data_spike.requests)
        assert max_spike > max_clean

    def test_generate_with_cold_start_anomalies(self):
        cfg = GeneratorConfig(
            num_requests=100,
            seed=42,
            anomalies=[
                AnomalyConfig(
                    type=AnomalyType.COLD_START,
                    cold_start_count=5,
                    cold_start_multiplier=4.0,
                )
            ],
        )
        data = BenchmarkGenerator(cfg).generate()
        # First 5 requests should have inflated TTFT
        early_ttfts = [data.requests[i].ttft_ms for i in range(5)]
        later_ttfts = [data.requests[i].ttft_ms for i in range(50, 55)]
        assert sum(early_ttfts) / 5 > sum(later_ttfts) / 5

    def test_generate_positive_values(self):
        cfg = GeneratorConfig(num_requests=500, seed=42)
        data = BenchmarkGenerator(cfg).generate()
        for r in data.requests:
            assert r.ttft_ms > 0
            assert r.tpot_ms > 0
            assert r.total_latency_ms > 0
            assert r.prompt_tokens >= 1
            assert r.output_tokens >= 1

    def test_generate_unique_request_ids(self):
        cfg = GeneratorConfig(num_requests=200, seed=42)
        data = BenchmarkGenerator(cfg).generate()
        ids = [r.request_id for r in data.requests]
        assert len(set(ids)) == 200

    def test_generate_ordered_timestamps(self):
        cfg = GeneratorConfig(num_requests=100, seed=42)
        data = BenchmarkGenerator(cfg).generate()
        timestamps = [r.timestamp for r in data.requests]
        assert timestamps == sorted(timestamps)

    def test_to_json(self, tmp_path: Path):
        cfg = GeneratorConfig(num_requests=50, seed=42)
        gen = BenchmarkGenerator(cfg)
        out = tmp_path / "benchmark.json"
        gen.to_json(out)
        assert out.exists()
        with open(out) as f:
            raw = json.load(f)
        data = BenchmarkData(**raw)
        assert len(data.requests) == 50

    def test_to_json_creates_parent_dirs(self, tmp_path: Path):
        cfg = GeneratorConfig(num_requests=10, seed=42)
        gen = BenchmarkGenerator(cfg)
        out = tmp_path / "sub" / "dir" / "benchmark.json"
        gen.to_json(out)
        assert out.exists()

    def test_metadata_correct(self):
        cfg = GeneratorConfig(
            num_requests=10,
            num_prefill_instances=3,
            num_decode_instances=5,
            measured_qps=150.0,
            seed=42,
        )
        data = BenchmarkGenerator(cfg).generate()
        assert data.metadata.num_prefill_instances == 3
        assert data.metadata.num_decode_instances == 5
        assert data.metadata.total_instances == 8
        assert data.metadata.measured_qps == 150.0

    def test_multiple_anomalies(self):
        cfg = GeneratorConfig(
            num_requests=200,
            seed=42,
            anomalies=[
                AnomalyConfig(type=AnomalyType.SPIKE, fraction=0.1, multiplier=5.0),
                AnomalyConfig(
                    type=AnomalyType.COLD_START,
                    cold_start_count=10,
                    cold_start_multiplier=3.0,
                ),
            ],
        )
        data = BenchmarkGenerator(cfg).generate()
        assert len(data.requests) == 200


class TestLoadGeneratorConfig:
    """Tests for YAML config loading."""

    def test_load_valid_config(self, tmp_path: Path):
        cfg_data = {
            "num_requests": 200,
            "num_prefill_instances": 3,
            "num_decode_instances": 5,
            "measured_qps": 50.0,
            "seed": 99,
            "latency": {"distribution": "normal", "ttft_mean_ms": 25.0},
        }
        cfg_file = tmp_path / "gen.yaml"
        import yaml

        cfg_file.write_text(yaml.dump(cfg_data))
        cfg = load_generator_config(cfg_file)
        assert cfg.num_requests == 200
        assert cfg.latency.distribution == DistributionType.NORMAL

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_generator_config("/nonexistent/gen.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("just a string")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_generator_config(cfg_file)


class TestGenerateBenchmarkAPI:
    """Tests for the programmatic generate_benchmark() API."""

    def test_default_generation(self):
        data = generate_benchmark(num_requests=50, seed=42)
        assert isinstance(data, BenchmarkData)
        assert len(data.requests) == 50

    def test_with_config_object(self):
        cfg = GeneratorConfig(num_requests=30, seed=42)
        data = generate_benchmark(cfg)
        assert len(data.requests) == 30

    def test_with_config_file(self, tmp_path: Path):
        import yaml

        cfg_file = tmp_path / "gen.yaml"
        cfg_file.write_text(yaml.dump({"num_requests": 25, "seed": 42}))
        data = generate_benchmark(cfg_file)
        assert len(data.requests) == 25

    def test_with_output_file(self, tmp_path: Path):
        out = tmp_path / "output.json"
        data = generate_benchmark(num_requests=20, seed=42, output=out)
        assert out.exists()
        assert len(data.requests) == 20

    def test_kwargs_override(self):
        cfg = GeneratorConfig(num_requests=100, seed=42)
        data = generate_benchmark(cfg, num_requests=10)
        assert len(data.requests) == 10

    def test_invalid_config_type(self):
        with pytest.raises(TypeError):
            generate_benchmark(12345)  # type: ignore[arg-type]

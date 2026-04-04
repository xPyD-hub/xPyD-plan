"""Benchmark simulation generator for synthetic data creation.

Generate realistic benchmark datasets for testing, demos, and validation
without running actual benchmarks. Supports configurable latency distributions
and anomaly injection.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)


class DistributionType(str, Enum):
    """Supported latency distribution types."""

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    BIMODAL = "bimodal"


class AnomalyType(str, Enum):
    """Supported anomaly types for injection."""

    SPIKE = "spike"
    COLD_START = "cold_start"


class LatencyProfile(BaseModel):
    """Configuration for latency generation."""

    distribution: DistributionType = Field(
        DistributionType.LOGNORMAL, description="Latency distribution type"
    )
    ttft_mean_ms: float = Field(30.0, gt=0, description="Mean TTFT in ms")
    ttft_stddev_ms: float = Field(10.0, gt=0, description="TTFT standard deviation in ms")
    tpot_mean_ms: float = Field(15.0, gt=0, description="Mean TPOT in ms")
    tpot_stddev_ms: float = Field(5.0, gt=0, description="TPOT standard deviation in ms")
    bimodal_split: float = Field(
        0.7, ge=0, le=1, description="Fraction of requests in the first mode (bimodal only)"
    )
    bimodal_second_factor: float = Field(
        3.0, gt=1, description="Multiplier for the second mode mean (bimodal only)"
    )


class AnomalyConfig(BaseModel):
    """Configuration for anomaly injection."""

    type: AnomalyType = Field(..., description="Anomaly type")
    fraction: float = Field(
        0.05, gt=0, le=1, description="Fraction of requests affected"
    )
    multiplier: float = Field(
        5.0, gt=1, description="Latency multiplier for spikes"
    )
    cold_start_count: int = Field(
        10, ge=1, description="Number of initial cold-start requests"
    )
    cold_start_multiplier: float = Field(
        3.0, gt=1, description="Latency multiplier for cold-start requests"
    )


class GeneratorConfig(BaseModel):
    """Complete configuration for benchmark generation."""

    num_requests: int = Field(1000, ge=1, description="Number of requests to generate")
    num_prefill_instances: int = Field(2, ge=1, description="Prefill instances")
    num_decode_instances: int = Field(2, ge=1, description="Decode instances")
    measured_qps: float = Field(100.0, gt=0, description="Simulated QPS")
    prompt_tokens_mean: int = Field(256, ge=1, description="Mean prompt tokens")
    prompt_tokens_stddev: int = Field(64, ge=1, description="Prompt tokens stddev")
    output_tokens_mean: int = Field(128, ge=1, description="Mean output tokens")
    output_tokens_stddev: int = Field(32, ge=1, description="Output tokens stddev")
    latency: LatencyProfile = Field(
        default_factory=LatencyProfile, description="Latency generation profile"
    )
    anomalies: list[AnomalyConfig] = Field(
        default_factory=list, description="Anomaly injection configs"
    )
    seed: int | None = Field(None, description="Random seed for reproducibility")

    @field_validator("num_prefill_instances", "num_decode_instances")
    @classmethod
    def _positive_instances(cls, v: int) -> int:
        if v < 1:
            msg = "Instance count must be >= 1"
            raise ValueError(msg)
        return v


def _sample_normal(mean: float, stddev: float, rng: random.Random) -> float:
    """Sample from normal distribution, clamped to > 0."""
    return max(0.1, rng.gauss(mean, stddev))


def _sample_lognormal(mean: float, stddev: float, rng: random.Random) -> float:
    """Sample from log-normal distribution."""
    if mean <= 0 or stddev <= 0:
        return max(0.1, mean)
    # Convert mean/stddev to log-space parameters
    variance = stddev**2
    mu = math.log(mean**2 / math.sqrt(variance + mean**2))
    sigma = math.sqrt(math.log(1 + variance / mean**2))
    return max(0.1, rng.lognormvariate(mu, sigma))


def _sample_bimodal(
    mean: float, stddev: float, split: float, second_factor: float, rng: random.Random
) -> float:
    """Sample from bimodal distribution (mixture of two normals)."""
    if rng.random() < split:
        return max(0.1, rng.gauss(mean, stddev))
    return max(0.1, rng.gauss(mean * second_factor, stddev * second_factor))


def _sample_latency(
    mean: float, stddev: float, profile: LatencyProfile, rng: random.Random
) -> float:
    """Sample a latency value based on the configured distribution."""
    if profile.distribution == DistributionType.NORMAL:
        return _sample_normal(mean, stddev, rng)
    if profile.distribution == DistributionType.LOGNORMAL:
        return _sample_lognormal(mean, stddev, rng)
    if profile.distribution == DistributionType.BIMODAL:
        return _sample_bimodal(
            mean, stddev, profile.bimodal_split, profile.bimodal_second_factor, rng
        )
    msg = f"Unknown distribution: {profile.distribution}"
    raise ValueError(msg)


class BenchmarkGenerator:
    """Generate synthetic benchmark data.

    Args:
        config: Generator configuration.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self._config = config
        self._rng = random.Random(config.seed)

    def generate(self) -> BenchmarkData:
        """Generate a complete benchmark dataset.

        Returns:
            BenchmarkData with synthetic requests.
        """
        cfg = self._config
        rng = self._rng

        requests: list[BenchmarkRequest] = []
        base_timestamp = 1700000000.0
        interval = 1.0 / cfg.measured_qps

        for i in range(cfg.num_requests):
            prompt_tokens = max(1, int(rng.gauss(cfg.prompt_tokens_mean, cfg.prompt_tokens_stddev)))
            output_tokens = max(1, int(rng.gauss(cfg.output_tokens_mean, cfg.output_tokens_stddev)))

            ttft = _sample_latency(
                cfg.latency.ttft_mean_ms, cfg.latency.ttft_stddev_ms, cfg.latency, rng
            )
            tpot = _sample_latency(
                cfg.latency.tpot_mean_ms, cfg.latency.tpot_stddev_ms, cfg.latency, rng
            )
            total_latency = ttft + tpot * output_tokens

            timestamp = base_timestamp + i * interval + rng.uniform(0, interval * 0.1)

            requests.append(
                BenchmarkRequest(
                    request_id=str(uuid.UUID(int=rng.getrandbits(128), version=4)),
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    ttft_ms=round(ttft, 3),
                    tpot_ms=round(tpot, 3),
                    total_latency_ms=round(total_latency, 3),
                    timestamp=round(timestamp, 6),
                )
            )

        # Apply anomalies
        requests = self._apply_anomalies(requests)

        metadata = BenchmarkMetadata(
            num_prefill_instances=cfg.num_prefill_instances,
            num_decode_instances=cfg.num_decode_instances,
            total_instances=cfg.num_prefill_instances + cfg.num_decode_instances,
            measured_qps=cfg.measured_qps,
        )

        return BenchmarkData(metadata=metadata, requests=requests)

    def _apply_anomalies(
        self, requests: list[BenchmarkRequest]
    ) -> list[BenchmarkRequest]:
        """Apply anomaly injection to generated requests."""
        result = list(requests)
        rng = self._rng

        for anomaly in self._config.anomalies:
            if anomaly.type == AnomalyType.SPIKE:
                num_spikes = max(1, int(len(result) * anomaly.fraction))
                indices = rng.sample(range(len(result)), min(num_spikes, len(result)))
                for idx in indices:
                    req = result[idx]
                    result[idx] = req.model_copy(
                        update={
                            "ttft_ms": round(req.ttft_ms * anomaly.multiplier, 3),
                            "tpot_ms": round(req.tpot_ms * anomaly.multiplier, 3),
                            "total_latency_ms": round(
                                req.total_latency_ms * anomaly.multiplier, 3
                            ),
                        }
                    )
            elif anomaly.type == AnomalyType.COLD_START:
                count = min(anomaly.cold_start_count, len(result))
                for idx in range(count):
                    req = result[idx]
                    result[idx] = req.model_copy(
                        update={
                            "ttft_ms": round(
                                req.ttft_ms * anomaly.cold_start_multiplier, 3
                            ),
                            "total_latency_ms": round(
                                req.total_latency_ms * anomaly.cold_start_multiplier, 3
                            ),
                        }
                    )

        return result

    def to_json(self, path: str | Path) -> None:
        """Generate benchmark data and write to JSON file.

        Args:
            path: Output file path.
        """
        data = self.generate()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(data.model_dump(), f, indent=2)


def load_generator_config(path: str | Path) -> GeneratorConfig:
    """Load generator configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated GeneratorConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    p = Path(path)
    if not p.exists():
        msg = f"Generator config not found: {p}"
        raise FileNotFoundError(msg)
    with open(p) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        msg = "Generator config must be a YAML mapping"
        raise ValueError(msg)
    return GeneratorConfig(**raw)


def generate_benchmark(
    config: GeneratorConfig | str | Path | None = None,
    *,
    output: str | Path | None = None,
    **kwargs: Any,
) -> BenchmarkData:
    """Programmatic API for benchmark generation.

    Args:
        config: GeneratorConfig, path to YAML config, or None for defaults.
        output: Optional output file path for JSON export.
        **kwargs: Override config fields (e.g., num_requests=500).

    Returns:
        Generated BenchmarkData.
    """
    if config is None:
        cfg = GeneratorConfig(**kwargs)
    elif isinstance(config, (str, Path)):
        cfg = load_generator_config(config)
        if kwargs:
            cfg = cfg.model_copy(update=kwargs)
    elif isinstance(config, GeneratorConfig):
        cfg = config
        if kwargs:
            cfg = cfg.model_copy(update=kwargs)
    else:
        msg = f"Invalid config type: {type(config)}"
        raise TypeError(msg)

    generator = BenchmarkGenerator(cfg)

    if output is not None:
        generator.to_json(output)
        # Re-generate with same seed for return value consistency
        generator2 = BenchmarkGenerator(cfg)
        return generator2.generate()

    return generator.generate()

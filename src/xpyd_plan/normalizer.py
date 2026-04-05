"""Benchmark normalization across GPU types.

Scales latency values using GPU-relative performance factors to enable
fair cross-hardware comparison.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData, BenchmarkRequest


class GPUType(str, Enum):
    """Known GPU types."""

    A100_80G = "A100-80G"
    H100_80G = "H100-80G"
    H200_141G = "H200-141G"
    A10G_24G = "A10G-24G"
    L40S_48G = "L40S-48G"


# Performance factors relative to A100-80G = 1.0.
# Higher means faster (lower latency). Based on public FP16 throughput ratios.
_DEFAULT_FACTORS: dict[GPUType, float] = {
    GPUType.A100_80G: 1.0,
    GPUType.H100_80G: 1.8,
    GPUType.H200_141G: 2.2,
    GPUType.A10G_24G: 0.45,
    GPUType.L40S_48G: 0.7,
}


class GPUPerformanceFactor(BaseModel):
    """Performance factor for a GPU type relative to reference."""

    gpu_type: str
    factor: float = Field(..., gt=0, description="Performance factor (higher = faster)")


class NormalizationConfig(BaseModel):
    """Configuration for benchmark normalization."""

    source_gpu: str = Field(..., description="GPU type the benchmark was run on")
    target_gpu: str = Field(
        "A100-80G", description="GPU type to normalize to (reference)"
    )
    custom_factors: dict[str, float] | None = Field(
        None, description="Custom GPU performance factors (overrides defaults)"
    )


class NormalizedStats(BaseModel):
    """Statistics before and after normalization."""

    original_p50: float
    original_p95: float
    original_p99: float
    normalized_p50: float
    normalized_p95: float
    normalized_p99: float
    scaling_factor: float


class NormalizationReport(BaseModel):
    """Complete normalization report."""

    config: NormalizationConfig
    source_factor: float
    target_factor: float
    scaling_factor: float = Field(
        ..., description="Multiplier applied to latencies (source_factor / target_factor)"
    )
    request_count: int
    ttft_stats: NormalizedStats
    tpot_stats: NormalizedStats
    total_latency_stats: NormalizedStats
    original_qps: float
    normalized_qps: float
    known_gpus: list[GPUPerformanceFactor]


class BenchmarkNormalizer:
    """Normalize benchmark latencies across GPU types."""

    def __init__(self, config: NormalizationConfig) -> None:
        self._config = config
        self._factors = self._build_factors()

    def _build_factors(self) -> dict[str, float]:
        factors: dict[str, float] = {g.value: f for g, f in _DEFAULT_FACTORS.items()}
        if self._config.custom_factors:
            factors.update(self._config.custom_factors)
        return factors

    def _get_factor(self, gpu: str) -> float:
        if gpu in self._factors:
            return self._factors[gpu]
        raise ValueError(
            f"Unknown GPU type '{gpu}'. Known types: {sorted(self._factors.keys())}"
        )

    def normalize(self, data: BenchmarkData) -> NormalizationReport:
        """Normalize benchmark data and return report."""
        import numpy as np

        cfg = self._config
        source_factor = self._get_factor(cfg.source_gpu)
        target_factor = self._get_factor(cfg.target_gpu)
        # If source is faster than target, latencies should increase (scale up)
        scaling = source_factor / target_factor

        requests = data.requests
        if not requests:
            raise ValueError("Benchmark contains no requests")

        ttft = np.array([r.ttft_ms for r in requests])
        tpot = np.array([r.tpot_ms for r in requests])
        total = np.array([r.total_latency_ms for r in requests])

        def _stats(orig: np.ndarray, scaled: np.ndarray) -> NormalizedStats:
            return NormalizedStats(
                original_p50=float(np.percentile(orig, 50)),
                original_p95=float(np.percentile(orig, 95)),
                original_p99=float(np.percentile(orig, 99)),
                normalized_p50=float(np.percentile(scaled, 50)),
                normalized_p95=float(np.percentile(scaled, 95)),
                normalized_p99=float(np.percentile(scaled, 99)),
                scaling_factor=scaling,
            )

        ttft_scaled = ttft * scaling
        tpot_scaled = tpot * scaling
        total_scaled = total * scaling

        original_qps = data.metadata.measured_qps
        # Faster GPU → higher QPS, so normalized QPS scales inversely
        normalized_qps = original_qps / scaling if scaling > 0 else 0.0

        known = [
            GPUPerformanceFactor(gpu_type=k, factor=v)
            for k, v in sorted(self._factors.items())
        ]

        return NormalizationReport(
            config=cfg,
            source_factor=source_factor,
            target_factor=target_factor,
            scaling_factor=scaling,
            request_count=len(requests),
            ttft_stats=_stats(ttft, ttft_scaled),
            tpot_stats=_stats(tpot, tpot_scaled),
            total_latency_stats=_stats(total, total_scaled),
            original_qps=original_qps,
            normalized_qps=normalized_qps,
            known_gpus=known,
        )

    def normalize_data(self, data: BenchmarkData) -> BenchmarkData:
        """Return a new BenchmarkData with normalized latencies."""
        cfg = self._config
        source_factor = self._get_factor(cfg.source_gpu)
        target_factor = self._get_factor(cfg.target_gpu)
        scaling = source_factor / target_factor

        new_requests = [
            BenchmarkRequest(
                request_id=r.request_id,
                prompt_tokens=r.prompt_tokens,
                output_tokens=r.output_tokens,
                ttft_ms=r.ttft_ms * scaling,
                tpot_ms=r.tpot_ms * scaling,
                total_latency_ms=r.total_latency_ms * scaling,
                timestamp=r.timestamp,
            )
            for r in data.requests
        ]

        from .benchmark_models import BenchmarkMetadata

        new_cluster = BenchmarkMetadata(
            num_prefill_instances=data.metadata.num_prefill_instances,
            num_decode_instances=data.metadata.num_decode_instances,
            total_instances=data.metadata.total_instances,
            measured_qps=data.metadata.measured_qps / scaling if scaling > 0 else 0.0,
        )

        return BenchmarkData(requests=new_requests, metadata=new_cluster)


def normalize_benchmark(
    data: BenchmarkData,
    source_gpu: str,
    target_gpu: str = "A100-80G",
    custom_factors: dict[str, float] | None = None,
) -> NormalizationReport:
    """Programmatic API for benchmark normalization."""
    config = NormalizationConfig(
        source_gpu=source_gpu,
        target_gpu=target_gpu,
        custom_factors=custom_factors,
    )
    return BenchmarkNormalizer(config).normalize(data)

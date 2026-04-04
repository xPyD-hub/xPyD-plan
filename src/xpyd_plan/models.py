"""Core data models for xPyD planner."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SLAConfig(BaseModel):
    """Service Level Agreement constraints (all in milliseconds)."""

    ttft_ms: float | None = Field(None, description="Max Time To First Token (ms)")
    tpot_ms: float | None = Field(None, description="Max Time Per Output Token (ms)")
    max_latency_ms: float | None = Field(None, description="Max end-to-end latency (ms)")
    sla_percentile: float = Field(
        95.0,
        ge=1.0,
        le=100.0,
        description="Percentile for SLA evaluation (e.g. 90, 95, 99). Default: 95",
    )

    def has_constraints(self) -> bool:
        """Return True if at least one SLA constraint is set."""
        return any(v is not None for v in (self.ttft_ms, self.tpot_ms, self.max_latency_ms))


class DatasetStats(BaseModel):
    """Statistics of a dataset's token length distribution."""

    prompt_len_mean: float = Field(..., description="Mean prompt length in tokens")
    prompt_len_p95: float = Field(..., description="P95 prompt length in tokens")
    output_len_mean: float = Field(..., description="Mean output length in tokens")
    output_len_p95: float = Field(..., description="P95 output length in tokens")
    num_requests: int = Field(1000, description="Number of concurrent requests to model")

    @classmethod
    def from_records(cls, records: list[dict[str, int]], num_requests: int = 1000) -> DatasetStats:
        """Compute stats from a list of {prompt_len, output_len} records."""
        if not records:
            raise ValueError("records must not be empty")
        prompt_lens = sorted(r["prompt_len"] for r in records)
        output_lens = sorted(r["output_len"] for r in records)

        def _mean(vals: list[int]) -> float:
            return sum(vals) / len(vals)

        def _percentile(vals: list[int], p: float) -> float:
            idx = int(len(vals) * p / 100)
            return float(vals[min(idx, len(vals) - 1)])

        return cls(
            prompt_len_mean=_mean(prompt_lens),
            prompt_len_p95=_percentile(prompt_lens, 95),
            output_len_mean=_mean(output_lens),
            output_len_p95=_percentile(output_lens, 95),
            num_requests=num_requests,
        )


class GPUProfile(BaseModel):
    """Hardware profile for a single GPU."""

    name: str = Field(..., description="GPU model name, e.g. 'A100-80G'")
    prefill_tokens_per_sec: float = Field(
        ..., description="Prefill throughput (tokens/s) per GPU"
    )
    decode_tokens_per_sec: float = Field(
        ..., description="Decode throughput (tokens/s) per GPU"
    )
    memory_gb: float = Field(80.0, description="GPU memory in GB")
    cost_per_hour: float = Field(1.0, description="Cost per GPU per hour ($/h)")


class PDConfig(BaseModel):
    """A specific Prefill:Decode GPU allocation."""

    num_prefill: int = Field(..., ge=1, description="Number of prefill GPUs")
    num_decode: int = Field(..., ge=1, description="Number of decode GPUs")

    @property
    def total(self) -> int:
        """Total number of GPUs."""
        return self.num_prefill + self.num_decode

    @property
    def ratio_str(self) -> str:
        """Human-readable P:D ratio string."""
        return f"{self.num_prefill}P:{self.num_decode}D"




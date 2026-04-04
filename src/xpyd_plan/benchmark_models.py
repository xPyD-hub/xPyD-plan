"""Benchmark data models for measured-data analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkRequest(BaseModel):
    """A single request record from benchmark output."""

    request_id: str = Field(..., description="Unique request identifier")
    prompt_tokens: int = Field(..., ge=1, description="Number of prompt tokens")
    output_tokens: int = Field(..., ge=1, description="Number of output tokens")
    ttft_ms: float = Field(..., ge=0, description="Time to first token (ms)")
    tpot_ms: float = Field(..., ge=0, description="Time per output token (ms)")
    total_latency_ms: float = Field(..., ge=0, description="Total end-to-end latency (ms)")
    timestamp: float = Field(..., description="Request start timestamp (epoch seconds)")


class BenchmarkMetadata(BaseModel):
    """Cluster configuration and metadata from a benchmark run."""

    num_prefill_instances: int = Field(..., ge=1, description="Prefill instances in this run")
    num_decode_instances: int = Field(..., ge=1, description="Decode instances in this run")
    total_instances: int = Field(..., ge=2, description="Total instances")
    measured_qps: float = Field(..., gt=0, description="Measured QPS during the run")


class BenchmarkData(BaseModel):
    """Complete benchmark dataset."""

    metadata: BenchmarkMetadata
    requests: list[BenchmarkRequest] = Field(..., min_length=1)


class SLACheck(BaseModel):
    """Result of SLA compliance check."""

    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    total_latency_p95_ms: float
    total_latency_p99_ms: float
    meets_ttft: bool
    meets_tpot: bool
    meets_total_latency: bool
    meets_all: bool


class UtilizationResult(BaseModel):
    """Utilization metrics for P and D instances."""

    prefill_utilization: float = Field(..., ge=0, le=1, description="Prefill instance utilization")
    decode_utilization: float = Field(..., ge=0, le=1, description="Decode instance utilization")
    waste_rate: float = Field(
        ..., ge=0, le=1, description="Waste rate: 1 - min(P_util, D_util)"
    )


class RatioCandidate(BaseModel):
    """A candidate P:D ratio with analysis results."""

    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    prefill_utilization: float
    decode_utilization: float
    waste_rate: float
    meets_sla: bool
    sla_check: SLACheck | None = None

    @property
    def total(self) -> int:
        return self.num_prefill + self.num_decode

    @property
    def ratio_str(self) -> str:
        return f"{self.num_prefill}P:{self.num_decode}D"


class AnalysisResult(BaseModel):
    """Complete analysis result."""

    best: RatioCandidate | None = Field(None, description="Best P:D ratio (None if none meets SLA)")
    candidates: list[RatioCandidate] = Field(default_factory=list)
    total_instances: int
    current_config: UtilizationResult | None = None
    current_sla_check: SLACheck | None = None


class ScenarioResult(BaseModel):
    """Analysis result for a single QPS scenario."""

    qps: float = Field(..., description="Measured QPS for this scenario")
    analysis: AnalysisResult


class MultiScenarioResult(BaseModel):
    """Complete multi-scenario analysis result."""

    scenarios: list[ScenarioResult] = Field(default_factory=list)
    total_instances: int
    unified_best: RatioCandidate | None = Field(
        None, description="Best P:D ratio that meets SLA across ALL scenarios"
    )
    per_scenario_best: list[RatioCandidate | None] = Field(
        default_factory=list,
        description="Best ratio per scenario (for reference)",
    )

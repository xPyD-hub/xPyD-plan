"""Multi-backend comparison report.

Compare benchmark results across different serving backends (vLLM, SGLang,
TensorRT-LLM, native) to identify the best backend for a given workload
based on latency, throughput, and SLA compliance.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.sglang_import import _detect_sglang_format, import_sglang_data
from xpyd_plan.trtllm_import import _detect_trtllm_format, import_trtllm_data
from xpyd_plan.vllm_import import VLLMImporter


class BackendFormat(str, Enum):
    """Supported benchmark data formats."""

    AUTO = "auto"
    NATIVE = "native"
    VLLM = "vllm"
    SGLANG = "sglang"
    TRTLLM = "trtllm"


class RankCriteria(str, Enum):
    """Criteria for ranking backends."""

    TTFT_P99 = "ttft_p99"
    TPOT_P99 = "tpot_p99"
    TOTAL_LATENCY_P99 = "total_latency_p99"
    THROUGHPUT = "throughput"


class BackendMetrics(BaseModel):
    """Aggregated metrics for a single backend benchmark run."""

    backend_label: str = Field(..., description="User-provided label for this backend")
    format_detected: str = Field(..., description="Detected input format")
    num_prefill_instances: int = Field(..., ge=1)
    num_decode_instances: int = Field(..., ge=1)
    total_instances: int = Field(..., ge=2)
    measured_qps: float = Field(..., gt=0)
    request_count: int = Field(..., ge=1)
    ttft_p50_ms: float = Field(..., ge=0)
    ttft_p95_ms: float = Field(..., ge=0)
    ttft_p99_ms: float = Field(..., ge=0)
    tpot_p50_ms: float = Field(..., ge=0)
    tpot_p95_ms: float = Field(..., ge=0)
    tpot_p99_ms: float = Field(..., ge=0)
    total_latency_p50_ms: float = Field(..., ge=0)
    total_latency_p95_ms: float = Field(..., ge=0)
    total_latency_p99_ms: float = Field(..., ge=0)
    throughput_rps: float = Field(..., ge=0, description="Effective throughput (requests/sec)")
    avg_prompt_tokens: float = Field(..., ge=0)
    avg_output_tokens: float = Field(..., ge=0)


class BackendRanking(BaseModel):
    """Ranking entry for a backend."""

    backend_label: str = Field(..., description="Backend label")
    rank: int = Field(..., ge=1, description="Rank (1 = best)")
    criteria: str = Field(..., description="Ranking criteria used")
    score: float = Field(..., description="Score value (lower is better for latency)")
    recommendation: str = Field(..., description="Human-readable recommendation")


class SLAResult(BaseModel):
    """SLA compliance result for a backend."""

    backend_label: str
    ttft_p99_pass: bool = Field(..., description="TTFT P99 meets SLA")
    tpot_p99_pass: bool = Field(..., description="TPOT P99 meets SLA")
    total_latency_p99_pass: bool = Field(..., description="Total latency P99 meets SLA")
    meets_all: bool = Field(..., description="All SLA constraints met")


class BackendComparisonConfig(BaseModel):
    """Configuration for backend comparison."""

    rank_by: RankCriteria = Field(
        RankCriteria.TTFT_P99, description="Criteria for ranking backends"
    )
    sla_ttft_p99_ms: Optional[float] = Field(
        None, ge=0, description="SLA threshold for TTFT P99 (ms)"
    )
    sla_tpot_p99_ms: Optional[float] = Field(
        None, ge=0, description="SLA threshold for TPOT P99 (ms)"
    )
    sla_total_latency_p99_ms: Optional[float] = Field(
        None, ge=0, description="SLA threshold for total latency P99 (ms)"
    )


class BackendComparisonReport(BaseModel):
    """Complete comparison report across backends."""

    metrics: list[BackendMetrics] = Field(default_factory=list)
    rankings: list[BackendRanking] = Field(default_factory=list)
    sla_results: list[SLAResult] = Field(default_factory=list)
    best_backend: str = Field(..., description="Best backend by ranking criteria")
    rank_criteria: str = Field(..., description="Criteria used for ranking")


def _detect_format(data: object) -> BackendFormat:
    """Auto-detect the benchmark data format."""
    if isinstance(data, dict):
        # Native format has 'metadata' and 'requests' at top level
        if "metadata" in data and "requests" in data:
            return BackendFormat.NATIVE
    if _detect_trtllm_format(data):
        return BackendFormat.TRTLLM
    if _detect_sglang_format(data):
        return BackendFormat.SGLANG
    # vLLM detection: list of dicts with 'request_latency' field
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, dict) and "request_latency" in first:
            return BackendFormat.VLLM
    return BackendFormat.NATIVE


def _load_benchmark(
    path: str | Path,
    fmt: BackendFormat = BackendFormat.AUTO,
    num_prefill: int = 1,
    num_decode: int = 1,
    total_instances: int = 2,
    measured_qps: float = 1.0,
) -> tuple[BenchmarkData, str]:
    """Load benchmark data from file, auto-detecting format.

    Returns (BenchmarkData, detected_format_name).
    """
    raw = json.loads(Path(path).read_text())

    if fmt == BackendFormat.AUTO:
        fmt = _detect_format(raw)

    if fmt == BackendFormat.NATIVE:
        bd = BenchmarkData.model_validate(raw)
        return bd, "native"
    elif fmt == BackendFormat.VLLM:
        importer = VLLMImporter()
        result = importer.import_data(raw)
        return result.benchmark_data, "vllm"
    elif fmt == BackendFormat.SGLANG:
        result = import_sglang_data(
            raw,
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=total_instances,
            measured_qps=measured_qps,
        )
        return result.benchmark_data, "sglang"
    elif fmt == BackendFormat.TRTLLM:
        result = import_trtllm_data(
            raw,
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=total_instances,
            measured_qps=measured_qps,
        )
        return result.benchmark_data, "trtllm"
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _compute_metrics(
    bd: BenchmarkData, label: str, format_name: str
) -> BackendMetrics:
    """Compute aggregated metrics from benchmark data."""
    ttfts = np.array([r.ttft_ms for r in bd.requests])
    tpots = np.array([r.tpot_ms for r in bd.requests])
    totals = np.array([r.total_latency_ms for r in bd.requests])
    prompts = np.array([r.prompt_tokens for r in bd.requests])
    outputs = np.array([r.output_tokens for r in bd.requests])

    return BackendMetrics(
        backend_label=label,
        format_detected=format_name,
        num_prefill_instances=bd.metadata.num_prefill_instances,
        num_decode_instances=bd.metadata.num_decode_instances,
        total_instances=bd.metadata.total_instances,
        measured_qps=bd.metadata.measured_qps,
        request_count=len(bd.requests),
        ttft_p50_ms=float(np.percentile(ttfts, 50)),
        ttft_p95_ms=float(np.percentile(ttfts, 95)),
        ttft_p99_ms=float(np.percentile(ttfts, 99)),
        tpot_p50_ms=float(np.percentile(tpots, 50)),
        tpot_p95_ms=float(np.percentile(tpots, 95)),
        tpot_p99_ms=float(np.percentile(tpots, 99)),
        total_latency_p50_ms=float(np.percentile(totals, 50)),
        total_latency_p95_ms=float(np.percentile(totals, 95)),
        total_latency_p99_ms=float(np.percentile(totals, 99)),
        throughput_rps=bd.metadata.measured_qps,
        avg_prompt_tokens=float(np.mean(prompts)),
        avg_output_tokens=float(np.mean(outputs)),
    )


def _check_sla(
    metrics: BackendMetrics, config: BackendComparisonConfig
) -> SLAResult:
    """Check SLA compliance for a backend."""
    ttft_pass = True
    tpot_pass = True
    total_pass = True

    if config.sla_ttft_p99_ms is not None:
        ttft_pass = metrics.ttft_p99_ms <= config.sla_ttft_p99_ms
    if config.sla_tpot_p99_ms is not None:
        tpot_pass = metrics.tpot_p99_ms <= config.sla_tpot_p99_ms
    if config.sla_total_latency_p99_ms is not None:
        total_pass = metrics.total_latency_p99_ms <= config.sla_total_latency_p99_ms

    return SLAResult(
        backend_label=metrics.backend_label,
        ttft_p99_pass=ttft_pass,
        tpot_p99_pass=tpot_pass,
        total_latency_p99_pass=total_pass,
        meets_all=ttft_pass and tpot_pass and total_pass,
    )


def _rank_backends(
    metrics_list: list[BackendMetrics],
    criteria: RankCriteria,
) -> list[BackendRanking]:
    """Rank backends by the chosen criteria."""
    score_map = {
        RankCriteria.TTFT_P99: lambda m: m.ttft_p99_ms,
        RankCriteria.TPOT_P99: lambda m: m.tpot_p99_ms,
        RankCriteria.TOTAL_LATENCY_P99: lambda m: m.total_latency_p99_ms,
        RankCriteria.THROUGHPUT: lambda m: -m.throughput_rps,  # negative: higher is better
    }

    key_fn = score_map[criteria]
    scored = [(m, key_fn(m)) for m in metrics_list]
    scored.sort(key=lambda x: x[1])

    rankings = []
    for rank_idx, (m, score) in enumerate(scored):
        if rank_idx == 0:
            rec = f"Best: {m.backend_label} wins on {criteria.value}"
        else:
            rec = f"{m.backend_label}: rank {rank_idx + 1} by {criteria.value}"
        rankings.append(
            BackendRanking(
                backend_label=m.backend_label,
                rank=rank_idx + 1,
                criteria=criteria.value,
                score=abs(score),
                recommendation=rec,
            )
        )

    return rankings


class BackendComparator:
    """Compare benchmark results across different serving backends."""

    def __init__(self, config: Optional[BackendComparisonConfig] = None):
        self._config = config or BackendComparisonConfig()

    def compare(
        self,
        benchmark_paths: list[str],
        backend_labels: list[str],
        formats: Optional[list[BackendFormat]] = None,
    ) -> BackendComparisonReport:
        """Compare multiple backends and produce a comparison report.

        Args:
            benchmark_paths: Paths to benchmark JSON files.
            backend_labels: Labels for each backend (e.g., "vllm", "sglang").
            formats: Optional format hints per file (default: auto-detect all).

        Returns:
            BackendComparisonReport with metrics, rankings, and SLA results.
        """
        if len(benchmark_paths) != len(backend_labels):
            raise ValueError(
                f"Number of benchmarks ({len(benchmark_paths)}) must match "
                f"number of labels ({len(backend_labels)})"
            )
        if len(benchmark_paths) < 2:
            raise ValueError("At least 2 benchmarks required for comparison")

        if formats is None:
            formats = [BackendFormat.AUTO] * len(benchmark_paths)
        elif len(formats) != len(benchmark_paths):
            raise ValueError(
                f"Number of formats ({len(formats)}) must match "
                f"number of benchmarks ({len(benchmark_paths)})"
            )

        metrics_list: list[BackendMetrics] = []
        for path, label, fmt in zip(benchmark_paths, backend_labels, formats):
            bd, detected = _load_benchmark(path, fmt)
            m = _compute_metrics(bd, label, detected)
            metrics_list.append(m)

        rankings = _rank_backends(metrics_list, self._config.rank_by)

        sla_results = [_check_sla(m, self._config) for m in metrics_list]

        best = rankings[0].backend_label if rankings else backend_labels[0]

        return BackendComparisonReport(
            metrics=metrics_list,
            rankings=rankings,
            sla_results=sla_results,
            best_backend=best,
            rank_criteria=self._config.rank_by.value,
        )


def compare_backends(
    benchmark_paths: list[str],
    backend_labels: list[str],
    rank_by: str = "ttft_p99",
    formats: Optional[list[str]] = None,
    sla_ttft_p99_ms: Optional[float] = None,
    sla_tpot_p99_ms: Optional[float] = None,
    sla_total_latency_p99_ms: Optional[float] = None,
) -> BackendComparisonReport:
    """Programmatic API: compare backends across benchmark files.

    Args:
        benchmark_paths: Paths to benchmark JSON files.
        backend_labels: Labels for each backend.
        rank_by: Ranking criteria (ttft_p99, tpot_p99, total_latency_p99, throughput).
        formats: Optional format per file (auto, native, vllm, sglang, trtllm).
        sla_ttft_p99_ms: SLA threshold for TTFT P99.
        sla_tpot_p99_ms: SLA threshold for TPOT P99.
        sla_total_latency_p99_ms: SLA threshold for total latency P99.

    Returns:
        BackendComparisonReport with metrics, rankings, and SLA results.
    """
    config = BackendComparisonConfig(
        rank_by=RankCriteria(rank_by),
        sla_ttft_p99_ms=sla_ttft_p99_ms,
        sla_tpot_p99_ms=sla_tpot_p99_ms,
        sla_total_latency_p99_ms=sla_total_latency_p99_ms,
    )
    fmt_enums = None
    if formats:
        fmt_enums = [BackendFormat(f) for f in formats]

    comparator = BackendComparator(config)
    return comparator.compare(benchmark_paths, backend_labels, fmt_enums)

"""Retry policy optimizer.

Sweeps retry parameters (max retries, thresholds, backoff strategies)
to find the configuration that maximizes effective goodput while keeping
load amplification within a specified budget.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData
from .retry_sim import BackoffType, RetryConfig, RetrySimulator


class RetryOptimizerConfig(BaseModel):
    """Configuration for retry policy optimization."""

    max_amplification: float = Field(
        2.0, gt=1.0, description="Maximum allowed load amplification factor"
    )
    max_retries_range: list[int] = Field(
        default_factory=lambda: [1, 2, 3, 5],
        description="Max retries values to search",
    )
    threshold_percentiles: list[float] = Field(
        default_factory=lambda: [90.0, 95.0, 99.0],
        description="Percentiles of latency distributions to use as retry thresholds",
    )
    backoff_types: list[BackoffType] = Field(
        default_factory=lambda: [BackoffType.CONSTANT, BackoffType.EXPONENTIAL],
        description="Backoff strategies to evaluate",
    )
    backoff_ms_values: list[float] = Field(
        default_factory=lambda: [50.0, 100.0, 200.0],
        description="Backoff base delay values to search (ms)",
    )
    sla_ttft_ms: float | None = Field(None, ge=0, description="SLA TTFT threshold (ms)")
    sla_tpot_ms: float | None = Field(None, ge=0, description="SLA TPOT threshold (ms)")
    sla_total_ms: float | None = Field(None, ge=0, description="SLA total latency threshold (ms)")


class PolicyCandidate(BaseModel):
    """A single retry policy candidate with measured outcomes."""

    config: RetryConfig
    effective_goodput: float = Field(..., ge=0, le=1)
    original_goodput: float = Field(..., ge=0, le=1)
    goodput_improvement: float = Field(..., description="Effective - original goodput")
    amplification_factor: float = Field(..., ge=1.0)
    retry_rate: float = Field(..., ge=0, le=1)
    within_budget: bool = Field(..., description="Whether amplification is within budget")


class OptimalRetryPolicy(BaseModel):
    """The recommended optimal retry policy."""

    config: RetryConfig
    effective_goodput: float = Field(..., ge=0, le=1)
    goodput_improvement: float
    amplification_factor: float = Field(..., ge=1.0)


class RetryOptimizerReport(BaseModel):
    """Complete retry optimization report."""

    optimizer_config: RetryOptimizerConfig
    candidates_evaluated: int = Field(..., ge=0)
    candidates_within_budget: int = Field(..., ge=0)
    best_policy: OptimalRetryPolicy | None = Field(
        None, description="Best policy within amplification budget (None if none qualify)"
    )
    pareto_frontier: list[PolicyCandidate] = Field(
        default_factory=list,
        description="Pareto-optimal candidates (goodput vs amplification tradeoff)",
    )
    all_candidates: list[PolicyCandidate] = Field(default_factory=list)
    recommendation: str


class RetryOptimizer:
    """Find optimal retry policy via grid search over parameters."""

    def __init__(self, config: RetryOptimizerConfig) -> None:
        if (
            config.sla_ttft_ms is None
            and config.sla_tpot_ms is None
            and config.sla_total_ms is None
        ):
            raise ValueError(
                "At least one SLA threshold must be specified "
                "(sla_ttft_ms, sla_tpot_ms, or sla_total_ms)"
            )
        self._config = config

    def optimize(self, data: BenchmarkData) -> RetryOptimizerReport:
        """Run grid search and return optimization report."""
        import numpy as np

        cfg = self._config
        requests = data.requests

        if len(requests) == 0:
            return RetryOptimizerReport(
                optimizer_config=cfg,
                candidates_evaluated=0,
                candidates_within_budget=0,
                best_policy=None,
                pareto_frontier=[],
                all_candidates=[],
                recommendation="No requests to analyze.",
            )

        # Compute latency percentiles for threshold generation
        ttft_vals = np.array([r.ttft_ms for r in requests])
        tpot_vals = np.array([r.tpot_ms for r in requests])
        total_vals = np.array([r.total_latency_ms for r in requests])

        ttft_thresholds: list[float | None] = [None]
        tpot_thresholds: list[float | None] = [None]
        total_thresholds: list[float | None] = [None]

        if cfg.sla_ttft_ms is not None:
            ttft_thresholds = [
                float(np.percentile(ttft_vals, p)) for p in cfg.threshold_percentiles
            ]
        if cfg.sla_tpot_ms is not None:
            tpot_thresholds = [
                float(np.percentile(tpot_vals, p)) for p in cfg.threshold_percentiles
            ]
        if cfg.sla_total_ms is not None:
            total_thresholds = [
                float(np.percentile(total_vals, p)) for p in cfg.threshold_percentiles
            ]

        candidates: list[PolicyCandidate] = []

        for max_retries in cfg.max_retries_range:
            for backoff_type in cfg.backoff_types:
                for backoff_ms in cfg.backoff_ms_values:
                    for ttft_th in ttft_thresholds:
                        for tpot_th in tpot_thresholds:
                            for total_th in total_thresholds:
                                # Skip if no threshold set
                                if (
                                    ttft_th is None
                                    and tpot_th is None
                                    and total_th is None
                                ):
                                    continue

                                retry_cfg = RetryConfig(
                                    max_retries=max_retries,
                                    retry_threshold_ttft_ms=ttft_th,
                                    retry_threshold_tpot_ms=tpot_th,
                                    retry_threshold_total_ms=total_th,
                                    backoff_ms=backoff_ms,
                                    backoff_type=backoff_type,
                                )

                                try:
                                    simulator = RetrySimulator(retry_cfg)
                                    report = simulator.simulate(data)
                                except (ValueError, Exception):
                                    continue

                                la = report.load_amplification
                                within = la.amplification_factor <= cfg.max_amplification

                                candidates.append(PolicyCandidate(
                                    config=retry_cfg,
                                    effective_goodput=report.effective_goodput,
                                    original_goodput=report.original_goodput,
                                    goodput_improvement=(
                                        report.effective_goodput - report.original_goodput
                                    ),
                                    amplification_factor=la.amplification_factor,
                                    retry_rate=la.retry_rate,
                                    within_budget=within,
                                ))

        within_budget = [c for c in candidates if c.within_budget]

        # Best policy: highest effective goodput within budget
        best: OptimalRetryPolicy | None = None
        if within_budget:
            best_candidate = max(
                within_budget,
                key=lambda c: (c.effective_goodput, -c.amplification_factor),
            )
            best = OptimalRetryPolicy(
                config=best_candidate.config,
                effective_goodput=best_candidate.effective_goodput,
                goodput_improvement=best_candidate.goodput_improvement,
                amplification_factor=best_candidate.amplification_factor,
            )

        # Pareto frontier: non-dominated in (goodput↑, amplification↓)
        pareto = _compute_pareto(candidates)

        # Recommendation
        if not candidates:
            rec = "No valid retry configurations found."
        elif best is None:
            rec = (
                f"No retry policy achieves goodput improvement within "
                f"{cfg.max_amplification:.1f}x amplification budget. "
                f"Consider increasing the budget or improving baseline performance."
            )
        elif best.goodput_improvement < 0.01:
            rec = (
                f"Best policy provides minimal goodput improvement "
                f"(+{best.goodput_improvement:.1%}). "
                f"Retries may not be worthwhile for this workload."
            )
        else:
            bc = best.config
            rec = (
                f"Recommended: max_retries={bc.max_retries}, "
                f"backoff={bc.backoff_ms:.0f}ms ({bc.backoff_type.value}). "
                f"Achieves {best.effective_goodput:.1%} goodput "
                f"(+{best.goodput_improvement:.1%}) at "
                f"{best.amplification_factor:.2f}x amplification."
            )

        return RetryOptimizerReport(
            optimizer_config=cfg,
            candidates_evaluated=len(candidates),
            candidates_within_budget=len(within_budget),
            best_policy=best,
            pareto_frontier=pareto,
            all_candidates=candidates,
            recommendation=rec,
        )


def _compute_pareto(candidates: list[PolicyCandidate]) -> list[PolicyCandidate]:
    """Find Pareto-optimal candidates (maximize goodput, minimize amplification)."""
    if not candidates:
        return []

    pareto: list[PolicyCandidate] = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if (
                other.effective_goodput >= c.effective_goodput
                and other.amplification_factor <= c.amplification_factor
                and (
                    other.effective_goodput > c.effective_goodput
                    or other.amplification_factor < c.amplification_factor
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(c)

    # Sort by amplification ascending
    pareto.sort(key=lambda c: c.amplification_factor)
    return pareto


def optimize_retry_policy(
    benchmark_path: str,
    max_amplification: float = 2.0,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_ms: float | None = None,
    max_retries_range: list[int] | None = None,
) -> dict:
    """Programmatic API for retry policy optimization.

    Args:
        benchmark_path: Path to benchmark JSON file.
        max_amplification: Maximum allowed load amplification factor.
        sla_ttft_ms: SLA TTFT threshold (ms).
        sla_tpot_ms: SLA TPOT threshold (ms).
        sla_total_ms: SLA total latency threshold (ms).
        max_retries_range: List of max_retries values to search.

    Returns:
        dict: RetryOptimizerReport as a dictionary.
    """
    from .bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    kwargs: dict = {
        "max_amplification": max_amplification,
        "sla_ttft_ms": sla_ttft_ms,
        "sla_tpot_ms": sla_tpot_ms,
        "sla_total_ms": sla_total_ms,
    }
    if max_retries_range is not None:
        kwargs["max_retries_range"] = max_retries_range

    config = RetryOptimizerConfig(**kwargs)
    optimizer = RetryOptimizer(config)
    report = optimizer.optimize(data)
    return report.model_dump()

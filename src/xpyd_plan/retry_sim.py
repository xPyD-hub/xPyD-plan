"""Request retry impact simulator.

Models how retry policies would affect latency distributions,
effective goodput, and total load amplification based on observed
benchmark data.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class BackoffType(str, Enum):
    """Backoff strategy for retries."""

    CONSTANT = "constant"
    EXPONENTIAL = "exponential"


class RetryConfig(BaseModel):
    """Configuration for retry policy simulation."""

    max_retries: int = Field(3, ge=0, description="Maximum retry attempts per request")
    retry_threshold_ttft_ms: float | None = Field(
        None, ge=0, description="TTFT threshold triggering retry (ms)"
    )
    retry_threshold_tpot_ms: float | None = Field(
        None, ge=0, description="TPOT threshold triggering retry (ms)"
    )
    retry_threshold_total_ms: float | None = Field(
        None, ge=0, description="Total latency threshold triggering retry (ms)"
    )
    backoff_ms: float = Field(
        100.0, ge=0, description="Base backoff delay in ms"
    )
    backoff_type: BackoffType = Field(
        BackoffType.CONSTANT, description="Backoff strategy"
    )


class LoadAmplification(BaseModel):
    """Load amplification due to retries."""

    original_requests: int = Field(..., ge=0)
    total_requests_with_retries: int = Field(..., ge=0)
    amplification_factor: float = Field(..., ge=1.0)
    total_retry_attempts: int = Field(..., ge=0)
    retry_rate: float = Field(
        ..., ge=0, le=1, description="Fraction of original requests that needed retries"
    )


class RetryImpact(BaseModel):
    """Per-metric impact of retries on latency distribution."""

    metric: str = Field(..., description="Metric name (ttft, tpot, total_latency)")
    original_p50: float = Field(..., ge=0)
    original_p95: float = Field(..., ge=0)
    original_p99: float = Field(..., ge=0)
    post_retry_p50: float = Field(..., ge=0)
    post_retry_p95: float = Field(..., ge=0)
    post_retry_p99: float = Field(..., ge=0)


class RetrySimReport(BaseModel):
    """Complete retry simulation report."""

    config: RetryConfig = Field(..., description="Retry configuration used")
    load_amplification: LoadAmplification = Field(
        ..., description="Load amplification statistics"
    )
    effective_goodput: float = Field(
        ..., ge=0, le=1,
        description="Fraction of requests meeting SLA after retries",
    )
    original_goodput: float = Field(
        ..., ge=0, le=1,
        description="Fraction of requests meeting SLA without retries",
    )
    latency_impact: list[RetryImpact] = Field(
        default_factory=list, description="Per-metric latency impact"
    )
    recommendation: str = Field(..., description="Human-readable recommendation")


def _exceeds_threshold(
    ttft_ms: float,
    tpot_ms: float,
    total_latency_ms: float,
    config: RetryConfig,
) -> bool:
    """Check if a request exceeds any configured retry threshold."""
    if config.retry_threshold_ttft_ms is not None and ttft_ms > config.retry_threshold_ttft_ms:
        return True
    if config.retry_threshold_tpot_ms is not None and tpot_ms > config.retry_threshold_tpot_ms:
        return True
    if (
        config.retry_threshold_total_ms is not None
        and total_latency_ms > config.retry_threshold_total_ms
    ):
        return True
    return False


def _backoff_delay(attempt: int, config: RetryConfig) -> float:
    """Compute backoff delay for a given retry attempt (0-indexed)."""
    if config.backoff_type == BackoffType.CONSTANT:
        return config.backoff_ms
    # Exponential: backoff_ms * 2^attempt
    return config.backoff_ms * (2 ** attempt)


class RetrySimulator:
    """Simulate retry impact on benchmark data.

    Given benchmark results and a retry policy, model how retries would
    affect the latency distribution and total load.
    """

    def __init__(self, config: RetryConfig) -> None:
        if (
            config.retry_threshold_ttft_ms is None
            and config.retry_threshold_tpot_ms is None
            and config.retry_threshold_total_ms is None
        ):
            raise ValueError(
                "At least one retry threshold must be specified "
                "(retry_threshold_ttft_ms, retry_threshold_tpot_ms, "
                "or retry_threshold_total_ms)"
            )
        self._config = config

    def simulate(self, data: BenchmarkData) -> RetrySimReport:
        """Run retry simulation on benchmark data."""
        import numpy as np

        requests = data.requests
        total = len(requests)

        if total == 0:
            return RetrySimReport(
                config=self._config,
                load_amplification=LoadAmplification(
                    original_requests=0,
                    total_requests_with_retries=0,
                    amplification_factor=1.0,
                    total_retry_attempts=0,
                    retry_rate=0.0,
                ),
                effective_goodput=1.0,
                original_goodput=1.0,
                latency_impact=[],
                recommendation="No requests to analyze.",
            )

        cfg = self._config

        # For each request, simulate retries
        # We assume each retry attempt has the same latency distribution
        # (sampled from the original data). The total latency for a retried
        # request = original latency + backoff delays + retry latency.
        # Simplified model: if a request exceeds threshold, each retry
        # randomly picks another request's latency values.

        original_ttft = np.array([r.ttft_ms for r in requests])
        original_tpot = np.array([r.tpot_ms for r in requests])
        original_total = np.array([r.total_latency_ms for r in requests])

        # Track which requests need retries and final effective latencies
        rng = np.random.RandomState(42)
        final_ttft = np.copy(original_ttft)
        final_tpot = np.copy(original_tpot)
        final_total = np.copy(original_total)

        total_retry_attempts = 0
        requests_needing_retry = 0
        requests_passing_after_retry = 0

        # Original goodput (no retries)
        original_pass_count = sum(
            1 for r in requests
            if not _exceeds_threshold(r.ttft_ms, r.tpot_ms, r.total_latency_ms, cfg)
        )
        original_goodput = original_pass_count / total

        for i in range(total):
            if not _exceeds_threshold(
                original_ttft[i], original_tpot[i], original_total[i], cfg
            ):
                continue

            requests_needing_retry += 1
            succeeded = False
            cumulative_backoff = 0.0

            for attempt in range(cfg.max_retries):
                total_retry_attempts += 1
                backoff = _backoff_delay(attempt, cfg)
                cumulative_backoff += backoff

                # Sample a random request's latency as the retry outcome
                idx = rng.randint(0, total)
                retry_ttft = original_ttft[idx]
                retry_tpot = original_tpot[idx]
                retry_total = original_total[idx] + cumulative_backoff

                if not _exceeds_threshold(retry_ttft, retry_tpot, retry_total, cfg):
                    # Retry succeeded
                    final_ttft[i] = retry_ttft
                    final_tpot[i] = retry_tpot
                    final_total[i] = retry_total
                    succeeded = True
                    break

            if not succeeded:
                # Use last retry attempt values (worst case with all backoff)
                final_total[i] = original_total[i] + cumulative_backoff

            if succeeded:
                requests_passing_after_retry += 1

        # Effective goodput after retries
        effective_pass = sum(
            1 for i in range(total)
            if not _exceeds_threshold(final_ttft[i], final_tpot[i], final_total[i], cfg)
        )
        effective_goodput = effective_pass / total

        # Load amplification
        total_with_retries = total + total_retry_attempts
        amplification = total_with_retries / total if total > 0 else 1.0
        retry_rate = requests_needing_retry / total if total > 0 else 0.0

        # Latency impact
        impacts: list[RetryImpact] = []
        for metric_name, orig_arr, final_arr in [
            ("ttft", original_ttft, final_ttft),
            ("tpot", original_tpot, final_tpot),
            ("total_latency", original_total, final_total),
        ]:
            impacts.append(RetryImpact(
                metric=metric_name,
                original_p50=float(np.percentile(orig_arr, 50)),
                original_p95=float(np.percentile(orig_arr, 95)),
                original_p99=float(np.percentile(orig_arr, 99)),
                post_retry_p50=float(np.percentile(final_arr, 50)),
                post_retry_p95=float(np.percentile(final_arr, 95)),
                post_retry_p99=float(np.percentile(final_arr, 99)),
            ))

        # Recommendation
        goodput_delta = effective_goodput - original_goodput
        if retry_rate == 0:
            rec = "No requests exceed retry thresholds. Retries have no impact."
        elif amplification > 2.0:
            rec = (
                f"Retry policy causes {amplification:.1f}x load amplification "
                f"({retry_rate:.1%} retry rate). Risk of cascading overload. "
                f"Consider raising thresholds or reducing max retries."
            )
        elif goodput_delta < 0.01:
            rec = (
                f"Retries provide minimal goodput improvement (+{goodput_delta:.1%}). "
                f"The {amplification:.2f}x load amplification may not be worth it. "
                f"Consider alternative approaches (scaling, optimization)."
            )
        else:
            rec = (
                f"Retries improve goodput by {goodput_delta:.1%} "
                f"(from {original_goodput:.1%} to {effective_goodput:.1%}) "
                f"with {amplification:.2f}x load amplification. "
                f"Acceptable tradeoff if backend can handle the extra load."
            )

        return RetrySimReport(
            config=cfg,
            load_amplification=LoadAmplification(
                original_requests=total,
                total_requests_with_retries=total_with_retries,
                amplification_factor=amplification,
                total_retry_attempts=total_retry_attempts,
                retry_rate=retry_rate,
            ),
            effective_goodput=effective_goodput,
            original_goodput=original_goodput,
            latency_impact=impacts,
            recommendation=rec,
        )


def simulate_retries(
    benchmark_path: str,
    max_retries: int = 3,
    retry_threshold_ttft_ms: float | None = None,
    retry_threshold_tpot_ms: float | None = None,
    retry_threshold_total_ms: float | None = None,
    backoff_ms: float = 100.0,
    backoff_type: str = "constant",
) -> dict:
    """Programmatic API for retry simulation.

    Args:
        benchmark_path: Path to benchmark JSON file.
        max_retries: Maximum retry attempts per request.
        retry_threshold_ttft_ms: TTFT threshold triggering retry (ms).
        retry_threshold_tpot_ms: TPOT threshold triggering retry (ms).
        retry_threshold_total_ms: Total latency threshold triggering retry (ms).
        backoff_ms: Base backoff delay in ms.
        backoff_type: Backoff strategy ('constant' or 'exponential').

    Returns:
        dict: RetrySimReport as a dictionary.
    """
    from .bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    config = RetryConfig(
        max_retries=max_retries,
        retry_threshold_ttft_ms=retry_threshold_ttft_ms,
        retry_threshold_tpot_ms=retry_threshold_tpot_ms,
        retry_threshold_total_ms=retry_threshold_total_ms,
        backoff_ms=backoff_ms,
        backoff_type=BackoffType(backoff_type),
    )
    simulator = RetrySimulator(config)
    report = simulator.simulate(data)
    return report.model_dump()

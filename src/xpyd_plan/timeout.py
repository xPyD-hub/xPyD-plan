"""Request timeout analysis — identify and characterize requests exceeding latency bounds."""

from __future__ import annotations

from enum import Enum
from statistics import mean, median

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData, BenchmarkRequest


class TimeoutSeverity(str, Enum):
    """Severity classification based on timeout rate."""

    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TimeoutConfig(BaseModel):
    """Configuration for timeout thresholds."""

    ttft_ms: float | None = Field(default=None, ge=0, description="TTFT timeout threshold in ms")
    tpot_ms: float | None = Field(default=None, ge=0, description="TPOT timeout threshold in ms")
    total_latency_ms: float | None = Field(
        default=None, ge=0, description="Total latency timeout threshold in ms"
    )


class TimeoutEvent(BaseModel):
    """A single timed-out request."""

    request_id: str = Field(..., description="Request identifier")
    index: int = Field(..., description="Index in the benchmark data")
    prompt_tokens: int = Field(..., description="Prompt token count")
    output_tokens: int = Field(..., description="Output token count")
    ttft_ms: float = Field(..., description="TTFT in ms")
    tpot_ms: float = Field(..., description="TPOT in ms")
    total_latency_ms: float = Field(..., description="Total latency in ms")
    timestamp: float = Field(..., description="Request timestamp")
    exceeded_metrics: list[str] = Field(..., description="Which thresholds were exceeded")


class TokenCharacterization(BaseModel):
    """Token distribution stats for timed-out requests."""

    prompt_tokens_mean: float = Field(..., description="Mean prompt tokens")
    prompt_tokens_median: float = Field(..., description="Median prompt tokens")
    prompt_tokens_min: int = Field(..., description="Min prompt tokens")
    prompt_tokens_max: int = Field(..., description="Max prompt tokens")
    output_tokens_mean: float = Field(..., description="Mean output tokens")
    output_tokens_median: float = Field(..., description="Median output tokens")
    output_tokens_min: int = Field(..., description="Min output tokens")
    output_tokens_max: int = Field(..., description="Max output tokens")


class TemporalCluster(BaseModel):
    """A temporal cluster of timeout events."""

    start_timestamp: float = Field(..., description="Cluster start time")
    end_timestamp: float = Field(..., description="Cluster end time")
    count: int = Field(..., ge=1, description="Number of timeouts in cluster")
    duration_seconds: float = Field(..., ge=0, description="Cluster duration in seconds")


class TimeoutReport(BaseModel):
    """Complete timeout analysis report."""

    total_requests: int = Field(..., description="Total requests in benchmark")
    timeout_count: int = Field(..., ge=0, description="Number of timed-out requests")
    timeout_rate: float = Field(..., ge=0, le=1, description="Fraction of requests that timed out")
    severity: TimeoutSeverity = Field(..., description="Timeout severity classification")
    config: TimeoutConfig = Field(..., description="Timeout thresholds used")
    events: list[TimeoutEvent] = Field(..., description="Individual timeout events")
    token_characterization: TokenCharacterization | None = Field(
        default=None, description="Token stats of timed-out requests"
    )
    temporal_clusters: list[TemporalCluster] = Field(
        default_factory=list, description="Temporal clusters of timeouts"
    )
    recommendation: str = Field(..., description="Actionable recommendation")
    per_metric_counts: dict[str, int] = Field(
        default_factory=dict, description="Timeout counts per metric"
    )


class TimeoutAnalyzer:
    """Identify and characterize requests exceeding latency thresholds."""

    def analyze(
        self,
        data: BenchmarkData,
        config: TimeoutConfig | None = None,
        ttft_ms: float | None = None,
        tpot_ms: float | None = None,
        total_latency_ms: float | None = None,
    ) -> TimeoutReport:
        """Analyze benchmark data for timeout events.

        Args:
            data: Benchmark data to analyze.
            config: Timeout configuration. If None, uses individual threshold args.
            ttft_ms: TTFT timeout threshold (overrides config).
            tpot_ms: TPOT timeout threshold (overrides config).
            total_latency_ms: Total latency timeout threshold (overrides config).

        Returns:
            TimeoutReport with analysis results.
        """
        if config is None:
            config = TimeoutConfig(
                ttft_ms=ttft_ms, tpot_ms=tpot_ms, total_latency_ms=total_latency_ms
            )
        else:
            if ttft_ms is not None:
                config = config.model_copy(update={"ttft_ms": ttft_ms})
            if tpot_ms is not None:
                config = config.model_copy(update={"tpot_ms": tpot_ms})
            if total_latency_ms is not None:
                config = config.model_copy(update={"total_latency_ms": total_latency_ms})

        events = self._find_timeouts(data.requests, config)
        timeout_count = len(events)
        total = len(data.requests)
        rate = timeout_count / total if total > 0 else 0.0
        severity = self._classify_severity(rate)

        per_metric: dict[str, int] = {}
        for metric in ("ttft_ms", "tpot_ms", "total_latency_ms"):
            per_metric[metric] = sum(1 for e in events if metric in e.exceeded_metrics)

        token_char = self._characterize_tokens(events) if events else None
        clusters = self._find_temporal_clusters(events) if events else []
        recommendation = self._recommend(severity, rate, per_metric, timeout_count, total)

        return TimeoutReport(
            total_requests=total,
            timeout_count=timeout_count,
            timeout_rate=rate,
            severity=severity,
            config=config,
            events=events,
            token_characterization=token_char,
            temporal_clusters=clusters,
            recommendation=recommendation,
            per_metric_counts=per_metric,
        )

    def _find_timeouts(
        self, requests: list[BenchmarkRequest], config: TimeoutConfig
    ) -> list[TimeoutEvent]:
        """Identify requests exceeding any configured threshold."""
        events: list[TimeoutEvent] = []
        for i, req in enumerate(requests):
            exceeded: list[str] = []
            if config.ttft_ms is not None and req.ttft_ms > config.ttft_ms:
                exceeded.append("ttft_ms")
            if config.tpot_ms is not None and req.tpot_ms > config.tpot_ms:
                exceeded.append("tpot_ms")
            if (
                config.total_latency_ms is not None
                and req.total_latency_ms > config.total_latency_ms
            ):
                exceeded.append("total_latency_ms")
            if exceeded:
                events.append(
                    TimeoutEvent(
                        request_id=req.request_id,
                        index=i,
                        prompt_tokens=req.prompt_tokens,
                        output_tokens=req.output_tokens,
                        ttft_ms=req.ttft_ms,
                        tpot_ms=req.tpot_ms,
                        total_latency_ms=req.total_latency_ms,
                        timestamp=req.timestamp,
                        exceeded_metrics=exceeded,
                    )
                )
        return events

    @staticmethod
    def _classify_severity(rate: float) -> TimeoutSeverity:
        """Classify timeout severity based on rate."""
        if rate == 0:
            return TimeoutSeverity.NONE
        if rate < 0.01:
            return TimeoutSeverity.LOW
        if rate < 0.05:
            return TimeoutSeverity.MODERATE
        if rate < 0.15:
            return TimeoutSeverity.HIGH
        return TimeoutSeverity.CRITICAL

    @staticmethod
    def _characterize_tokens(events: list[TimeoutEvent]) -> TokenCharacterization:
        """Compute token distribution statistics for timed-out requests."""
        prompt = [e.prompt_tokens for e in events]
        output = [e.output_tokens for e in events]
        return TokenCharacterization(
            prompt_tokens_mean=mean(prompt),
            prompt_tokens_median=median(prompt),
            prompt_tokens_min=min(prompt),
            prompt_tokens_max=max(prompt),
            output_tokens_mean=mean(output),
            output_tokens_median=median(output),
            output_tokens_min=min(output),
            output_tokens_max=max(output),
        )

    @staticmethod
    def _find_temporal_clusters(
        events: list[TimeoutEvent], gap_seconds: float = 5.0
    ) -> list[TemporalCluster]:
        """Group timeout events into temporal clusters.

        Events within `gap_seconds` of each other are grouped together.
        """
        if not events:
            return []

        sorted_events = sorted(events, key=lambda e: e.timestamp)
        clusters: list[TemporalCluster] = []
        cluster_start = sorted_events[0].timestamp
        cluster_end = sorted_events[0].timestamp
        cluster_count = 1

        for event in sorted_events[1:]:
            if event.timestamp - cluster_end <= gap_seconds:
                cluster_end = event.timestamp
                cluster_count += 1
            else:
                clusters.append(
                    TemporalCluster(
                        start_timestamp=cluster_start,
                        end_timestamp=cluster_end,
                        count=cluster_count,
                        duration_seconds=cluster_end - cluster_start,
                    )
                )
                cluster_start = event.timestamp
                cluster_end = event.timestamp
                cluster_count = 1

        clusters.append(
            TemporalCluster(
                start_timestamp=cluster_start,
                end_timestamp=cluster_end,
                count=cluster_count,
                duration_seconds=cluster_end - cluster_start,
            )
        )
        return clusters

    @staticmethod
    def _recommend(
        severity: TimeoutSeverity,
        rate: float,
        per_metric: dict[str, int],
        timeout_count: int,
        total: int,
    ) -> str:
        """Generate actionable recommendation."""
        if severity == TimeoutSeverity.NONE:
            return "No timeouts detected. All requests completed within thresholds."

        dominant = max(per_metric, key=lambda k: per_metric[k]) if per_metric else "unknown"
        dominant_label = {
            "ttft_ms": "TTFT (time to first token)",
            "tpot_ms": "TPOT (time per output token)",
            "total_latency_ms": "total latency",
        }.get(dominant, dominant)

        if severity == TimeoutSeverity.LOW:
            return (
                f"{timeout_count}/{total} requests ({rate:.1%}) timed out. "
                f"Primary bottleneck: {dominant_label}. "
                f"Low severity — monitor but likely acceptable."
            )
        if severity == TimeoutSeverity.MODERATE:
            return (
                f"{timeout_count}/{total} requests ({rate:.1%}) timed out. "
                f"Primary bottleneck: {dominant_label}. "
                f"Consider scaling {'prefill' if dominant == 'ttft_ms' else 'decode'} instances."
            )
        if severity == TimeoutSeverity.HIGH:
            return (
                f"{timeout_count}/{total} requests ({rate:.1%}) timed out. "
                f"Primary bottleneck: {dominant_label}. "
                f"Immediate action needed: increase instance count or adjust P:D ratio."
            )
        return (
            f"CRITICAL: {timeout_count}/{total} requests ({rate:.1%}) timed out. "
            f"Primary bottleneck: {dominant_label}. "
            f"Configuration is severely under-provisioned. Urgent rebalancing required."
        )


def analyze_timeouts(
    data: BenchmarkData,
    ttft_ms: float | None = None,
    tpot_ms: float | None = None,
    total_latency_ms: float | None = None,
) -> TimeoutReport:
    """Programmatic API for timeout analysis.

    Args:
        data: Benchmark data to analyze.
        ttft_ms: TTFT timeout threshold in ms.
        tpot_ms: TPOT timeout threshold in ms.
        total_latency_ms: Total latency timeout threshold in ms.

    Returns:
        TimeoutReport with analysis results.
    """
    analyzer = TimeoutAnalyzer()
    return analyzer.analyze(
        data, ttft_ms=ttft_ms, tpot_ms=tpot_ms, total_latency_ms=total_latency_ms
    )

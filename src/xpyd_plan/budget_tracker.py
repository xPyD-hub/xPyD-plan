"""Latency budget tracking — how close are requests to exhausting SLA limits."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class BudgetStatus(str, Enum):
    """Budget consumption classification."""

    COMFORTABLE = "COMFORTABLE"  # <50% consumed
    MODERATE = "MODERATE"  # 50-80% consumed
    NEAR_MISS = "NEAR_MISS"  # 80-100% consumed
    EXCEEDED = "EXCEEDED"  # >100% consumed


class RequestBudget(BaseModel):
    """Per-request budget consumption."""

    request_id: str = Field(..., description="Request identifier")
    ttft_ratio: float | None = Field(None, description="TTFT budget consumption ratio")
    tpot_ratio: float | None = Field(None, description="TPOT budget consumption ratio")
    total_ratio: float | None = Field(None, description="Total latency budget consumption ratio")
    worst_metric: str = Field(..., description="Metric with highest consumption")
    worst_ratio: float = Field(..., description="Highest consumption ratio")
    status: BudgetStatus = Field(..., description="Budget status classification")


class BudgetDistribution(BaseModel):
    """Distribution statistics for budget consumption ratios."""

    metric: str = Field(..., description="Metric name")
    mean: float = Field(..., description="Mean consumption ratio")
    p50: float = Field(..., description="P50 consumption ratio")
    p95: float = Field(..., description="P95 consumption ratio")
    p99: float = Field(..., description="P99 consumption ratio")
    max: float = Field(..., description="Max consumption ratio")
    near_miss_count: int = Field(..., description="Requests above near-miss threshold")
    exceeded_count: int = Field(..., description="Requests exceeding SLA")


class BudgetAlert(BaseModel):
    """Alert about budget consumption patterns."""

    message: str = Field(..., description="Alert message")
    severity: str = Field(..., description="Alert severity: INFO, WARNING, CRITICAL")
    metric: str | None = Field(None, description="Related metric if applicable")


class BudgetReport(BaseModel):
    """Complete latency budget tracking report."""

    total_requests: int = Field(..., description="Total requests analyzed")
    near_miss_threshold: float = Field(..., description="Near-miss threshold (0-1)")
    near_miss_count: int = Field(..., description="Total near-miss requests")
    exceeded_count: int = Field(..., description="Total SLA-exceeded requests")
    comfortable_count: int = Field(..., description="Requests with comfortable budget")
    distributions: list[BudgetDistribution] = Field(
        default_factory=list, description="Per-metric distribution stats"
    )
    worst_requests: list[RequestBudget] = Field(
        default_factory=list, description="Top worst requests by budget consumption"
    )
    alerts: list[BudgetAlert] = Field(default_factory=list, description="Budget alerts")


def _classify(ratio: float, near_miss_threshold: float) -> BudgetStatus:
    if ratio > 1.0:
        return BudgetStatus.EXCEEDED
    if ratio > near_miss_threshold:
        return BudgetStatus.NEAR_MISS
    if ratio > 0.5:
        return BudgetStatus.MODERATE
    return BudgetStatus.COMFORTABLE


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile from sorted values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = p / 100.0 * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class LatencyBudgetTracker:
    """Track how close requests are to exhausting SLA budgets."""

    def __init__(
        self,
        sla_ttft_ms: float | None = None,
        sla_tpot_ms: float | None = None,
        sla_total_ms: float | None = None,
        near_miss_threshold: float = 0.8,
    ) -> None:
        if sla_ttft_ms is None and sla_tpot_ms is None and sla_total_ms is None:
            msg = "At least one SLA threshold must be provided"
            raise ValueError(msg)
        if not 0 < near_miss_threshold < 1:
            msg = "near_miss_threshold must be between 0 and 1 (exclusive)"
            raise ValueError(msg)
        self.sla_ttft_ms = sla_ttft_ms
        self.sla_tpot_ms = sla_tpot_ms
        self.sla_total_ms = sla_total_ms
        self.near_miss_threshold = near_miss_threshold

    def analyze(self, data: BenchmarkData, top_n: int = 10) -> BudgetReport:
        """Analyze budget consumption for all requests."""
        budgets: list[RequestBudget] = []
        ttft_ratios: list[float] = []
        tpot_ratios: list[float] = []
        total_ratios: list[float] = []

        for req in data.requests:
            ttft_r = req.ttft_ms / self.sla_ttft_ms if self.sla_ttft_ms else None
            tpot_r = req.tpot_ms / self.sla_tpot_ms if self.sla_tpot_ms else None
            total_r = req.total_latency_ms / self.sla_total_ms if self.sla_total_ms else None

            if ttft_r is not None:
                ttft_ratios.append(ttft_r)
            if tpot_r is not None:
                tpot_ratios.append(tpot_r)
            if total_r is not None:
                total_ratios.append(total_r)

            # Find worst metric
            candidates: list[tuple[str, float]] = []
            if ttft_r is not None:
                candidates.append(("ttft", ttft_r))
            if tpot_r is not None:
                candidates.append(("tpot", tpot_r))
            if total_r is not None:
                candidates.append(("total_latency", total_r))

            worst_name, worst_val = max(candidates, key=lambda x: x[1])
            status = _classify(worst_val, self.near_miss_threshold)

            budgets.append(
                RequestBudget(
                    request_id=req.request_id,
                    ttft_ratio=ttft_r,
                    tpot_ratio=tpot_r,
                    total_ratio=total_r,
                    worst_metric=worst_name,
                    worst_ratio=round(worst_val, 4),
                    status=status,
                )
            )

        # Distribution stats
        distributions: list[BudgetDistribution] = []
        for name, ratios in [
            ("ttft", ttft_ratios),
            ("tpot", tpot_ratios),
            ("total_latency", total_ratios),
        ]:
            if not ratios:
                continue
            distributions.append(
                BudgetDistribution(
                    metric=name,
                    mean=round(sum(ratios) / len(ratios), 4),
                    p50=round(_percentile(ratios, 50), 4),
                    p95=round(_percentile(ratios, 95), 4),
                    p99=round(_percentile(ratios, 99), 4),
                    max=round(max(ratios), 4),
                    near_miss_count=sum(
                        1 for r in ratios if self.near_miss_threshold < r <= 1.0
                    ),
                    exceeded_count=sum(1 for r in ratios if r > 1.0),
                )
            )

        # Counts
        near_miss = sum(1 for b in budgets if b.status == BudgetStatus.NEAR_MISS)
        exceeded = sum(1 for b in budgets if b.status == BudgetStatus.EXCEEDED)
        comfortable = sum(1 for b in budgets if b.status == BudgetStatus.COMFORTABLE)

        # Top worst requests
        worst = sorted(budgets, key=lambda b: b.worst_ratio, reverse=True)[:top_n]

        # Alerts
        alerts: list[BudgetAlert] = []
        total = len(budgets)
        if total > 0:
            near_miss_pct = near_miss / total * 100
            exceeded_pct = exceeded / total * 100
            if exceeded_pct > 5:
                alerts.append(
                    BudgetAlert(
                        message=f"{exceeded_pct:.1f}% of requests exceed SLA",
                        severity="CRITICAL",
                    )
                )
            if near_miss_pct > 20:
                alerts.append(
                    BudgetAlert(
                        message=f"{near_miss_pct:.1f}% of requests are near-miss",
                        severity="WARNING",
                    )
                )
            # Check per-metric
            for dist in distributions:
                if dist.p95 > 0.9:
                    alerts.append(
                        BudgetAlert(
                            message=f"{dist.metric} P95 budget consumption at {dist.p95:.0%}",
                            severity="WARNING",
                            metric=dist.metric,
                        )
                    )

        return BudgetReport(
            total_requests=total,
            near_miss_threshold=self.near_miss_threshold,
            near_miss_count=near_miss,
            exceeded_count=exceeded,
            comfortable_count=comfortable,
            distributions=distributions,
            worst_requests=worst,
            alerts=alerts,
        )


def track_latency_budget(
    benchmark_path: str,
    sla_ttft_ms: float | None = None,
    sla_tpot_ms: float | None = None,
    sla_total_ms: float | None = None,
    near_miss_threshold: float = 0.8,
    top_n: int = 10,
) -> dict:
    """Programmatic API for latency budget tracking.

    Args:
        benchmark_path: Path to benchmark JSON file.
        sla_ttft_ms: TTFT SLA threshold in milliseconds.
        sla_tpot_ms: TPOT SLA threshold in milliseconds.
        sla_total_ms: Total latency SLA threshold in milliseconds.
        near_miss_threshold: Budget consumption ratio above which requests are near-miss (0-1).
        top_n: Number of worst requests to include.

    Returns:
        Dictionary with budget tracking results.
    """
    from xpyd_plan.bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    tracker = LatencyBudgetTracker(
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
        sla_total_ms=sla_total_ms,
        near_miss_threshold=near_miss_threshold,
    )
    report = tracker.analyze(data, top_n=top_n)
    return report.model_dump()

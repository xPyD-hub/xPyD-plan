"""Benchmark data analyzer — find optimal P:D ratio from measured data.

Core principle: no modeling, no simulation. Everything is based on actual
benchmark measurements. We analyze how measured latencies and utilization
would change under different P:D ratios by scaling observed per-instance loads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xpyd_plan.benchmark_models import (
    AnalysisResult,
    BenchmarkData,
    RatioCandidate,
    SLACheck,
    UtilizationResult,
)
from xpyd_plan.models import SLAConfig


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list of values."""
    if not values:
        return 0.0
    arr = np.array(values)
    return float(np.percentile(arr, p))


class BenchmarkAnalyzer:
    """Analyze benchmark data to find optimal P:D instance ratio.

    The analyzer works with measured data from xpyd-bench runs. It computes
    latency percentiles, instance utilization, and finds the P:D ratio that
    minimizes resource waste while meeting SLA constraints.
    """

    def __init__(self) -> None:
        self._data: BenchmarkData | None = None

    @property
    def data(self) -> BenchmarkData:
        if self._data is None:
            raise RuntimeError("No benchmark data loaded. Call load_data() first.")
        return self._data

    def load_data(self, path: str | Path) -> BenchmarkData:
        """Load benchmark data from a JSON file.

        Args:
            path: Path to the benchmark JSON file.

        Returns:
            Parsed BenchmarkData.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the data is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        raw = json.loads(path.read_text())
        self._data = BenchmarkData(**raw)
        return self._data

    def load_data_from_dict(self, data: dict) -> BenchmarkData:
        """Load benchmark data from a dictionary.

        Args:
            data: Dictionary matching BenchmarkData schema.

        Returns:
            Parsed BenchmarkData.
        """
        self._data = BenchmarkData(**data)
        return self._data

    def check_sla(self, sla: SLAConfig) -> SLACheck:
        """Check SLA compliance based on measured latency distributions.

        Uses P95 for SLA pass/fail determination and also reports P99.

        Args:
            sla: SLA configuration with thresholds.

        Returns:
            SLACheck with percentile values and compliance flags.
        """
        reqs = self.data.requests
        ttfts = [r.ttft_ms for r in reqs]
        tpots = [r.tpot_ms for r in reqs]
        total_lats = [r.total_latency_ms for r in reqs]

        ttft_p95 = _percentile(ttfts, 95)
        ttft_p99 = _percentile(ttfts, 99)
        tpot_p95 = _percentile(tpots, 95)
        tpot_p99 = _percentile(tpots, 99)
        total_p95 = _percentile(total_lats, 95)
        total_p99 = _percentile(total_lats, 99)

        meets_ttft = sla.ttft_ms is None or ttft_p95 <= sla.ttft_ms
        meets_tpot = sla.tpot_ms is None or tpot_p95 <= sla.tpot_ms
        meets_total = sla.max_latency_ms is None or total_p95 <= sla.max_latency_ms

        return SLACheck(
            ttft_p95_ms=ttft_p95,
            ttft_p99_ms=ttft_p99,
            tpot_p95_ms=tpot_p95,
            tpot_p99_ms=tpot_p99,
            total_latency_p95_ms=total_p95,
            total_latency_p99_ms=total_p99,
            meets_ttft=meets_ttft,
            meets_tpot=meets_tpot,
            meets_total_latency=meets_total,
            meets_all=meets_ttft and meets_tpot and meets_total,
        )

    def compute_utilization(self) -> UtilizationResult:
        """Compute P and D instance utilization from measured data.

        Utilization is estimated from the measured data:
        - Prefill utilization: total prefill processing time / (num_P * benchmark_duration)
        - Decode utilization: total decode processing time / (num_D * benchmark_duration)

        Returns:
            UtilizationResult with P/D utilization and waste rate.
        """
        reqs = self.data.requests
        meta = self.data.metadata

        # Estimate benchmark duration from timestamps
        timestamps = [r.timestamp for r in reqs]
        duration_s = max(timestamps) - min(timestamps)
        if duration_s <= 0:
            # If all same timestamp, estimate from total latency
            duration_s = max(r.total_latency_ms for r in reqs) / 1000.0

        # Total prefill work (ms): sum of TTFT across all requests
        # TTFT represents the time a prefill instance spent on this request
        total_prefill_ms = sum(r.ttft_ms for r in reqs)
        # Total decode work (ms): sum of (tpot * output_tokens) across all requests
        total_decode_ms = sum(r.tpot_ms * r.output_tokens for r in reqs)

        # Available capacity: num_instances * duration in ms
        prefill_capacity_ms = meta.num_prefill_instances * duration_s * 1000.0
        decode_capacity_ms = meta.num_decode_instances * duration_s * 1000.0

        p_util = min(total_prefill_ms / prefill_capacity_ms, 1.0) if prefill_capacity_ms > 0 else 0
        d_util = min(total_decode_ms / decode_capacity_ms, 1.0) if decode_capacity_ms > 0 else 0

        waste = 1.0 - min(p_util, d_util)

        return UtilizationResult(
            prefill_utilization=round(p_util, 4),
            decode_utilization=round(d_util, 4),
            waste_rate=round(waste, 4),
        )

    def _estimate_ratio_performance(
        self,
        num_prefill: int,
        num_decode: int,
        sla: SLAConfig,
    ) -> RatioCandidate:
        """Estimate performance for a hypothetical P:D ratio based on measured data.

        Scaling logic:
        - If we change from P_orig to P_new prefill instances, the per-instance
          prefill load scales by (P_orig / P_new). More instances = less load each.
        - TTFT scales proportionally to per-instance load (more load = longer queue).
        - Same logic for decode side with TPOT.
        - Total latency = scaled_ttft + scaled_tpot * output_tokens (approximately).
        """
        meta = self.data.metadata
        reqs = self.data.requests

        p_scale = meta.num_prefill_instances / num_prefill
        d_scale = meta.num_decode_instances / num_decode

        # Scale latencies
        scaled_ttfts = [r.ttft_ms * p_scale for r in reqs]
        scaled_tpots = [r.tpot_ms * d_scale for r in reqs]
        scaled_totals = [
            r.ttft_ms * p_scale + r.tpot_ms * d_scale * r.output_tokens
            for r in reqs
        ]

        ttft_p95 = _percentile(scaled_ttfts, 95)
        ttft_p99 = _percentile(scaled_ttfts, 99)
        tpot_p95 = _percentile(scaled_tpots, 95)
        tpot_p99 = _percentile(scaled_tpots, 99)
        total_p95 = _percentile(scaled_totals, 95)
        total_p99 = _percentile(scaled_totals, 99)

        meets_ttft = sla.ttft_ms is None or ttft_p95 <= sla.ttft_ms
        meets_tpot = sla.tpot_ms is None or tpot_p95 <= sla.tpot_ms
        meets_total = sla.max_latency_ms is None or total_p95 <= sla.max_latency_ms
        meets_all = meets_ttft and meets_tpot and meets_total

        sla_check = SLACheck(
            ttft_p95_ms=ttft_p95,
            ttft_p99_ms=ttft_p99,
            tpot_p95_ms=tpot_p95,
            tpot_p99_ms=tpot_p99,
            total_latency_p95_ms=total_p95,
            total_latency_p99_ms=total_p99,
            meets_ttft=meets_ttft,
            meets_tpot=meets_tpot,
            meets_total_latency=meets_total,
            meets_all=meets_all,
        )

        # Compute utilization at this ratio
        timestamps = [r.timestamp for r in reqs]
        duration_s = max(timestamps) - min(timestamps)
        if duration_s <= 0:
            duration_s = max(r.total_latency_ms for r in reqs) / 1000.0

        total_prefill_ms = sum(r.ttft_ms for r in reqs)
        total_decode_ms = sum(r.tpot_ms * r.output_tokens for r in reqs)

        # Scale work stays the same, but capacity changes with instance count
        prefill_capacity_ms = num_prefill * duration_s * 1000.0
        decode_capacity_ms = num_decode * duration_s * 1000.0

        p_util = min(total_prefill_ms / prefill_capacity_ms, 1.0) if prefill_capacity_ms > 0 else 0
        d_util = min(total_decode_ms / decode_capacity_ms, 1.0) if decode_capacity_ms > 0 else 0
        waste = 1.0 - min(p_util, d_util)

        return RatioCandidate(
            num_prefill=num_prefill,
            num_decode=num_decode,
            prefill_utilization=round(p_util, 4),
            decode_utilization=round(d_util, 4),
            waste_rate=round(waste, 4),
            meets_sla=meets_all,
            sla_check=sla_check,
        )

    def find_optimal_ratio(
        self,
        total_instances: int,
        sla: SLAConfig,
    ) -> AnalysisResult:
        """Find the P:D ratio that minimizes waste while meeting SLA.

        Enumerates all possible P:D splits for the given total instance count,
        scales the measured data to estimate performance at each ratio, and
        returns the one with minimum waste that still meets SLA.

        Args:
            total_instances: Total number of instances to allocate.
            sla: SLA constraints.

        Returns:
            AnalysisResult with best ratio and all candidates.
        """
        if total_instances < 2:
            return AnalysisResult(
                best=None, candidates=[], total_instances=total_instances
            )

        candidates: list[RatioCandidate] = []
        for num_p in range(1, total_instances):
            num_d = total_instances - num_p
            candidate = self._estimate_ratio_performance(num_p, num_d, sla)
            candidates.append(candidate)

        # Sort: SLA-meeting first, then by waste_rate ascending
        candidates.sort(key=lambda c: (-int(c.meets_sla), c.waste_rate))

        best = candidates[0] if candidates and candidates[0].meets_sla else None

        # Current config analysis
        current_util = self.compute_utilization()
        current_sla = self.check_sla(sla)

        return AnalysisResult(
            best=best,
            candidates=candidates,
            total_instances=total_instances,
            current_config=current_util,
            current_sla_check=current_sla,
        )

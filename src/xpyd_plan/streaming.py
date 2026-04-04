"""Streaming benchmark analyzer — process records incrementally.

Useful for live analysis during ongoing benchmark runs. Reads JSONL
(one request record per line) from a stream and emits periodic analysis
snapshots.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, TextIO

import numpy as np

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.models import SLAConfig


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


@dataclass
class StreamingSnapshot:
    """A point-in-time analysis snapshot from streaming data."""

    request_count: int
    elapsed_seconds: float
    measured_qps: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    total_latency_p95_ms: float
    total_latency_p99_ms: float
    meets_sla: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "measured_qps": round(self.measured_qps, 2),
            "ttft_p95_ms": round(self.ttft_p95_ms, 1),
            "ttft_p99_ms": round(self.ttft_p99_ms, 1),
            "tpot_p95_ms": round(self.tpot_p95_ms, 1),
            "tpot_p99_ms": round(self.tpot_p99_ms, 1),
            "total_latency_p95_ms": round(self.total_latency_p95_ms, 1),
            "total_latency_p99_ms": round(self.total_latency_p99_ms, 1),
            "meets_sla": self.meets_sla,
        }


class StreamingAnalyzer:
    """Incrementally analyze benchmark records as they arrive.

    Collects request records and periodically computes analysis snapshots.
    Can read from stdin, a file, or accept records programmatically.
    """

    def __init__(
        self,
        sla: SLAConfig,
        snapshot_interval: int = 10,
    ) -> None:
        """Initialize streaming analyzer.

        Args:
            sla: SLA constraints.
            snapshot_interval: Number of new requests between snapshots.
        """
        self.sla = sla
        self.snapshot_interval = snapshot_interval
        self._requests: list[BenchmarkRequest] = []
        self._snapshots: list[StreamingSnapshot] = []
        self._start_time: float | None = None
        self._last_snapshot_count: int = 0

    @property
    def request_count(self) -> int:
        return len(self._requests)

    @property
    def snapshots(self) -> list[StreamingSnapshot]:
        return list(self._snapshots)

    def add_request(self, record: dict[str, Any]) -> StreamingSnapshot | None:
        """Add a single request record and optionally produce a snapshot.

        Accepts both native and xpyd-bench field names.

        Args:
            record: Request record dict.

        Returns:
            StreamingSnapshot if interval reached, else None.
        """
        req = self._parse_request(record)
        self._requests.append(req)

        if self._start_time is None:
            self._start_time = req.timestamp

        if self.request_count - self._last_snapshot_count >= self.snapshot_interval:
            return self._take_snapshot()
        return None

    def finalize(self) -> StreamingSnapshot:
        """Produce a final snapshot with all collected data.

        Returns:
            Final StreamingSnapshot.
        """
        return self._take_snapshot()

    def _parse_request(self, record: dict[str, Any]) -> BenchmarkRequest:
        """Parse a request record, handling both native and xpyd-bench field names."""
        return BenchmarkRequest(
            request_id=record.get("request_id") or record.get("id", f"req-{len(self._requests)}"),
            prompt_tokens=record.get("prompt_tokens") or record.get("input_tokens", 0),
            output_tokens=record.get("output_tokens", 0),
            ttft_ms=record.get("ttft_ms") or record.get("time_to_first_token_ms", 0),
            tpot_ms=record.get("tpot_ms") or record.get("time_per_output_token_ms", 0),
            total_latency_ms=(
                record.get("total_latency_ms") or record.get("end_to_end_latency_ms", 0)
            ),
            timestamp=record.get("timestamp") or record.get("start_time", time.time()),
        )

    def _take_snapshot(self) -> StreamingSnapshot:
        """Compute and store a snapshot from current data."""
        ttfts = [r.ttft_ms for r in self._requests]
        tpots = [r.tpot_ms for r in self._requests]
        total_lats = [r.total_latency_ms for r in self._requests]

        ttft_p95 = _percentile(ttfts, 95)
        ttft_p99 = _percentile(ttfts, 99)
        tpot_p95 = _percentile(tpots, 95)
        tpot_p99 = _percentile(tpots, 99)
        total_p95 = _percentile(total_lats, 95)
        total_p99 = _percentile(total_lats, 99)

        pctl = self.sla.sla_percentile
        ttft_eval = _percentile(ttfts, pctl)
        tpot_eval = _percentile(tpots, pctl)
        total_eval = _percentile(total_lats, pctl)

        meets_ttft = self.sla.ttft_ms is None or ttft_eval <= self.sla.ttft_ms
        meets_tpot = self.sla.tpot_ms is None or tpot_eval <= self.sla.tpot_ms
        meets_total = self.sla.max_latency_ms is None or total_eval <= self.sla.max_latency_ms
        meets_all = meets_ttft and meets_tpot and meets_total

        timestamps = [r.timestamp for r in self._requests]
        elapsed = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
        qps = len(self._requests) / elapsed if elapsed > 0 else 0.0

        snapshot = StreamingSnapshot(
            request_count=len(self._requests),
            elapsed_seconds=elapsed,
            measured_qps=qps,
            ttft_p95_ms=ttft_p95,
            ttft_p99_ms=ttft_p99,
            tpot_p95_ms=tpot_p95,
            tpot_p99_ms=tpot_p99,
            total_latency_p95_ms=total_p95,
            total_latency_p99_ms=total_p99,
            meets_sla=meets_all,
        )

        self._snapshots.append(snapshot)
        self._last_snapshot_count = self.request_count
        return snapshot

    def to_benchmark_data(self, metadata: BenchmarkMetadata) -> BenchmarkData:
        """Convert collected streaming data to BenchmarkData for full analysis.

        Args:
            metadata: Cluster configuration metadata.

        Returns:
            BenchmarkData containing all collected requests.
        """
        return BenchmarkData(metadata=metadata, requests=list(self._requests))


def stream_from_stdin(
    sla: SLAConfig,
    snapshot_interval: int = 10,
    output: TextIO = sys.stdout,
) -> StreamingAnalyzer:
    """Read JSONL records from stdin and run streaming analysis.

    Each line should be a JSON object representing a single request record.
    Prints snapshot summaries to output as they are produced.

    Args:
        sla: SLA constraints.
        snapshot_interval: Requests between snapshots.
        output: Output stream for snapshots (default: stdout).

    Returns:
        StreamingAnalyzer with all collected data.
    """
    analyzer = StreamingAnalyzer(sla=sla, snapshot_interval=snapshot_interval)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        snapshot = analyzer.add_request(record)
        if snapshot:
            _print_snapshot(snapshot, output)

    if analyzer.request_count > 0:
        final = analyzer.finalize()
        output.write("--- Final Snapshot ---\n")
        _print_snapshot(final, output)

    return analyzer


def _print_snapshot(snapshot: StreamingSnapshot, output: TextIO) -> None:
    """Print a snapshot summary line."""
    sla_str = "✅" if snapshot.meets_sla else "❌"
    output.write(
        f"[{snapshot.request_count:>5} reqs | "
        f"QPS {snapshot.measured_qps:>6.1f} | "
        f"TTFT P95 {snapshot.ttft_p95_ms:>7.1f}ms | "
        f"TPOT P95 {snapshot.tpot_p95_ms:>7.1f}ms | "
        f"SLA {sla_str}]\n"
    )
    output.flush()

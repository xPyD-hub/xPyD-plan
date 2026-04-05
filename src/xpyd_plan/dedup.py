"""Request deduplication analysis — detect duplicate requests in benchmark data."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData, BenchmarkRequest


class DuplicateGroup(BaseModel):
    """A group of duplicate or near-duplicate requests."""

    group_id: int = Field(..., description="Unique group identifier")
    count: int = Field(..., ge=2, description="Number of requests in group")
    representative_prompt_tokens: int = Field(..., description="Prompt tokens of the group")
    representative_output_tokens: int = Field(..., description="Output tokens of the group")
    request_indices: list[int] = Field(..., description="Indices of duplicate requests")
    is_exact: bool = Field(..., description="True if all duplicates are exact matches")


class DeduplicationReport(BaseModel):
    """Complete deduplication analysis report."""

    total_requests: int = Field(..., description="Total requests in benchmark")
    unique_requests: int = Field(..., description="Requests after deduplication")
    duplicate_count: int = Field(..., ge=0, description="Number of duplicate requests removed")
    duplication_rate: float = Field(
        ..., ge=0, le=1, description="Fraction of requests that are duplicates"
    )
    group_count: int = Field(..., ge=0, description="Number of duplicate groups found")
    largest_group_size: int = Field(..., ge=0, description="Size of largest duplicate group")
    groups: list[DuplicateGroup] = Field(..., description="All duplicate groups")
    recommendation: str = Field(..., description="Human-readable recommendation")


class DeduplicationConfig(BaseModel):
    """Configuration for deduplication analysis."""

    tolerance_ms: float = Field(
        default=0.0,
        ge=0,
        description="Latency tolerance in ms for near-duplicate detection",
    )


class DeduplicationAnalyzer:
    """Detect and report duplicate requests in benchmark data."""

    def analyze(
        self,
        data: BenchmarkData,
        *,
        tolerance_ms: float = 0.0,
    ) -> DeduplicationReport:
        """Analyze benchmark data for duplicate requests.

        Args:
            data: Benchmark data to analyze.
            tolerance_ms: Latency tolerance for near-duplicate detection.
                0.0 means exact match only.

        Returns:
            DeduplicationReport with duplicate groups and statistics.
        """
        requests = data.requests
        total = len(requests)

        if total == 0:
            return DeduplicationReport(
                total_requests=0,
                unique_requests=0,
                duplicate_count=0,
                duplication_rate=0.0,
                group_count=0,
                largest_group_size=0,
                groups=[],
                recommendation="No requests to analyze.",
            )

        groups = self._find_groups(requests, tolerance_ms)

        duplicate_count = sum(g.count - 1 for g in groups)
        unique_requests = total - duplicate_count
        duplication_rate = duplicate_count / total if total > 0 else 0.0
        largest = max((g.count for g in groups), default=0)

        if duplication_rate > 0.1:
            recommendation = (
                f"High duplication rate ({duplication_rate:.1%}). "
                "Consider deduplicating before analysis to avoid biased results."
            )
        elif duplication_rate > 0.01:
            recommendation = (
                f"Moderate duplication rate ({duplication_rate:.1%}). "
                "Duplicates are unlikely to significantly affect results."
            )
        else:
            recommendation = "Minimal or no duplication detected. Data is clean."

        return DeduplicationReport(
            total_requests=total,
            unique_requests=unique_requests,
            duplicate_count=duplicate_count,
            duplication_rate=duplication_rate,
            group_count=len(groups),
            largest_group_size=largest,
            groups=groups,
            recommendation=recommendation,
        )

    def deduplicate(
        self,
        data: BenchmarkData,
        *,
        tolerance_ms: float = 0.0,
    ) -> BenchmarkData:
        """Return a new BenchmarkData with duplicates removed (keep first occurrence).

        Args:
            data: Benchmark data to deduplicate.
            tolerance_ms: Latency tolerance for near-duplicate detection.

        Returns:
            New BenchmarkData with duplicates removed.
        """
        requests = data.requests
        if not requests:
            return data.model_copy()

        report = self.analyze(data, tolerance_ms=tolerance_ms)

        # Collect indices to remove (all but first in each group)
        remove_indices: set[int] = set()
        for group in report.groups:
            for idx in group.request_indices[1:]:
                remove_indices.add(idx)

        kept = [r for i, r in enumerate(requests) if i not in remove_indices]

        # Adjust QPS proportionally
        ratio = len(kept) / len(requests) if requests else 1.0
        new_qps = data.metadata.measured_qps * ratio

        new_metadata = data.metadata.model_copy(
            update={"measured_qps": round(new_qps, 2)}
        )

        return data.model_copy(
            update={
                "requests": kept,
                "metadata": new_metadata,
            }
        )

    def _find_groups(
        self,
        requests: list[BenchmarkRequest],
        tolerance_ms: float,
    ) -> list[DuplicateGroup]:
        """Find groups of duplicate requests."""
        if tolerance_ms <= 0:
            return self._find_exact_groups(requests)
        return self._find_near_groups(requests, tolerance_ms)

    def _find_exact_groups(
        self,
        requests: list[BenchmarkRequest],
    ) -> list[DuplicateGroup]:
        """Find exact duplicate groups."""
        buckets: dict[tuple[int, int, float, float], list[int]] = defaultdict(list)
        for i, r in enumerate(requests):
            key = (r.prompt_tokens, r.output_tokens, r.ttft_ms, r.tpot_ms)
            buckets[key].append(i)

        groups: list[DuplicateGroup] = []
        gid = 0
        for key, indices in buckets.items():
            if len(indices) >= 2:
                groups.append(
                    DuplicateGroup(
                        group_id=gid,
                        count=len(indices),
                        representative_prompt_tokens=key[0],
                        representative_output_tokens=key[1],
                        request_indices=indices,
                        is_exact=True,
                    )
                )
                gid += 1

        return groups

    def _find_near_groups(
        self,
        requests: list[BenchmarkRequest],
        tolerance_ms: float,
    ) -> list[DuplicateGroup]:
        """Find near-duplicate groups (same tokens, latency within tolerance)."""
        # Group by token counts first
        token_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, r in enumerate(requests):
            key = (r.prompt_tokens, r.output_tokens)
            token_buckets[key].append(i)

        groups: list[DuplicateGroup] = []
        gid = 0

        for (pt, ot), indices in token_buckets.items():
            if len(indices) < 2:
                continue

            # Within same token bucket, cluster by latency proximity
            assigned: set[int] = set()
            for i, idx_a in enumerate(indices):
                if idx_a in assigned:
                    continue
                cluster = [idx_a]
                ra = requests[idx_a]
                for idx_b in indices[i + 1 :]:
                    if idx_b in assigned:
                        continue
                    rb = requests[idx_b]
                    if (
                        abs(ra.ttft_ms - rb.ttft_ms) <= tolerance_ms
                        and abs(ra.tpot_ms - rb.tpot_ms) <= tolerance_ms
                    ):
                        cluster.append(idx_b)
                        assigned.add(idx_b)

                if len(cluster) >= 2:
                    # Check if all are exact
                    first = requests[cluster[0]]
                    is_exact = all(
                        requests[c].ttft_ms == first.ttft_ms
                        and requests[c].tpot_ms == first.tpot_ms
                        for c in cluster
                    )
                    groups.append(
                        DuplicateGroup(
                            group_id=gid,
                            count=len(cluster),
                            representative_prompt_tokens=pt,
                            representative_output_tokens=ot,
                            request_indices=cluster,
                            is_exact=is_exact,
                        )
                    )
                    gid += 1
                    assigned.update(cluster)

        return groups


def analyze_dedup(
    benchmark_path: str,
    *,
    tolerance_ms: float = 0.0,
) -> dict:
    """Programmatic API for deduplication analysis.

    Args:
        benchmark_path: Path to benchmark JSON file.
        tolerance_ms: Latency tolerance for near-duplicate detection.

    Returns:
        Dict with deduplication analysis results.
    """
    from .bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(benchmark_path)
    analyzer = DeduplicationAnalyzer()
    report = analyzer.analyze(data, tolerance_ms=tolerance_ms)
    return report.model_dump()

"""Benchmark merge and aggregation — combine multiple benchmark files."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest


class MergeStrategy(str, Enum):
    """Strategy for handling overlapping request IDs."""

    UNION = "union"
    INTERSECTION = "intersection"


class MergeConfig(BaseModel):
    """Configuration for benchmark merging."""

    strategy: MergeStrategy = Field(
        MergeStrategy.UNION,
        description="How to handle overlapping request IDs",
    )
    require_same_config: bool = Field(
        True,
        description="Require all benchmarks to have identical cluster configuration",
    )


class MergeResult(BaseModel):
    """Result of benchmark merge operation."""

    merged: BenchmarkData
    source_count: int = Field(..., ge=2, description="Number of source benchmark files merged")
    total_requests_before: int = Field(..., description="Total requests across all sources")
    total_requests_after: int = Field(..., description="Requests in merged dataset")
    duplicates_removed: int = Field(0, description="Duplicate request IDs resolved")
    strategy_used: MergeStrategy


class BenchmarkMerger:
    """Merge multiple benchmark datasets into one aggregated dataset."""

    def __init__(self, config: MergeConfig | None = None) -> None:
        self.config = config or MergeConfig()

    def merge(self, datasets: list[BenchmarkData]) -> MergeResult:
        """Merge multiple benchmark datasets.

        Args:
            datasets: List of BenchmarkData to merge (minimum 2).

        Returns:
            MergeResult with merged data and merge statistics.

        Raises:
            ValueError: If fewer than 2 datasets, or incompatible configurations.
        """
        if len(datasets) < 2:
            raise ValueError("At least 2 benchmark datasets are required for merging")

        if self.config.require_same_config:
            self._validate_compatible(datasets)

        total_before = sum(len(d.requests) for d in datasets)

        if self.config.strategy == MergeStrategy.UNION:
            merged_requests = self._merge_union(datasets)
        else:
            merged_requests = self._merge_intersection(datasets)

        if not merged_requests:
            raise ValueError(
                "Merge produced an empty dataset — no common requests found "
                "(intersection strategy with disjoint datasets)"
            )

        # Aggregate metadata
        metadata = self._aggregate_metadata(datasets, merged_requests)

        merged_data = BenchmarkData(metadata=metadata, requests=merged_requests)

        return MergeResult(
            merged=merged_data,
            source_count=len(datasets),
            total_requests_before=total_before,
            total_requests_after=len(merged_requests),
            duplicates_removed=total_before - len(merged_requests),
            strategy_used=self.config.strategy,
        )

    def _validate_compatible(self, datasets: list[BenchmarkData]) -> None:
        """Ensure all datasets have the same cluster configuration."""
        ref = datasets[0].metadata
        for i, ds in enumerate(datasets[1:], start=2):
            m = ds.metadata
            if (
                m.num_prefill_instances != ref.num_prefill_instances
                or m.num_decode_instances != ref.num_decode_instances
                or m.total_instances != ref.total_instances
            ):
                raise ValueError(
                    f"Incompatible cluster config: dataset 1 has "
                    f"{ref.num_prefill_instances}P:{ref.num_decode_instances}D, "
                    f"dataset {i} has {m.num_prefill_instances}P:{m.num_decode_instances}D"
                )

    def _merge_union(self, datasets: list[BenchmarkData]) -> list[BenchmarkRequest]:
        """Union merge: keep all unique requests; first occurrence wins for duplicates."""
        seen: dict[str, BenchmarkRequest] = {}
        for ds in datasets:
            for req in ds.requests:
                if req.request_id not in seen:
                    seen[req.request_id] = req
        return sorted(seen.values(), key=lambda r: r.timestamp)

    def _merge_intersection(self, datasets: list[BenchmarkData]) -> list[BenchmarkRequest]:
        """Intersection merge: keep only requests present in ALL datasets."""
        # Find common request IDs
        id_sets = [set(r.request_id for r in ds.requests) for ds in datasets]
        common_ids = id_sets[0]
        for s in id_sets[1:]:
            common_ids &= s

        # Take from first dataset for consistent values
        first_map = {r.request_id: r for r in datasets[0].requests}
        return sorted(
            [first_map[rid] for rid in common_ids if rid in first_map],
            key=lambda r: r.timestamp,
        )

    def _aggregate_metadata(
        self,
        datasets: list[BenchmarkData],
        merged_requests: list[BenchmarkRequest],
    ) -> BenchmarkMetadata:
        """Compute aggregated metadata from merged datasets."""
        ref = datasets[0].metadata

        # Combined QPS: sum of individual QPS values (more data from same config)
        combined_qps = sum(d.metadata.measured_qps for d in datasets)

        return BenchmarkMetadata(
            num_prefill_instances=ref.num_prefill_instances,
            num_decode_instances=ref.num_decode_instances,
            total_instances=ref.total_instances,
            measured_qps=combined_qps,
        )


def merge_benchmarks(
    datasets: list[BenchmarkData],
    strategy: str = "union",
    require_same_config: bool = True,
) -> dict:
    """Programmatic API for merging benchmarks.

    Args:
        datasets: List of BenchmarkData to merge.
        strategy: "union" or "intersection".
        require_same_config: Whether to require identical cluster configs.

    Returns:
        Dict with merge results.
    """
    config = MergeConfig(
        strategy=MergeStrategy(strategy),
        require_same_config=require_same_config,
    )
    merger = BenchmarkMerger(config)
    result = merger.merge(datasets)
    return result.model_dump()

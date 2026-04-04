"""Multi-SLA tier analysis — find optimal P:D ratios for multiple SLA policies.

When serving heterogeneous traffic classes (e.g., premium vs standard),
different SLA policies apply.  This module analyzes a single benchmark
dataset against multiple named SLA tiers and produces a unified report
showing per-tier optimal ratios and whether a single P:D ratio can
satisfy all tiers simultaneously.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .analyzer import BenchmarkAnalyzer
from .benchmark_models import BenchmarkData, RatioCandidate
from .models import SLAConfig


class SLATier(BaseModel):
    """A named SLA policy tier."""

    name: str = Field(..., description="Tier name, e.g. 'premium', 'standard'")
    sla: SLAConfig = Field(..., description="SLA constraints for this tier")


class TierResult(BaseModel):
    """Analysis result for a single SLA tier."""

    tier: SLATier
    best: RatioCandidate | None = Field(
        None, description="Optimal P:D ratio for this tier (None if no ratio meets SLA)"
    )
    candidates: list[RatioCandidate] = Field(default_factory=list)


class MultiTierReport(BaseModel):
    """Report covering all SLA tiers."""

    tier_results: list[TierResult] = Field(default_factory=list)
    unified_best: RatioCandidate | None = Field(
        None,
        description="Single P:D ratio meeting ALL tiers with minimum waste (None if impossible)",
    )
    total_instances: int = Field(..., ge=2)


class SLATierAnalyzer:
    """Analyze benchmark data against multiple SLA tiers."""

    def __init__(self, data: BenchmarkData) -> None:
        self._data = data

    def analyze(self, tiers: list[SLATier]) -> MultiTierReport:
        """Run analysis for each tier and find unified optimum.

        Args:
            tiers: List of SLA tiers to evaluate.

        Returns:
            MultiTierReport with per-tier results and unified recommendation.

        Raises:
            ValueError: If tiers is empty.
        """
        if not tiers:
            raise ValueError("At least one SLA tier is required.")

        analyzer = BenchmarkAnalyzer()
        analyzer._data = self._data

        total = self._data.metadata.total_instances
        tier_results: list[TierResult] = []

        for tier in tiers:
            result = analyzer.find_optimal_ratio(
                total_instances=total,
                sla=tier.sla,
            )
            tier_results.append(
                TierResult(
                    tier=tier,
                    best=result.best,
                    candidates=result.candidates,
                )
            )

        # Find unified ratio: meets ALL tiers, minimum waste
        unified_best = self._find_unified(tier_results, total)

        return MultiTierReport(
            tier_results=tier_results,
            unified_best=unified_best,
            total_instances=total,
        )

    def _find_unified(
        self, tier_results: list[TierResult], total: int
    ) -> RatioCandidate | None:
        """Find the single P:D ratio that meets all tiers with minimum waste."""
        if not tier_results:
            return None

        # Collect candidate keys that pass each tier
        passing_sets: list[set[str]] = []
        # Map ratio_str -> RatioCandidate (from any tier, since metrics are same)
        candidate_map: dict[str, RatioCandidate] = {}

        for tr in tier_results:
            passing = set()
            for c in tr.candidates:
                if c.meets_sla:
                    passing.add(c.ratio_str)
                    candidate_map[c.ratio_str] = c
            passing_sets.append(passing)

        # Intersection: must pass ALL tiers
        if not passing_sets:
            return None
        common = passing_sets[0]
        for s in passing_sets[1:]:
            common = common & s

        if not common:
            return None

        # Pick lowest waste among common
        best: RatioCandidate | None = None
        for key in common:
            c = candidate_map[key]
            if best is None or c.waste_rate < best.waste_rate:
                best = c

        return best


def load_tiers_from_yaml(path: str | Path) -> list[SLATier]:
    """Load SLA tier definitions from a YAML file.

    Expected format::

        tiers:
          - name: premium
            ttft_ms: 100
            tpot_ms: 20
            max_latency_ms: 2000
            percentile: 99
          - name: standard
            ttft_ms: 500
            tpot_ms: 50
            percentile: 95
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "tiers" not in raw:
        raise ValueError(f"YAML file must contain a 'tiers' key: {path}")

    tiers: list[SLATier] = []
    for entry in raw["tiers"]:
        sla = SLAConfig(
            ttft_ms=entry.get("ttft_ms"),
            tpot_ms=entry.get("tpot_ms"),
            max_latency_ms=entry.get("max_latency_ms"),
            sla_percentile=entry.get("percentile", 95.0),
        )
        tiers.append(SLATier(name=entry["name"], sla=sla))

    return tiers


def analyze_sla_tiers(
    benchmark_path: str | Path,
    tiers: list[SLATier] | None = None,
    tiers_path: str | Path | None = None,
) -> dict[str, Any]:
    """Programmatic API for multi-SLA tier analysis.

    Args:
        benchmark_path: Path to benchmark JSON file.
        tiers: List of SLATier objects (mutually exclusive with tiers_path).
        tiers_path: Path to YAML tier definitions (mutually exclusive with tiers).

    Returns:
        Dict with 'tier_results', 'unified_best', 'total_instances'.

    Raises:
        ValueError: If neither tiers nor tiers_path is provided, or both are.
    """
    if tiers is None and tiers_path is None:
        raise ValueError("Provide either 'tiers' or 'tiers_path'.")
    if tiers is not None and tiers_path is not None:
        raise ValueError("Provide either 'tiers' or 'tiers_path', not both.")

    if tiers_path is not None:
        tiers = load_tiers_from_yaml(tiers_path)

    analyzer = BenchmarkAnalyzer()
    data = analyzer.load_data(benchmark_path)

    tier_analyzer = SLATierAnalyzer(data)
    report = tier_analyzer.analyze(tiers)  # type: ignore[arg-type]
    return report.model_dump()

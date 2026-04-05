"""Benchmark coverage score — assess P:D ratio space exploration completeness.

Given multiple benchmark files, evaluate how well they cover the possible
P:D ratio space, identify gaps, and recommend next benchmarks to run.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class CoverageGrade(str, Enum):
    """Letter grade for benchmark coverage quality."""

    EXCELLENT = "A"
    GOOD = "B"
    FAIR = "C"
    POOR = "D"
    MINIMAL = "F"


class RatioGap(BaseModel):
    """An unexplored P:D ratio in the configuration space."""

    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    distance_to_nearest: int = Field(
        ..., ge=1, description="Min distance to nearest benchmarked ratio"
    )
    priority: str = Field(..., description="Suggested benchmark priority")

    @property
    def ratio_str(self) -> str:
        return f"{self.num_prefill}P:{self.num_decode}D"


class CoverageMetrics(BaseModel):
    """Quantitative coverage metrics."""

    total_possible_ratios: int = Field(..., ge=1)
    benchmarked_ratios: int = Field(..., ge=0)
    coverage_fraction: float = Field(..., ge=0.0, le=1.0)
    has_boundaries: bool = Field(..., description="Benchmarked extreme P-heavy and D-heavy ratios")
    has_balanced: bool = Field(..., description="Benchmarked near-balanced ratio")
    max_gap_size: int = Field(..., ge=0, description="Largest gap between consecutive benchmarked")
    spread_score: float = Field(
        ..., ge=0.0, le=1.0, description="How evenly benchmarks are spread"
    )


class CoverageReport(BaseModel):
    """Full coverage analysis report."""

    total_instances: int = Field(..., ge=2)
    benchmarked: list[tuple[int, int]] = Field(..., description="(P, D) pairs benchmarked")
    metrics: CoverageMetrics
    score: float = Field(..., ge=0.0, le=100.0)
    grade: CoverageGrade
    gaps: list[RatioGap] = Field(..., description="Unexplored ratios, sorted by priority")
    recommendations: list[str] = Field(..., description="Actionable suggestions")


class CoverageAnalyzer:
    """Analyze benchmark coverage of the P:D ratio space."""

    def __init__(self, datasets: list[BenchmarkData]) -> None:
        if not datasets:
            raise ValueError("At least one benchmark dataset is required")
        self._datasets = datasets
        total_set = set()
        for ds in datasets:
            t = ds.metadata.num_prefill_instances + ds.metadata.num_decode_instances
            total_set.add(t)
        if len(total_set) != 1:
            raise ValueError(
                f"All benchmarks must have the same total instances, got {total_set}"
            )
        self._total = total_set.pop()

    def analyze(self) -> CoverageReport:
        """Run coverage analysis."""
        benchmarked = sorted(
            {
                (ds.metadata.num_prefill_instances, ds.metadata.num_decode_instances)
                for ds in self._datasets
            }
        )
        all_ratios = [(p, self._total - p) for p in range(1, self._total)]
        total_possible = len(all_ratios)
        n_benchmarked = len(benchmarked)

        # Boundary check
        prefill_vals = [p for p, _ in benchmarked]
        has_boundaries = (1 in prefill_vals) and (self._total - 1 in prefill_vals)

        # Balanced check
        mid = self._total / 2
        has_balanced = any(abs(p - mid) <= 1 for p, _ in benchmarked)

        # Gap analysis
        benchmarked_p = sorted(p for p, _ in benchmarked)
        max_gap = 0
        if benchmarked_p:
            # Gap from 1 to first benchmarked
            max_gap = max(max_gap, benchmarked_p[0] - 1)
            # Gap from last benchmarked to total-1
            max_gap = max(max_gap, (self._total - 1) - benchmarked_p[-1])
            for i in range(len(benchmarked_p) - 1):
                gap = benchmarked_p[i + 1] - benchmarked_p[i] - 1
                max_gap = max(max_gap, gap)

        # Spread score: ideal is uniform spacing. Use 1 - normalized_std_of_gaps
        spread_score = self._compute_spread_score(benchmarked_p)

        coverage_fraction = n_benchmarked / total_possible if total_possible > 0 else 0.0

        metrics = CoverageMetrics(
            total_possible_ratios=total_possible,
            benchmarked_ratios=n_benchmarked,
            coverage_fraction=round(coverage_fraction, 4),
            has_boundaries=has_boundaries,
            has_balanced=has_balanced,
            max_gap_size=max_gap,
            spread_score=round(spread_score, 4),
        )

        # Score: weighted composite
        score = self._compute_score(metrics)
        grade = self._grade(score)

        # Gaps
        benchmarked_set = set(benchmarked)
        gaps = self._find_gaps(all_ratios, benchmarked_set)

        # Recommendations
        recommendations = self._generate_recommendations(metrics, gaps)

        return CoverageReport(
            total_instances=self._total,
            benchmarked=benchmarked,
            metrics=metrics,
            score=round(score, 1),
            grade=grade,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _compute_spread_score(self, benchmarked_p: list[int]) -> float:
        """Compute how evenly benchmarks are spread across the ratio space."""
        if len(benchmarked_p) <= 1:
            return 0.0

        gaps = []
        gaps.append(benchmarked_p[0] - 1)
        for i in range(len(benchmarked_p) - 1):
            gaps.append(benchmarked_p[i + 1] - benchmarked_p[i] - 1)
        gaps.append((self._total - 1) - benchmarked_p[-1])

        if not gaps:
            return 1.0

        mean_gap = sum(gaps) / len(gaps)
        if mean_gap == 0:
            return 1.0

        variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        std = variance**0.5
        cv = std / mean_gap if mean_gap > 0 else 0.0
        # CV = 0 means perfectly even → score = 1.0; cap at CV=2
        return max(0.0, 1.0 - cv / 2.0)

    def _compute_score(self, m: CoverageMetrics) -> float:
        """Compute composite 0-100 coverage score."""
        # Coverage fraction: 40 points
        cov_pts = min(m.coverage_fraction * 100, 100) * 0.4

        # Boundary coverage: 15 points
        boundary_pts = 15.0 if m.has_boundaries else 0.0

        # Balanced coverage: 15 points
        balanced_pts = 15.0 if m.has_balanced else 0.0

        # Spread evenness: 20 points
        spread_pts = m.spread_score * 20.0

        # Gap penalty: up to 10 points off for large gaps
        total_possible = m.total_possible_ratios
        if total_possible > 0:
            gap_ratio = m.max_gap_size / total_possible
        else:
            gap_ratio = 0.0
        gap_pts = max(0.0, 10.0 * (1.0 - gap_ratio * 2))

        return min(100.0, cov_pts + boundary_pts + balanced_pts + spread_pts + gap_pts)

    def _grade(self, score: float) -> CoverageGrade:
        if score >= 90:
            return CoverageGrade.EXCELLENT
        if score >= 75:
            return CoverageGrade.GOOD
        if score >= 60:
            return CoverageGrade.FAIR
        if score >= 40:
            return CoverageGrade.POOR
        return CoverageGrade.MINIMAL

    def _find_gaps(
        self,
        all_ratios: list[tuple[int, int]],
        benchmarked_set: set[tuple[int, int]],
    ) -> list[RatioGap]:
        """Find unexplored ratios and compute distance to nearest benchmarked."""
        benchmarked_p = sorted(p for p, _ in benchmarked_set)
        if not benchmarked_p:
            return []

        gaps: list[RatioGap] = []
        for p, d in all_ratios:
            if (p, d) in benchmarked_set:
                continue
            dist = min(abs(p - bp) for bp in benchmarked_p)
            if dist >= 3:
                priority = "critical"
            elif dist >= 2:
                priority = "high"
            else:
                priority = "medium"
            gaps.append(
                RatioGap(
                    num_prefill=p,
                    num_decode=d,
                    distance_to_nearest=dist,
                    priority=priority,
                )
            )

        gaps.sort(key=lambda g: (-g.distance_to_nearest, g.num_prefill))
        return gaps

    def _generate_recommendations(
        self, metrics: CoverageMetrics, gaps: list[RatioGap]
    ) -> list[str]:
        recs: list[str] = []
        if not metrics.has_boundaries:
            recs.append("Benchmark boundary ratios (1P:ND and NP:1D) to establish extremes")
        if not metrics.has_balanced:
            recs.append("Benchmark a balanced ratio near the midpoint")
        if metrics.max_gap_size >= 3:
            critical = [g for g in gaps if g.priority == "critical"]
            if critical:
                top = critical[0]
                recs.append(
                    f"Large gap detected — benchmark {top.ratio_str} to fill the largest hole"
                )
        if metrics.coverage_fraction < 0.3:
            recs.append("Coverage below 30% — run more benchmark configurations")
        if metrics.spread_score < 0.5:
            recs.append("Benchmarks are clustered — spread them more evenly across ratios")
        if not recs:
            recs.append("Coverage is good — consider re-running benchmarks for reproducibility")
        return recs


def analyze_coverage(datasets: list[BenchmarkData]) -> dict:
    """Programmatic API for benchmark coverage analysis.

    Returns a dict with coverage report details.
    """
    analyzer = CoverageAnalyzer(datasets)
    report = analyzer.analyze()
    return report.model_dump()

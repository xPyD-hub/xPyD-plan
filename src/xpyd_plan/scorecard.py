"""Efficiency scorecard — composite 0-100 score for P:D configurations.

Combines SLA compliance, resource utilization, and cost efficiency into
a single comparable score, enabling quick ranking of configurations.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import AnalysisResult, RatioCandidate


class ScoreGrade(str, Enum):
    """Letter grade for a configuration."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension (0-100)."""

    name: str = Field(..., description="Dimension name")
    score: float = Field(..., ge=0, le=100, description="Score 0-100")
    weight: float = Field(..., ge=0, le=1, description="Weight in composite")
    details: str = Field("", description="Human-readable explanation")


class ConfigScorecard(BaseModel):
    """Scorecard for a single P:D configuration."""

    ratio: str = Field(..., description="P:D ratio string (e.g. '2P:4D')")
    num_prefill: int = Field(..., ge=1)
    num_decode: int = Field(..., ge=1)
    composite_score: float = Field(..., ge=0, le=100, description="Weighted composite score")
    grade: ScoreGrade = Field(..., description="Letter grade")
    dimensions: list[DimensionScore] = Field(..., description="Per-dimension scores")
    sla_passed: bool = Field(..., description="Whether SLA constraints were met")


class ScorecardReport(BaseModel):
    """Complete scorecard report across all evaluated configurations."""

    scorecards: list[ConfigScorecard] = Field(..., description="Scored configs, best first")
    best_ratio: str = Field(..., description="Ratio with highest composite score")
    best_score: float = Field(..., ge=0, le=100)
    summary: str = Field(..., description="Human-readable summary")


def _grade_from_score(score: float) -> ScoreGrade:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return ScoreGrade.A
    if score >= 75:
        return ScoreGrade.B
    if score >= 60:
        return ScoreGrade.C
    if score >= 40:
        return ScoreGrade.D
    return ScoreGrade.F


def _sla_score(candidate: RatioCandidate) -> tuple[float, str]:
    """Score SLA compliance (0-100).

    100 = all constraints met, 0 = all failed.
    Uses meets_sla and individual metric booleans.
    """
    sla = candidate.sla_check
    if sla is None:
        return 50.0, "No SLA check data"

    checks = [sla.meets_ttft, sla.meets_tpot, sla.meets_total_latency]
    passed_count = sum(1 for c in checks if c)
    total_count = len(checks)

    if sla.meets_all:
        score = 100.0
    else:
        # Partial credit: each passing check contributes proportionally
        score = (passed_count / total_count) * 60.0

    detail = (
        f"SLA {'PASS' if sla.meets_all else 'FAIL'} "
        f"({passed_count}/{total_count} metrics pass)"
    )
    return round(score, 2), detail


def _utilization_score(candidate: RatioCandidate) -> tuple[float, str]:
    """Score utilization efficiency (0-100).

    Optimal is 70-90%. Penalize both under and over-utilization.
    """
    p_util = min(candidate.prefill_utilization, 1.0)
    d_util = min(candidate.decode_utilization, 1.0)

    def _util_to_score(u: float) -> float:
        if u <= 0:
            return 0.0
        if u <= 0.8:
            return u / 0.8 * 100.0
        if u <= 0.95:
            return 100.0
        return max(70.0, 100.0 - (u - 0.95) * 600.0)

    p_score = _util_to_score(p_util)
    d_score = _util_to_score(d_util)
    avg = (p_score + d_score) / 2

    detail = (
        f"Prefill util {p_util:.0%} (score {p_score:.0f}), "
        f"Decode util {d_util:.0%} (score {d_score:.0f})"
    )
    return round(avg, 2), detail


def _waste_score(candidate: RatioCandidate) -> tuple[float, str]:
    """Score resource waste (0-100, higher = less waste = better)."""
    waste = candidate.waste_rate

    score = max(0.0, 100.0 * (1.0 - waste))
    detail = f"Waste rate {waste:.1%}"
    return round(score, 2), detail


class ScorecardCalculator:
    """Calculate efficiency scorecards for P:D configurations."""

    def __init__(
        self,
        sla_weight: float = 0.4,
        utilization_weight: float = 0.3,
        waste_weight: float = 0.3,
    ) -> None:
        """Initialize calculator with dimension weights.

        Args:
            sla_weight: Weight for SLA compliance dimension (0-1).
            utilization_weight: Weight for utilization dimension (0-1).
            waste_weight: Weight for waste dimension (0-1).

        Raises:
            ValueError: If weights don't sum to ~1.0 or any weight is negative.
        """
        if any(w < 0 for w in (sla_weight, utilization_weight, waste_weight)):
            raise ValueError("Weights must be non-negative")
        total = sla_weight + utilization_weight + waste_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        self.sla_weight = sla_weight
        self.utilization_weight = utilization_weight
        self.waste_weight = waste_weight

    def score(self, analysis: AnalysisResult) -> ScorecardReport:
        """Score all ratio candidates from an analysis result.

        Args:
            analysis: AnalysisResult from BenchmarkAnalyzer.

        Returns:
            ScorecardReport with ranked configurations.

        Raises:
            ValueError: If no candidates to score.
        """
        if not analysis.candidates:
            raise ValueError("No ratio candidates to score")

        scorecards: list[ConfigScorecard] = []

        for cand in analysis.candidates:
            sla_s, sla_d = _sla_score(cand)
            util_s, util_d = _utilization_score(cand)
            waste_s, waste_d = _waste_score(cand)

            dimensions = [
                DimensionScore(
                    name="SLA Compliance",
                    score=sla_s,
                    weight=self.sla_weight,
                    details=sla_d,
                ),
                DimensionScore(
                    name="Utilization",
                    score=util_s,
                    weight=self.utilization_weight,
                    details=util_d,
                ),
                DimensionScore(
                    name="Resource Waste",
                    score=waste_s,
                    weight=self.waste_weight,
                    details=waste_d,
                ),
            ]

            composite = sum(d.score * d.weight for d in dimensions)
            composite = round(min(100.0, max(0.0, composite)), 2)

            scorecards.append(ConfigScorecard(
                ratio=cand.ratio_str,
                num_prefill=cand.num_prefill,
                num_decode=cand.num_decode,
                composite_score=composite,
                grade=_grade_from_score(composite),
                dimensions=dimensions,
                sla_passed=cand.meets_sla,
            ))

        scorecards.sort(key=lambda s: s.composite_score, reverse=True)

        best = scorecards[0]
        passing = [s for s in scorecards if s.sla_passed]
        if passing:
            best_passing = passing[0]
            summary = (
                f"Best configuration: {best_passing.ratio} "
                f"(score {best_passing.composite_score:.0f}, grade {best_passing.grade.value}). "
                f"{len(passing)}/{len(scorecards)} configurations pass SLA."
            )
        else:
            summary = (
                f"No configuration passes SLA. "
                f"Highest score: {best.ratio} ({best.composite_score:.0f}, "
                f"grade {best.grade.value})."
            )

        return ScorecardReport(
            scorecards=scorecards,
            best_ratio=best.ratio,
            best_score=best.composite_score,
            summary=summary,
        )


def calculate_scorecard(
    analysis: AnalysisResult,
    sla_weight: float = 0.4,
    utilization_weight: float = 0.3,
    waste_weight: float = 0.3,
) -> dict:
    """Programmatic API for efficiency scorecard.

    Args:
        analysis: AnalysisResult from BenchmarkAnalyzer.
        sla_weight: Weight for SLA compliance (0-1).
        utilization_weight: Weight for utilization (0-1).
        waste_weight: Weight for waste (0-1).

    Returns:
        Dictionary representation of the ScorecardReport.
    """
    calc = ScorecardCalculator(
        sla_weight=sla_weight,
        utilization_weight=utilization_weight,
        waste_weight=waste_weight,
    )
    report = calc.score(analysis)
    return report.model_dump()

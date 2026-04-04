"""Tests for efficiency scorecard module."""

from __future__ import annotations

import pytest

from xpyd_plan.benchmark_models import AnalysisResult, RatioCandidate, SLACheck
from xpyd_plan.scorecard import (
    ScorecardCalculator,
    ScoreGrade,
    _grade_from_score,
    _sla_score,
    _utilization_score,
    _waste_score,
    calculate_scorecard,
)


def _make_sla_check(
    meets_ttft: bool = True,
    meets_tpot: bool = True,
    meets_total: bool = True,
) -> SLACheck:
    return SLACheck(
        ttft_p95_ms=10.0,
        ttft_p99_ms=15.0,
        tpot_p95_ms=5.0,
        tpot_p99_ms=8.0,
        total_latency_p95_ms=100.0,
        total_latency_p99_ms=150.0,
        meets_ttft=meets_ttft,
        meets_tpot=meets_tpot,
        meets_total_latency=meets_total,
        meets_all=meets_ttft and meets_tpot and meets_total,
    )


def _make_candidate(
    num_prefill: int = 2,
    num_decode: int = 4,
    prefill_util: float = 0.7,
    decode_util: float = 0.8,
    waste_rate: float = 0.1,
    meets_sla: bool = True,
    sla_check: SLACheck | None = None,
) -> RatioCandidate:
    return RatioCandidate(
        num_prefill=num_prefill,
        num_decode=num_decode,
        prefill_utilization=prefill_util,
        decode_utilization=decode_util,
        waste_rate=waste_rate,
        meets_sla=meets_sla,
        sla_check=sla_check or _make_sla_check(),
    )


def _make_analysis(*candidates: RatioCandidate) -> AnalysisResult:
    cands = list(candidates) if candidates else [_make_candidate()]
    return AnalysisResult(
        candidates=cands,
        total_instances=cands[0].num_prefill + cands[0].num_decode,
        best=cands[0] if cands[0].meets_sla else None,
    )


class TestGradeFromScore:
    def test_grade_a(self) -> None:
        assert _grade_from_score(95.0) == ScoreGrade.A
        assert _grade_from_score(90.0) == ScoreGrade.A

    def test_grade_b(self) -> None:
        assert _grade_from_score(80.0) == ScoreGrade.B
        assert _grade_from_score(75.0) == ScoreGrade.B

    def test_grade_c(self) -> None:
        assert _grade_from_score(60.0) == ScoreGrade.C
        assert _grade_from_score(70.0) == ScoreGrade.C

    def test_grade_d(self) -> None:
        assert _grade_from_score(40.0) == ScoreGrade.D
        assert _grade_from_score(55.0) == ScoreGrade.D

    def test_grade_f(self) -> None:
        assert _grade_from_score(0.0) == ScoreGrade.F
        assert _grade_from_score(39.0) == ScoreGrade.F


class TestSlaScore:
    def test_all_pass(self) -> None:
        cand = _make_candidate(sla_check=_make_sla_check())
        score, detail = _sla_score(cand)
        assert score == 100.0
        assert "PASS" in detail

    def test_all_fail(self) -> None:
        sla = _make_sla_check(meets_ttft=False, meets_tpot=False, meets_total=False)
        cand = _make_candidate(meets_sla=False, sla_check=sla)
        score, detail = _sla_score(cand)
        assert score == 0.0
        assert "FAIL" in detail

    def test_partial_pass(self) -> None:
        sla = _make_sla_check(meets_ttft=True, meets_tpot=False, meets_total=True)
        cand = _make_candidate(meets_sla=False, sla_check=sla)
        score, _ = _sla_score(cand)
        assert 0 < score < 100

    def test_no_sla_check(self) -> None:
        cand = _make_candidate(sla_check=None)
        # Override sla_check to None
        cand = RatioCandidate(
            num_prefill=2, num_decode=4,
            prefill_utilization=0.7, decode_utilization=0.8,
            waste_rate=0.1, meets_sla=True, sla_check=None,
        )
        score, detail = _sla_score(cand)
        assert score == 50.0
        assert "No SLA" in detail


class TestUtilizationScore:
    def test_optimal_utilization(self) -> None:
        cand = _make_candidate(prefill_util=0.85, decode_util=0.85)
        score, _ = _utilization_score(cand)
        assert score == 100.0

    def test_low_utilization(self) -> None:
        cand = _make_candidate(prefill_util=0.2, decode_util=0.2)
        score, _ = _utilization_score(cand)
        assert score < 50

    def test_zero_utilization(self) -> None:
        cand = _make_candidate(prefill_util=0.0, decode_util=0.0)
        score, _ = _utilization_score(cand)
        assert score == 0.0

    def test_high_utilization_slight_penalty(self) -> None:
        cand = _make_candidate(prefill_util=0.99, decode_util=0.99)
        score, _ = _utilization_score(cand)
        assert 70 <= score < 100


class TestWasteScore:
    def test_no_waste(self) -> None:
        cand = _make_candidate(waste_rate=0.0)
        score, _ = _waste_score(cand)
        assert score == 100.0

    def test_full_waste(self) -> None:
        cand = _make_candidate(waste_rate=1.0)
        score, _ = _waste_score(cand)
        assert score == 0.0

    def test_half_waste(self) -> None:
        cand = _make_candidate(waste_rate=0.5)
        score, _ = _waste_score(cand)
        assert score == 50.0


class TestScorecardCalculator:
    def test_basic_scoring(self) -> None:
        analysis = _make_analysis(_make_candidate())
        calc = ScorecardCalculator()
        report = calc.score(analysis)
        assert len(report.scorecards) == 1
        assert report.best_ratio == "2P:4D"
        assert report.best_score > 0

    def test_multiple_candidates_ranked(self) -> None:
        good = _make_candidate(
            num_prefill=2, num_decode=4,
            prefill_util=0.85, decode_util=0.85,
            waste_rate=0.05, meets_sla=True,
        )
        bad = _make_candidate(
            num_prefill=1, num_decode=5,
            prefill_util=0.2, decode_util=0.2,
            waste_rate=0.6, meets_sla=False,
            sla_check=_make_sla_check(
                meets_ttft=False, meets_tpot=False, meets_total=False,
            ),
        )
        analysis = _make_analysis(good, bad)
        report = ScorecardCalculator().score(analysis)
        assert report.scorecards[0].ratio == "2P:4D"
        assert report.scorecards[0].composite_score > report.scorecards[1].composite_score

    def test_no_candidates_raises(self) -> None:
        analysis = AnalysisResult(candidates=[], total_instances=6)
        with pytest.raises(ValueError, match="No ratio candidates"):
            ScorecardCalculator().score(analysis)

    def test_invalid_weights_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ScorecardCalculator(sla_weight=-0.1, utilization_weight=0.6, waste_weight=0.5)

    def test_invalid_weights_sum(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScorecardCalculator(sla_weight=0.5, utilization_weight=0.5, waste_weight=0.5)

    def test_custom_weights(self) -> None:
        analysis = _make_analysis(_make_candidate())
        report = ScorecardCalculator(
            sla_weight=0.8, utilization_weight=0.1, waste_weight=0.1,
        ).score(analysis)
        assert report.best_score > 0

    def test_all_fail_sla_summary(self) -> None:
        sla = _make_sla_check(meets_ttft=False, meets_tpot=False, meets_total=False)
        cand = _make_candidate(meets_sla=False, sla_check=sla)
        report = ScorecardCalculator().score(_make_analysis(cand))
        assert "No configuration passes SLA" in report.summary

    def test_grade_assignment(self) -> None:
        analysis = _make_analysis(_make_candidate(
            prefill_util=0.85, decode_util=0.85,
            waste_rate=0.0, meets_sla=True,
        ))
        report = ScorecardCalculator().score(analysis)
        assert report.scorecards[0].grade in (ScoreGrade.A, ScoreGrade.B)

    def test_dimensions_have_correct_weights(self) -> None:
        report = ScorecardCalculator(
            sla_weight=0.5, utilization_weight=0.3, waste_weight=0.2,
        ).score(_make_analysis())
        dims = report.scorecards[0].dimensions
        assert dims[0].weight == 0.5
        assert dims[1].weight == 0.3
        assert dims[2].weight == 0.2


class TestCalculateScorecard:
    def test_programmatic_api(self) -> None:
        analysis = _make_analysis(_make_candidate())
        result = calculate_scorecard(analysis)
        assert "scorecards" in result
        assert "best_ratio" in result
        assert "best_score" in result
        assert "summary" in result
        assert len(result["scorecards"]) == 1

    def test_custom_weights_api(self) -> None:
        analysis = _make_analysis(_make_candidate())
        result = calculate_scorecard(
            analysis, sla_weight=0.6, utilization_weight=0.2, waste_weight=0.2,
        )
        assert result["best_score"] > 0


class TestConfigScorecard:
    def test_sla_passed_flag(self) -> None:
        cand_pass = _make_candidate(meets_sla=True)
        cand_fail = _make_candidate(
            num_prefill=1, num_decode=5, meets_sla=False,
            sla_check=_make_sla_check(
                meets_ttft=False, meets_tpot=False, meets_total=False,
            ),
        )
        report = ScorecardCalculator().score(_make_analysis(cand_pass, cand_fail))
        pass_cards = [s for s in report.scorecards if s.sla_passed]
        fail_cards = [s for s in report.scorecards if not s.sla_passed]
        assert len(pass_cards) == 1
        assert len(fail_cards) == 1


class TestEdgeCases:
    def test_single_candidate(self) -> None:
        report = ScorecardCalculator().score(_make_analysis(_make_candidate()))
        assert len(report.scorecards) == 1
        assert report.best_ratio == report.scorecards[0].ratio

    def test_score_bounds(self) -> None:
        """Composite score must always be 0-100."""
        cand = _make_candidate(
            prefill_util=0.0, decode_util=0.0,
            waste_rate=1.0, meets_sla=False,
            sla_check=_make_sla_check(
                meets_ttft=False, meets_tpot=False, meets_total=False,
            ),
        )
        report = ScorecardCalculator().score(_make_analysis(cand))
        for sc in report.scorecards:
            assert 0 <= sc.composite_score <= 100

    def test_perfect_candidate(self) -> None:
        cand = _make_candidate(
            prefill_util=0.85, decode_util=0.85,
            waste_rate=0.0, meets_sla=True,
        )
        report = ScorecardCalculator().score(_make_analysis(cand))
        assert report.scorecards[0].composite_score == 100.0
        assert report.scorecards[0].grade == ScoreGrade.A

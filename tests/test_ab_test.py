"""Tests for A/B test analysis (M33)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xpyd_plan.ab_test import (
    ABTestAnalyzer,
    ABTestConfig,
    ABTestReport,
    EffectMagnitude,
    EffectSize,
    PowerWarning,
    _classify_cohens_d,
    _cohens_d,
    _mann_whitney_u,
    _norm_cdf,
    _norm_quantile,
    _t_cdf,
    _welch_t_test,
    analyze_ab_test,
)


def _make_benchmark(
    num_prefill: int = 2,
    num_decode: int = 6,
    qps: float = 10.0,
    ttft_base: float = 50.0,
    tpot_base: float = 20.0,
    total_base: float = 200.0,
    n_requests: int = 100,
    seed: int = 42,
) -> dict:
    """Create a synthetic benchmark dict."""
    rng = np.random.default_rng(seed)
    requests = []
    for i in range(n_requests):
        ttft = max(1.0, ttft_base + rng.normal(0, ttft_base * 0.1))
        tpot = max(1.0, tpot_base + rng.normal(0, tpot_base * 0.1))
        total = max(1.0, total_base + rng.normal(0, total_base * 0.1))
        requests.append({
            "request_id": f"req-{i}",
            "prompt_tokens": int(rng.integers(50, 500)),
            "output_tokens": int(rng.integers(10, 200)),
            "ttft_ms": float(ttft),
            "tpot_ms": float(tpot),
            "total_latency_ms": float(total),
            "timestamp": 1700000000.0 + i,
        })
    return {
        "metadata": {
            "num_prefill_instances": num_prefill,
            "num_decode_instances": num_decode,
            "total_instances": num_prefill + num_decode,
            "measured_qps": qps,
        },
        "requests": requests,
    }


def _write_benchmark(tmp_dir: Path, name: str, **kwargs) -> str:
    path = tmp_dir / f"{name}.json"
    path.write_text(json.dumps(_make_benchmark(**kwargs)))
    return str(path)


class TestStatisticalHelpers:
    def test_norm_cdf_center(self):
        assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-10)

    def test_norm_cdf_positive(self):
        assert _norm_cdf(1.96) == pytest.approx(0.975, abs=0.001)

    def test_norm_cdf_negative(self):
        assert _norm_cdf(-1.96) == pytest.approx(0.025, abs=0.001)

    def test_t_cdf_large_df(self):
        # For large df, should approximate normal
        assert _t_cdf(1.96, 100) == pytest.approx(0.975, abs=0.01)

    def test_t_cdf_small_df(self):
        # For small df, heavier tails
        val = _t_cdf(1.96, 5)
        assert 0.9 < val < 0.98

    def test_norm_quantile_center(self):
        assert _norm_quantile(0.5) == pytest.approx(0.0, abs=1e-10)

    def test_norm_quantile_975(self):
        assert _norm_quantile(0.975) == pytest.approx(1.96, abs=0.01)

    def test_norm_quantile_extremes(self):
        assert _norm_quantile(0.0) == -10.0
        assert _norm_quantile(1.0) == 10.0


class TestClassifyCohensD:
    def test_negligible(self):
        assert _classify_cohens_d(0.1) == EffectMagnitude.NEGLIGIBLE

    def test_small(self):
        assert _classify_cohens_d(0.3) == EffectMagnitude.SMALL

    def test_medium(self):
        assert _classify_cohens_d(0.6) == EffectMagnitude.MEDIUM

    def test_large(self):
        assert _classify_cohens_d(1.0) == EffectMagnitude.LARGE

    def test_negative(self):
        assert _classify_cohens_d(-0.9) == EffectMagnitude.LARGE


class TestWelchTTest:
    def test_same_distributions(self):
        rng = np.random.default_rng(42)
        a = rng.normal(50, 5, 100)
        b = rng.normal(50, 5, 100)
        result = _welch_t_test(a, b, 0.05)
        assert result.test_name == "welch_t"
        assert not result.significant  # same distribution, should not be significant

    def test_different_distributions(self):
        rng = np.random.default_rng(42)
        a = rng.normal(50, 5, 200)
        b = rng.normal(60, 5, 200)
        result = _welch_t_test(a, b, 0.05)
        assert result.significant
        assert result.statistic > 0  # b > a

    def test_p_value_bounds(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        result = _welch_t_test(a, b, 0.05)
        assert 0.0 <= result.p_value <= 1.0


class TestMannWhitneyU:
    def test_same_distributions(self):
        rng = np.random.default_rng(42)
        a = rng.normal(50, 5, 100)
        b = rng.normal(50, 5, 100)
        result = _mann_whitney_u(a, b, 0.05)
        assert result.test_name == "mann_whitney_u"
        assert not result.significant

    def test_different_distributions(self):
        rng = np.random.default_rng(42)
        a = rng.normal(50, 5, 200)
        b = rng.normal(70, 5, 200)
        result = _mann_whitney_u(a, b, 0.05)
        assert result.significant


class TestCohensD:
    def test_same_values(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _cohens_d(a, b) == pytest.approx(0.0)

    def test_large_effect(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        d = _cohens_d(a, b)
        assert abs(d) > 0.8  # large effect

    def test_zero_variance(self):
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        assert _cohens_d(a, b) == 0.0


class TestPowerWarning:
    def test_adequate(self):
        pw = PowerWarning(adequate=True, reason="OK", control_n=100, treatment_n=100)
        assert pw.adequate

    def test_inadequate(self):
        pw = PowerWarning(adequate=False, reason="Too small", control_n=10, treatment_n=10)
        assert not pw.adequate


class TestEffectSize:
    def test_fields(self):
        es = EffectSize(cohens_d=0.5, magnitude=EffectMagnitude.MEDIUM)
        assert es.cohens_d == 0.5
        assert es.magnitude == EffectMagnitude.MEDIUM


class TestABTestAnalyzer:
    def test_same_data(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=42)
        analyzer = ABTestAnalyzer()
        report = analyzer.analyze(path_a, path_b)

        assert isinstance(report, ABTestReport)
        assert len(report.results) == 3
        for r in report.results:
            assert not r.welch_t.significant
            assert r.winner is None

    def test_different_data(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", ttft_base=50.0, seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", ttft_base=80.0, seed=99)
        analyzer = ABTestAnalyzer()
        report = analyzer.analyze(path_a, path_b)

        ttft_result = next(r for r in report.results if r.metric == "ttft_ms")
        assert ttft_result.welch_t.significant
        assert ttft_result.winner == "control"  # lower latency wins
        assert ttft_result.mean_difference > 0  # treatment higher

    def test_custom_alpha(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=43)
        config = ABTestConfig(alpha=0.001)
        analyzer = ABTestAnalyzer(config=config)
        report = analyzer.analyze(path_a, path_b)
        assert report.alpha == 0.001

    def test_custom_metrics(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=42)
        config = ABTestConfig(metrics=["ttft_ms"])
        analyzer = ABTestAnalyzer(config=config)
        report = analyzer.analyze(path_a, path_b)
        assert len(report.results) == 1
        assert report.results[0].metric == "ttft_ms"

    def test_invalid_metric(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=42)
        config = ABTestConfig(metrics=["invalid_metric"])
        analyzer = ABTestAnalyzer(config=config)
        with pytest.raises(ValueError, match="Invalid metric"):
            analyzer.analyze(path_a, path_b)

    def test_small_sample_warning(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", n_requests=10, seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", n_requests=10, seed=43)
        analyzer = ABTestAnalyzer()
        report = analyzer.analyze(path_a, path_b)
        for r in report.results:
            assert not r.power_warning.adequate

    def test_summary_no_significant(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=42)
        report = ABTestAnalyzer().analyze(path_a, path_b)
        assert "No statistically significant" in report.summary

    def test_summary_with_significant(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", ttft_base=50.0, seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", ttft_base=100.0, seed=99)
        report = ABTestAnalyzer().analyze(path_a, path_b)
        assert "Significant differences" in report.summary

    def test_ci_contains_zero_for_same_data(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=42)
        report = ABTestAnalyzer().analyze(path_a, path_b)
        for r in report.results:
            assert r.ci.lower <= 0 <= r.ci.upper

    def test_effect_size_negligible_for_same_data(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=42)
        report = ABTestAnalyzer().analyze(path_a, path_b)
        for r in report.results:
            assert r.effect_size.magnitude == EffectMagnitude.NEGLIGIBLE


class TestAnalyzeABTestAPI:
    def test_default(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=42)
        report = analyze_ab_test(path_a, path_b)
        assert isinstance(report, ABTestReport)
        assert len(report.results) == 3

    def test_custom_params(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=42)
        report = analyze_ab_test(path_a, path_b, alpha=0.01, metrics=["tpot_ms"])
        assert len(report.results) == 1
        assert report.alpha == 0.01

    def test_json_serialization(self, tmp_path):
        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=42)
        report = analyze_ab_test(path_a, path_b)
        j = json.loads(report.model_dump_json())
        assert "results" in j
        assert "summary" in j
        assert len(j["results"]) == 3


class TestCLIABTest:
    def test_cli_table_output(self, tmp_path):
        from xpyd_plan.cli._main import main

        path_a = _write_benchmark(tmp_path, "control", ttft_base=50.0, seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", ttft_base=80.0, seed=99)
        main([
            "ab-test",
            "--control", path_a,
            "--treatment", path_b,
        ])

    def test_cli_json_output(self, tmp_path, capsys):
        from xpyd_plan.cli._main import main

        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=42)
        main([
            "ab-test",
            "--control", path_a,
            "--treatment", path_b,
            "--output-format", "json",
        ])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "results" in data

    def test_cli_custom_alpha(self, tmp_path):
        from xpyd_plan.cli._main import main

        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=42)
        main([
            "ab-test",
            "--control", path_a,
            "--treatment", path_b,
            "--alpha", "0.01",
        ])

    def test_cli_custom_metric(self, tmp_path, capsys):
        from xpyd_plan.cli._main import main

        path_a = _write_benchmark(tmp_path, "control", seed=42)
        path_b = _write_benchmark(tmp_path, "treatment", seed=42)
        main([
            "ab-test",
            "--control", path_a,
            "--treatment", path_b,
            "--metric", "ttft_ms",
            "--output-format", "json",
        ])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert len(data["results"]) == 1

"""Tests for latency-token regression analysis."""

from __future__ import annotations

import json
import subprocess
import tempfile

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.regression import (
    RegressionAnalyzer,
    RegressionReport,
    analyze_regression,
)


def _make_benchmark(
    num_requests: int = 100,
    linear: bool = True,
) -> BenchmarkData:
    """Create benchmark data. If linear=True, latency scales linearly with tokens."""
    requests = []
    for i in range(num_requests):
        prompt = 50 + i * 10
        output = 20 + i * 5
        if linear:
            ttft = 10.0 + 0.05 * prompt + (i % 3) * 0.5  # small noise
            tpot = 5.0 + 0.1 * output + (i % 4) * 0.3
            total = ttft + tpot * output
        else:
            # Random-ish, weak correlation
            ttft = 50.0 + (i % 7) * 10.0
            tpot = 20.0 + (i % 5) * 8.0
            total = 300.0 + (i % 11) * 50.0

        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=prompt,
                output_tokens=output,
                ttft_ms=ttft,
                tpot_ms=tpot,
                total_latency_ms=total,
                timestamp=1000.0 + i * 0.1,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=6,
            total_instances=8,
            measured_qps=10.0,
        ),
        requests=requests,
    )


class TestRegressionAnalyzer:
    """Tests for RegressionAnalyzer."""

    def test_basic_fits(self) -> None:
        data = _make_benchmark(linear=True)
        analyzer = RegressionAnalyzer(data)
        report = analyzer.analyze()

        assert len(report.fits) == 3
        assert report.fits[0].predictor == "prompt_tokens"
        assert report.fits[0].target == "ttft_ms"
        assert report.fits[1].predictor == "output_tokens"
        assert report.fits[1].target == "tpot_ms"
        assert report.fits[2].predictor == "total_tokens"
        assert report.fits[2].target == "total_latency_ms"

    def test_linear_data_high_r_squared(self) -> None:
        data = _make_benchmark(linear=True)
        report = analyze_regression(data)
        # With strong linear relationship, R² should be high
        for fit in report.fits:
            assert fit.r_squared > 0.8, f"{fit.predictor}->{fit.target} R²={fit.r_squared}"

    def test_weak_correlation_low_r_squared(self) -> None:
        data = _make_benchmark(linear=False)
        report = analyze_regression(data)
        # TTFT with weak pattern should have lower R²
        ttft_fit = report.fits[0]
        # Not necessarily very low but shouldn't be near 1.0
        assert ttft_fit.r_squared < 0.95

    def test_slope_positive_for_linear(self) -> None:
        data = _make_benchmark(linear=True)
        report = analyze_regression(data)
        for fit in report.fits:
            assert fit.slope > 0, f"{fit.predictor} slope should be positive"

    def test_n_samples(self) -> None:
        data = _make_benchmark(num_requests=50, linear=True)
        report = analyze_regression(data)
        for fit in report.fits:
            assert fit.n_samples == 50

    def test_no_predictions_by_default(self) -> None:
        data = _make_benchmark(linear=True)
        report = analyze_regression(data)
        assert report.predictions == []

    def test_predict_prompt(self) -> None:
        data = _make_benchmark(linear=True)
        report = analyze_regression(data, predict_prompt=500)
        assert len(report.predictions) == 1
        pred = report.predictions[0]
        assert pred.metric == "ttft_ms"
        assert pred.token_count == 500
        assert pred.ci_lower_ms <= pred.predicted_ms <= pred.ci_upper_ms

    def test_predict_output(self) -> None:
        data = _make_benchmark(linear=True)
        report = analyze_regression(data, predict_output=200)
        assert len(report.predictions) == 1
        pred = report.predictions[0]
        assert pred.metric == "tpot_ms"
        assert pred.token_count == 200

    def test_predict_both(self) -> None:
        data = _make_benchmark(linear=True)
        report = analyze_regression(data, predict_prompt=500, predict_output=200)
        assert len(report.predictions) == 3
        metrics = [p.metric for p in report.predictions]
        assert "ttft_ms" in metrics
        assert "tpot_ms" in metrics
        assert "total_latency_ms" in metrics

    def test_total_prediction_token_count(self) -> None:
        data = _make_benchmark(linear=True)
        report = analyze_regression(data, predict_prompt=500, predict_output=200)
        total_pred = [p for p in report.predictions if p.metric == "total_latency_ms"][0]
        assert total_pred.token_count == 700

    def test_ci_width(self) -> None:
        data = _make_benchmark(linear=True, num_requests=100)
        report = analyze_regression(data, predict_prompt=500)
        pred = report.predictions[0]
        ci_width = pred.ci_upper_ms - pred.ci_lower_ms
        assert ci_width > 0

    def test_small_dataset(self) -> None:
        """Should work with minimum viable data (2 points)."""
        requests = [
            BenchmarkRequest(
                request_id="r0", prompt_tokens=100, output_tokens=50,
                ttft_ms=20.0, tpot_ms=10.0, total_latency_ms=100.0, timestamp=1000.0,
            ),
            BenchmarkRequest(
                request_id="r1", prompt_tokens=200, output_tokens=100,
                ttft_ms=30.0, tpot_ms=15.0, total_latency_ms=200.0, timestamp=1001.0,
            ),
        ]
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1, num_decode_instances=1,
                total_instances=2, measured_qps=2.0,
            ),
            requests=requests,
        )
        report = analyze_regression(data)
        assert len(report.fits) == 3
        # Perfect fit with 2 points
        for fit in report.fits:
            assert fit.r_squared == pytest.approx(1.0, abs=0.01)

    def test_constant_predictor(self) -> None:
        """All same token count -> slope 0, R²=0."""
        requests = [
            BenchmarkRequest(
                request_id=f"r{i}", prompt_tokens=100, output_tokens=50,
                ttft_ms=20.0 + i, tpot_ms=10.0 + i, total_latency_ms=100.0 + i,
                timestamp=1000.0 + i,
            )
            for i in range(10)
        ]
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1, num_decode_instances=1,
                total_instances=2, measured_qps=10.0,
            ),
            requests=requests,
        )
        report = analyze_regression(data)
        ttft_fit = report.fits[0]
        assert ttft_fit.slope == 0.0
        assert ttft_fit.r_squared == 0.0

    def test_report_serialization(self) -> None:
        data = _make_benchmark(linear=True)
        report = analyze_regression(data, predict_prompt=500)
        d = report.model_dump()
        assert "fits" in d
        assert "predictions" in d
        # Round-trip
        report2 = RegressionReport.model_validate(d)
        assert len(report2.fits) == len(report.fits)

    def test_slope_std_err_positive(self) -> None:
        data = _make_benchmark(linear=True, num_requests=50)
        report = analyze_regression(data)
        for fit in report.fits:
            assert fit.slope_std_err >= 0


class TestCLIRegression:
    """Test CLI integration."""

    def test_cli_json_output(self) -> None:

        data = _make_benchmark(linear=True, num_requests=20)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json())
            f.flush()

            result = subprocess.run(
                ["xpyd-plan", "regression",
                 "--benchmark", f.name, "--output-format", "json"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, result.stderr
            out = json.loads(result.stdout)
            assert "fits" in out
            assert len(out["fits"]) == 3

    def test_cli_with_prediction(self) -> None:

        data = _make_benchmark(linear=True, num_requests=20)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json())
            f.flush()

            result = subprocess.run(
                ["xpyd-plan", "regression",
                 "--benchmark", f.name, "--predict-prompt", "500",
                 "--predict-output", "200", "--output-format", "json"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, result.stderr
            out = json.loads(result.stdout)
            assert len(out["predictions"]) == 3

    def test_cli_table_output(self) -> None:

        data = _make_benchmark(linear=True, num_requests=20)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json())
            f.flush()

            result = subprocess.run(
                ["xpyd-plan", "regression",
                 "--benchmark", f.name],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, result.stderr
            assert "Regression" in result.stdout

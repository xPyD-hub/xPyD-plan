"""Tests for benchmark duration advisor."""

from __future__ import annotations

import json
import tempfile

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.duration_advisor import (
    DurationAdvisor,
    DurationAdvisorReport,
    DurationVerdict,
    advise_duration,
)


def _make_requests(
    n: int,
    *,
    base_ttft: float = 50.0,
    base_tpot: float = 10.0,
    base_total: float = 200.0,
    ttft_noise: float = 5.0,
    tpot_noise: float = 1.0,
    total_noise: float = 20.0,
    qps: float = 10.0,
    warmup_count: int = 0,
    warmup_factor: float = 3.0,
) -> list[BenchmarkRequest]:
    """Generate synthetic requests with optional warmup period."""
    import random

    rng = random.Random(42)
    requests = []
    for i in range(n):
        t = i / qps
        if i < warmup_count:
            factor = warmup_factor * (1 - i / warmup_count) + 1.0
        else:
            factor = 1.0
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=rng.randint(100, 500),
                output_tokens=rng.randint(50, 200),
                ttft_ms=max(1, base_ttft * factor + rng.gauss(0, ttft_noise)),
                tpot_ms=max(1, base_tpot * factor + rng.gauss(0, tpot_noise)),
                total_latency_ms=max(1, base_total * factor + rng.gauss(0, total_noise)),
                timestamp=1000.0 + t,
            )
        )
    return requests


def _make_data(requests: list[BenchmarkRequest]) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=4,
            total_instances=6,
            measured_qps=10.0,
        ),
        requests=requests,
    )


class TestDurationAdvisor:
    def test_stable_benchmark_sufficient(self):
        """A long, stable benchmark should be SUFFICIENT."""
        requests = _make_requests(500, ttft_noise=2.0, tpot_noise=0.5, total_noise=5.0)
        data = _make_data(requests)
        advisor = DurationAdvisor(data)
        report = advisor.analyze()
        assert report.verdict == DurationVerdict.SUFFICIENT
        assert report.actual_request_count == 500
        assert all(m.stabilized for m in report.metrics)

    def test_too_few_requests(self):
        """Less than 10 requests should be TOO_SHORT."""
        requests = _make_requests(5)
        data = _make_data(requests)
        report = DurationAdvisor(data).analyze()
        assert report.verdict == DurationVerdict.TOO_SHORT
        assert report.metrics == []

    def test_unstable_benchmark(self):
        """Highly noisy data should not fully stabilize."""
        requests = _make_requests(
            50, ttft_noise=50.0, tpot_noise=10.0, total_noise=100.0
        )
        data = _make_data(requests)
        report = DurationAdvisor(data, tolerance=0.01).analyze()
        # With very tight tolerance and high noise, at least some metrics won't stabilize
        assert report.verdict in (
            DurationVerdict.INSUFFICIENT,
            DurationVerdict.MARGINAL,
        )

    def test_warmup_affects_stabilization(self):
        """Warmup period should delay stabilization."""
        requests = _make_requests(
            200, warmup_count=50, warmup_factor=5.0, ttft_noise=2.0
        )
        data = _make_data(requests)
        report = DurationAdvisor(data).analyze()
        assert report.actual_request_count == 200
        # Should still stabilize eventually
        stabilized_metrics = [m for m in report.metrics if m.stabilized]
        assert len(stabilized_metrics) >= 1

    def test_per_metric_details(self):
        """Each of 3 metrics should be reported."""
        requests = _make_requests(300)
        data = _make_data(requests)
        report = DurationAdvisor(data).analyze()
        assert len(report.metrics) == 3
        metric_names = {m.metric for m in report.metrics}
        assert metric_names == {"ttft_ms", "tpot_ms", "total_latency_ms"}

    def test_stabilization_request_index(self):
        """Stabilized metrics should have a request index."""
        requests = _make_requests(500, ttft_noise=1.0, tpot_noise=0.2, total_noise=3.0)
        data = _make_data(requests)
        report = DurationAdvisor(data).analyze()
        for m in report.metrics:
            if m.stabilized:
                assert m.stabilization_request_index is not None
                assert m.stabilization_request_index > 0
                assert m.stabilization_time_s is not None
                assert m.stabilization_time_s >= 0

    def test_recommended_requests_at_least_100(self):
        """Minimum recommended request count should be 100."""
        requests = _make_requests(500, ttft_noise=0.1, tpot_noise=0.01, total_noise=0.1)
        data = _make_data(requests)
        report = DurationAdvisor(data).analyze()
        assert report.recommended_request_count >= 100

    def test_safety_multiplier(self):
        """Higher safety multiplier should increase recommendations."""
        requests = _make_requests(300, ttft_noise=2.0)
        data = _make_data(requests)
        r1 = DurationAdvisor(data, safety_multiplier=1.5).analyze()
        r2 = DurationAdvisor(data, safety_multiplier=3.0).analyze()
        assert r2.recommended_request_count >= r1.recommended_request_count

    def test_custom_percentile(self):
        """Should work with P99 instead of P95."""
        requests = _make_requests(300)
        data = _make_data(requests)
        report = DurationAdvisor(data, percentile=99.0).analyze()
        assert isinstance(report, DurationAdvisorReport)
        assert len(report.metrics) == 3

    def test_tolerance_affects_stabilization(self):
        """Tighter tolerance should make stabilization harder."""
        requests = _make_requests(200, ttft_noise=10.0)
        data = _make_data(requests)
        r_loose = DurationAdvisor(data, tolerance=0.20).analyze()
        r_tight = DurationAdvisor(data, tolerance=0.01).analyze()
        loose_stab = sum(1 for m in r_loose.metrics if m.stabilized)
        tight_stab = sum(1 for m in r_tight.metrics if m.stabilized)
        assert loose_stab >= tight_stab

    def test_final_cv_nonnegative(self):
        """Final CV should be non-negative."""
        requests = _make_requests(200)
        data = _make_data(requests)
        report = DurationAdvisor(data).analyze()
        for m in report.metrics:
            assert m.final_cv >= 0

    def test_actual_duration_computed(self):
        """Duration should be computed from timestamps."""
        requests = _make_requests(100, qps=10.0)
        data = _make_data(requests)
        report = DurationAdvisor(data).analyze()
        # 100 requests at 10 qps = ~9.9s
        assert report.actual_duration_s > 9.0
        assert report.actual_duration_s < 11.0

    def test_recommended_duration_positive(self):
        """Recommended duration should be positive for non-trivial input."""
        requests = _make_requests(200)
        data = _make_data(requests)
        report = DurationAdvisor(data).analyze()
        assert report.recommended_duration_s > 0

    def test_model_dump_serializable(self):
        """Report should be JSON-serializable."""
        requests = _make_requests(200)
        data = _make_data(requests)
        report = DurationAdvisor(data).analyze()
        dumped = report.model_dump()
        assert json.dumps(dumped)  # Should not raise

    def test_advise_duration_api(self):
        """Programmatic API should return a dict."""
        requests = _make_requests(200)
        data = _make_data(requests)
        result = advise_duration(data)
        assert isinstance(result, dict)
        assert "verdict" in result
        assert "metrics" in result
        assert "recommended_request_count" in result

    def test_advise_duration_api_kwargs(self):
        """API should accept keyword arguments."""
        requests = _make_requests(200)
        data = _make_data(requests)
        result = advise_duration(
            data, percentile=99.0, tolerance=0.10, safety_multiplier=2.0
        )
        assert isinstance(result, dict)

    def test_window_steps_minimum(self):
        """Window steps should be at least 5."""
        requests = _make_requests(200)
        data = _make_data(requests)
        report = DurationAdvisor(data, window_steps=2).analyze()
        # Should not crash; internally clamped to 5
        assert isinstance(report, DurationAdvisorReport)

    def test_identical_values(self):
        """Should handle constant latency gracefully."""
        requests = _make_requests(200, ttft_noise=0.0, tpot_noise=0.0, total_noise=0.0)
        data = _make_data(requests)
        report = DurationAdvisor(data).analyze()
        assert report.verdict == DurationVerdict.SUFFICIENT
        for m in report.metrics:
            assert m.stabilized

    def test_cli_json_output(self):
        """CLI JSON output should be valid."""
        import subprocess

        requests = _make_requests(100)
        data = _make_data(requests)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(data.model_dump_json())
            f.flush()
            result = subprocess.run(
                [
                    "xpyd-plan",
                    "duration-advisor",
                    "--benchmark",
                    f.name,
                    "--output-format",
                    "json",
                ],
                capture_output=True,
                text=True,
            )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "verdict" in output

    def test_verdict_values(self):
        """DurationVerdict enum should have expected values."""
        assert DurationVerdict.SUFFICIENT.value == "SUFFICIENT"
        assert DurationVerdict.MARGINAL.value == "MARGINAL"
        assert DurationVerdict.INSUFFICIENT.value == "INSUFFICIENT"
        assert DurationVerdict.TOO_SHORT.value == "TOO_SHORT"

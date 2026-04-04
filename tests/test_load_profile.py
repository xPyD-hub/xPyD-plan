"""Tests for load profile classification."""

from __future__ import annotations

import json
import random
import tempfile

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.load_profile import (
    LoadProfile,
    LoadProfileClassifier,
    ProfileType,
    RateWindow,
    classify_load_profile,
)


def _make_requests(
    n: int = 200,
    *,
    seed: int = 42,
    pattern: str = "steady",
    duration: float = 100.0,
) -> list[BenchmarkRequest]:
    """Generate requests with a specific arrival pattern."""
    rng = random.Random(seed)
    requests = []

    for i in range(n):
        prompt_tokens = rng.randint(50, 500)
        output_tokens = rng.randint(20, 200)
        ttft = 50.0 + rng.gauss(0, 5)
        tpot = 10.0 + rng.gauss(0, 1)
        total = max(ttft + tpot * output_tokens, 1.0)

        if pattern == "steady":
            ts = 1700000000.0 + (i / n) * duration
        elif pattern == "ramp_up":
            # Quadratic: more requests at the end
            frac = (i / n) ** 0.5
            ts = 1700000000.0 + frac * duration
        elif pattern == "ramp_down":
            # Inverse: more requests at the start
            frac = 1.0 - ((n - 1 - i) / n) ** 0.5
            ts = 1700000000.0 + frac * duration
        elif pattern == "burst":
            # All requests in 2 short bursts
            if i < n // 2:
                ts = 1700000000.0 + rng.uniform(0, 2)
            else:
                ts = 1700000000.0 + duration - 2 + rng.uniform(0, 2)
        elif pattern == "cyclic":
            # Sine-wave modulated arrival
            import math
            phase = (i / n) * 4 * math.pi  # 2 full cycles
            offset = (math.sin(phase) + 1) / 2  # 0 to 1
            ts = 1700000000.0 + (i / n) * duration + offset * 2
        else:
            ts = 1700000000.0 + (i / n) * duration

        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                ttft_ms=round(max(ttft, 0.1), 2),
                tpot_ms=round(max(tpot, 0.1), 2),
                total_latency_ms=round(total, 2),
                timestamp=ts,
            )
        )

    return requests


def _make_benchmark(requests: list[BenchmarkRequest]) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=4,
            total_instances=6,
            measured_qps=10.0,
        ),
        requests=requests,
    )


class TestLoadProfileClassifier:
    """Tests for LoadProfileClassifier."""

    def test_steady_state_classification(self):
        data = _make_benchmark(_make_requests(500, pattern="steady", duration=100))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify(window_size=5.0)
        assert report.profile.profile_type == ProfileType.STEADY_STATE

    def test_ramp_up_classification(self):
        data = _make_benchmark(_make_requests(500, pattern="ramp_up", duration=100))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify(window_size=5.0)
        assert report.profile.profile_type in (ProfileType.RAMP_UP, ProfileType.BURST)

    def test_ramp_down_classification(self):
        data = _make_benchmark(_make_requests(500, pattern="ramp_down", duration=100))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify(window_size=5.0)
        assert report.profile.profile_type in (ProfileType.RAMP_DOWN, ProfileType.BURST)

    def test_burst_classification(self):
        data = _make_benchmark(_make_requests(500, pattern="burst", duration=100))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify(window_size=5.0)
        assert report.profile.profile_type == ProfileType.BURST

    def test_report_has_windows(self):
        data = _make_benchmark(_make_requests(200, pattern="steady", duration=50))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify(window_size=5.0)
        assert len(report.windows) == 10

    def test_report_total_requests(self):
        data = _make_benchmark(_make_requests(200, pattern="steady"))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify()
        assert report.total_requests == 200

    def test_report_duration(self):
        data = _make_benchmark(_make_requests(200, pattern="steady", duration=100))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify()
        assert report.duration_seconds > 0

    def test_overall_rate(self):
        data = _make_benchmark(_make_requests(200, pattern="steady", duration=100))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify()
        assert report.overall_rate_rps > 0

    def test_confidence_range(self):
        data = _make_benchmark(_make_requests(200, pattern="steady"))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify()
        assert 0.0 <= report.profile.confidence <= 1.0

    def test_rate_cv_nonnegative(self):
        data = _make_benchmark(_make_requests(200, pattern="steady"))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify()
        assert report.profile.rate_cv >= 0

    def test_peak_trough_at_least_one(self):
        data = _make_benchmark(_make_requests(200, pattern="steady"))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify()
        assert report.profile.peak_to_trough_ratio >= 1.0

    def test_single_request(self):
        reqs = _make_requests(1, pattern="steady")
        data = _make_benchmark(reqs)
        classifier = LoadProfileClassifier(data)
        report = classifier.classify()
        assert report.total_requests == 1

    def test_custom_window_size(self):
        data = _make_benchmark(_make_requests(200, pattern="steady", duration=100))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify(window_size=10.0)
        assert len(report.windows) == 10

    def test_description_not_empty(self):
        data = _make_benchmark(_make_requests(200, pattern="steady"))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify()
        assert len(report.profile.description) > 0

    def test_window_rates_nonnegative(self):
        data = _make_benchmark(_make_requests(200, pattern="steady"))
        classifier = LoadProfileClassifier(data)
        report = classifier.classify()
        for w in report.windows:
            assert w.rate_rps >= 0
            assert w.request_count >= 0

    def test_profile_type_enum(self):
        assert ProfileType.STEADY_STATE.value == "steady_state"
        assert ProfileType.RAMP_UP.value == "ramp_up"
        assert ProfileType.BURST.value == "burst"
        assert ProfileType.CYCLIC.value == "cyclic"

    def test_rate_window_model(self):
        w = RateWindow(
            start_time=1700000000.0,
            end_time=1700000005.0,
            request_count=50,
            rate_rps=10.0,
        )
        assert w.request_count == 50

    def test_load_profile_model(self):
        p = LoadProfile(
            profile_type=ProfileType.STEADY_STATE,
            confidence=0.9,
            rate_cv=0.05,
            rate_trend_slope=0.0,
            peak_to_trough_ratio=1.2,
            description="Stable.",
        )
        assert p.profile_type == ProfileType.STEADY_STATE


class TestClassifyLoadProfileAPI:
    """Tests for the programmatic API."""

    def test_returns_dict(self):
        data = _make_benchmark(_make_requests(100, pattern="steady"))
        result = classify_load_profile(data)
        assert isinstance(result, dict)
        assert "profile" in result
        assert "windows" in result
        assert "total_requests" in result

    def test_custom_window_size(self):
        data = _make_benchmark(_make_requests(100, pattern="steady", duration=50))
        result = classify_load_profile(data, window_size=10.0)
        assert isinstance(result, dict)
        assert len(result["windows"]) == 5


class TestLoadProfileCLI:
    """Tests for CLI integration."""

    def test_cli_json_output(self):
        import subprocess

        data = _make_benchmark(_make_requests(100, pattern="steady", duration=50))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json(indent=2))
            f.flush()
            result = subprocess.run(
                ["xpyd-plan", "load-profile",
                 "--benchmark", f.name, "--output-format", "json"],
                capture_output=True, text=True,
            )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "profile" in output
        assert "windows" in output

    def test_cli_table_output(self):
        import subprocess

        data = _make_benchmark(_make_requests(100, pattern="steady", duration=50))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json(indent=2))
            f.flush()
            result = subprocess.run(
                ["xpyd-plan", "load-profile",
                 "--benchmark", f.name, "--output-format", "table"],
                capture_output=True, text=True,
            )
        assert result.returncode == 0
        assert "rate" in result.stdout.lower() or "profile" in result.stdout.lower()

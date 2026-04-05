"""Tests for replay module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.replay import (
    ReplayConfig,
    ReplayEntry,
    ReplayGenerator,
    generate_replay,
)


def _make_benchmark_data(n: int = 10, base_ts: float = 1000.0, interval: float = 0.1) -> dict:
    """Create benchmark data with n requests spaced by interval seconds."""
    return {
        "metadata": {
            "num_prefill_instances": 2,
            "num_decode_instances": 2,
            "total_instances": 4,
            "measured_qps": n / (n * interval) if n > 0 else 0,
        },
        "requests": [
            {
                "request_id": f"r{i}",
                "prompt_tokens": 100 + i * 10,
                "output_tokens": 50 + i * 5,
                "ttft_ms": 20.0,
                "tpot_ms": 5.0,
                "total_latency_ms": 270.0,
                "timestamp": base_ts + i * interval,
            }
            for i in range(n)
        ],
    }


# ---------- ReplayEntry ----------


class TestReplayEntry:
    def test_create(self) -> None:
        e = ReplayEntry(offset_ms=100.0, prompt_tokens=128, output_tokens=64)
        assert e.offset_ms == 100.0
        assert e.prompt_tokens == 128
        assert e.output_tokens == 64

    def test_offset_non_negative(self) -> None:
        with pytest.raises(Exception):
            ReplayEntry(offset_ms=-1.0, prompt_tokens=0, output_tokens=0)


# ---------- ReplayConfig ----------


class TestReplayConfig:
    def test_defaults(self) -> None:
        c = ReplayConfig()
        assert c.time_scale == 1.0
        assert c.target_qps is None

    def test_time_scale_positive(self) -> None:
        with pytest.raises(Exception):
            ReplayConfig(time_scale=0)

    def test_target_qps_positive(self) -> None:
        with pytest.raises(Exception):
            ReplayConfig(target_qps=0)


# ---------- ReplayGenerator basic ----------


class TestReplayGeneratorBasic:
    def test_generate_preserves_order(self) -> None:
        data = _make_benchmark_data(5)
        gen = ReplayGenerator()
        schedule = gen.generate(data)
        assert schedule.request_count == 5
        offsets = [e.offset_ms for e in schedule.entries]
        assert offsets == sorted(offsets)

    def test_first_offset_is_zero(self) -> None:
        data = _make_benchmark_data(3)
        gen = ReplayGenerator()
        schedule = gen.generate(data)
        assert schedule.entries[0].offset_ms == 0.0

    def test_relative_timestamps(self) -> None:
        data = _make_benchmark_data(3, base_ts=5000.0, interval=0.5)
        gen = ReplayGenerator()
        schedule = gen.generate(data)
        assert schedule.entries[0].offset_ms == pytest.approx(0.0)
        assert schedule.entries[1].offset_ms == pytest.approx(500.0)
        assert schedule.entries[2].offset_ms == pytest.approx(1000.0)

    def test_token_counts_preserved(self) -> None:
        data = _make_benchmark_data(3)
        gen = ReplayGenerator()
        schedule = gen.generate(data)
        assert schedule.entries[0].prompt_tokens == 100
        assert schedule.entries[0].output_tokens == 50
        assert schedule.entries[2].prompt_tokens == 120
        assert schedule.entries[2].output_tokens == 60

    def test_empty_requests(self) -> None:
        data = {"metadata": {}, "requests": []}
        gen = ReplayGenerator()
        schedule = gen.generate(data)
        assert schedule.request_count == 0
        assert schedule.total_duration_ms == 0.0
        assert schedule.effective_qps == 0.0

    def test_single_request(self) -> None:
        data = _make_benchmark_data(1)
        gen = ReplayGenerator()
        schedule = gen.generate(data)
        assert schedule.request_count == 1
        assert schedule.entries[0].offset_ms == 0.0
        assert schedule.total_duration_ms == 0.0


# ---------- Time scaling ----------


class TestTimeScaling:
    def test_scale_2x_faster(self) -> None:
        data = _make_benchmark_data(5, interval=1.0)
        config = ReplayConfig(time_scale=2.0)
        gen = ReplayGenerator(config)
        schedule = gen.generate(data)
        # Original: 0, 1000, 2000, 3000, 4000 ms
        # Scaled 2x: 0, 500, 1000, 1500, 2000 ms
        assert schedule.entries[1].offset_ms == pytest.approx(500.0)
        assert schedule.entries[4].offset_ms == pytest.approx(2000.0)

    def test_scale_half_speed(self) -> None:
        data = _make_benchmark_data(3, interval=0.1)
        config = ReplayConfig(time_scale=0.5)
        gen = ReplayGenerator(config)
        schedule = gen.generate(data)
        # Original: 0, 100, 200 ms → Scaled 0.5x: 0, 200, 400 ms
        assert schedule.entries[1].offset_ms == pytest.approx(200.0)
        assert schedule.entries[2].offset_ms == pytest.approx(400.0)


# ---------- Target QPS ----------


class TestTargetQPS:
    def test_uniform_distribution(self) -> None:
        data = _make_benchmark_data(5)
        config = ReplayConfig(target_qps=10.0)
        gen = ReplayGenerator(config)
        schedule = gen.generate(data)
        # 10 QPS → 100ms interval
        assert schedule.entries[0].offset_ms == pytest.approx(0.0)
        assert schedule.entries[1].offset_ms == pytest.approx(100.0)
        assert schedule.entries[4].offset_ms == pytest.approx(400.0)

    def test_target_qps_preserves_tokens(self) -> None:
        data = _make_benchmark_data(3)
        config = ReplayConfig(target_qps=5.0)
        gen = ReplayGenerator(config)
        schedule = gen.generate(data)
        assert schedule.entries[0].prompt_tokens == 100
        assert schedule.entries[2].prompt_tokens == 120

    def test_target_qps_with_time_scale(self) -> None:
        data = _make_benchmark_data(4)
        config = ReplayConfig(target_qps=10.0, time_scale=2.0)
        gen = ReplayGenerator(config)
        schedule = gen.generate(data)
        # target_qps=10 → 100ms intervals, then scale 2x → 50ms intervals
        assert schedule.entries[1].offset_ms == pytest.approx(50.0)


# ---------- ReplaySchedule fields ----------


class TestReplaySchedule:
    def test_effective_qps(self) -> None:
        data = _make_benchmark_data(10, interval=0.1)
        gen = ReplayGenerator()
        schedule = gen.generate(data)
        # 10 requests over 900ms = ~11.1 QPS
        assert schedule.effective_qps > 0
        assert schedule.total_duration_ms == pytest.approx(900.0)

    def test_config_stored(self) -> None:
        config = ReplayConfig(time_scale=3.0, target_qps=20.0)
        gen = ReplayGenerator(config)
        schedule = gen.generate(_make_benchmark_data(5))
        assert schedule.config.time_scale == 3.0
        assert schedule.config.target_qps == 20.0


# ---------- Error handling ----------


class TestErrors:
    def test_missing_requests_field(self) -> None:
        with pytest.raises(ValueError, match="requests"):
            ReplayGenerator().generate({"foo": "bar"})

    def test_missing_timestamp(self) -> None:
        data = {"requests": [{"prompt_tokens": 100, "output_tokens": 50}]}
        with pytest.raises(ValueError, match="timestamp"):
            ReplayGenerator().generate(data)


# ---------- File-based ----------


class TestFileGeneration:
    def test_generate_from_file(self, tmp_path: Path) -> None:
        f = tmp_path / "bench.json"
        f.write_text(json.dumps(_make_benchmark_data(5)))
        gen = ReplayGenerator()
        schedule = gen.generate_from_file(f)
        assert schedule.request_count == 5

    def test_generate_from_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ReplayGenerator().generate_from_file("/nonexistent/file.json")


# ---------- Programmatic API ----------


class TestGenerateReplay:
    def test_api_basic(self, tmp_path: Path) -> None:
        f = tmp_path / "bench.json"
        f.write_text(json.dumps(_make_benchmark_data(5)))
        schedule = generate_replay(str(f))
        assert schedule.request_count == 5

    def test_api_with_options(self, tmp_path: Path) -> None:
        f = tmp_path / "bench.json"
        f.write_text(json.dumps(_make_benchmark_data(5)))
        schedule = generate_replay(str(f), time_scale=2.0, target_qps=10.0)
        assert schedule.config.time_scale == 2.0
        assert schedule.config.target_qps == 10.0


# ---------- xpyd-bench format ----------


class TestXpydBenchFormat:
    def test_results_field(self) -> None:
        data = {
            "bench_config": {},
            "results": [
                {
                    "prompt_tokens": 100,
                    "output_tokens": 50,
                    "timestamp": 1000.0,
                },
                {
                    "prompt_tokens": 200,
                    "output_tokens": 100,
                    "timestamp": 1001.0,
                },
            ],
        }
        gen = ReplayGenerator()
        schedule = gen.generate(data)
        assert schedule.request_count == 2
        assert schedule.entries[0].prompt_tokens == 100
        assert schedule.entries[1].offset_ms == pytest.approx(1000.0)

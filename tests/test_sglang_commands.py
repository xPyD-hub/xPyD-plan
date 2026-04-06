"""Tests for SGLang Benchmark Command Generator (M111)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from xpyd_plan.cli import main
from xpyd_plan.sglang_commands import (
    SGLangCommandConfig,
    SGLangCommandGenerator,
    SGLangCommandSet,
    generate_sglang_commands,
)

# ── Config validation ──────────────────────────────────────────────


def test_config_valid():
    cfg = SGLangCommandConfig(
        model="meta-llama/Llama-3-8B", total_instances=4, qps_levels=[1, 2]
    )
    assert cfg.total_instances == 4
    assert cfg.port == 30000


def test_config_min_instances():
    with pytest.raises(Exception):
        SGLangCommandConfig(
            model="m", total_instances=1, qps_levels=[1]
        )


def test_config_empty_qps():
    with pytest.raises(Exception):
        SGLangCommandConfig(
            model="m", total_instances=4, qps_levels=[]
        )


# ── Generator basics ──────────────────────────────────────────────


def _make_gen(total: int = 4, qps: list[float] | None = None) -> SGLangCommandGenerator:
    cfg = SGLangCommandConfig(
        model="meta-llama/Llama-3-8B",
        total_instances=total,
        qps_levels=qps or [1.0],
    )
    return SGLangCommandGenerator(cfg)


def test_generate_count():
    result = _make_gen(4).generate()
    # 1P:3D, 2P:2D, 3P:1D
    assert len(result) == 3


def test_generate_ratios():
    result = _make_gen(4).generate()
    ratios = [cs.server.ratio for cs in result]
    assert ratios == ["1P:3D", "2P:2D", "3P:1D"]


def test_generate_min_instances():
    result = _make_gen(2).generate()
    assert len(result) == 1
    assert result[0].server.ratio == "1P:1D"


def test_server_command_contains_model():
    result = _make_gen(3).generate()
    assert "meta-llama/Llama-3-8B" in result[0].server.command


def test_server_command_contains_sglang():
    result = _make_gen(3).generate()
    assert "sglang.launch_server" in result[0].server.command


def test_benchmark_command_contains_sglang():
    result = _make_gen(3).generate()
    assert "sglang.bench_serving" in result[0].benchmarks[0].command


def test_benchmark_command_qps():
    result = _make_gen(3, qps=[2.5]).generate()
    assert "--request-rate 2.5" in result[0].benchmarks[0].command


def test_multiple_qps_levels():
    result = _make_gen(3, qps=[1.0, 2.0, 4.0]).generate()
    assert len(result[0].benchmarks) == 3


def test_server_command_prefill_decode():
    result = _make_gen(5).generate()
    first = result[0]
    assert first.server.prefill_instances == 1
    assert first.server.decode_instances == 4


# ── Optional config flags ──────────────────────────────────────────


def test_chunked_prefill_flag():
    cfg = SGLangCommandConfig(
        model="m", total_instances=3, qps_levels=[1], chunked_prefill=True
    )
    gen = SGLangCommandGenerator(cfg)
    result = gen.generate()
    assert "--chunked-prefill-size" in result[0].server.command


def test_max_model_len():
    cfg = SGLangCommandConfig(
        model="m", total_instances=3, qps_levels=[1], max_model_len=4096
    )
    gen = SGLangCommandGenerator(cfg)
    result = gen.generate()
    assert "--context-length 4096" in result[0].server.command


def test_dp_size():
    cfg = SGLangCommandConfig(
        model="m", total_instances=3, qps_levels=[1], dp_size=2
    )
    gen = SGLangCommandGenerator(cfg)
    result = gen.generate()
    assert "--dp 2" in result[0].server.command


def test_dataset_flag():
    cfg = SGLangCommandConfig(
        model="m", total_instances=3, qps_levels=[1], dataset="/data/sharegpt.json"
    )
    gen = SGLangCommandGenerator(cfg)
    result = gen.generate()
    assert "--dataset-path /data/sharegpt.json" in result[0].benchmarks[0].command


def test_custom_port():
    cfg = SGLangCommandConfig(
        model="m", total_instances=3, qps_levels=[1], port=8080
    )
    gen = SGLangCommandGenerator(cfg)
    result = gen.generate()
    assert "--port 8080" in result[0].server.command
    assert "--port 8080" in result[0].benchmarks[0].command


# ── Script snippet ──────────────────────────────────────────────


def test_script_snippet_has_kill():
    result = _make_gen(3).generate()
    assert "kill $SERVER_PID" in result[0].script_snippet


def test_script_snippet_has_sleep():
    result = _make_gen(3).generate()
    assert "sleep 120" in result[0].script_snippet


# ── Programmatic API ──────────────────────────────────────────────


def test_programmatic_api():
    result = generate_sglang_commands(
        model="m", total_instances=4, qps_levels=[1.0, 2.0]
    )
    assert len(result) == 3
    assert all(isinstance(cs, SGLangCommandSet) for cs in result)


def test_programmatic_api_options():
    result = generate_sglang_commands(
        model="m",
        total_instances=3,
        qps_levels=[1.0],
        tp_size=2,
        dp_size=4,
        chunked_prefill=True,
        port=9000,
    )
    assert "--tp 2" in result[0].server.command
    assert "--dp 4" in result[0].server.command


# ── CLI integration ──────────────────────────────────────────────


def test_cli_table_output(capsys):
    with patch(
        "sys.argv",
        [
            "xpyd-plan",
            "sglang-commands",
            "--model",
            "meta-llama/Llama-3-8B",
            "--total-instances",
            "4",
            "--qps",
            "1,2",
        ],
    ):
        main()
    captured = capsys.readouterr()
    assert "SGLang Benchmark Commands" in captured.out


def test_cli_json_output(capsys):
    with patch(
        "sys.argv",
        [
            "xpyd-plan",
            "sglang-commands",
            "--model",
            "m",
            "--total-instances",
            "3",
            "--qps",
            "1",
            "--output-format",
            "json",
        ],
    ):
        main()
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert len(data) == 2


def test_cli_output_script():
    with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
        path = f.name

    with patch(
        "sys.argv",
        [
            "xpyd-plan",
            "sglang-commands",
            "--model",
            "m",
            "--total-instances",
            "3",
            "--qps",
            "1",
            "--output-script",
            path,
        ],
    ):
        main()
    content = Path(path).read_text()
    assert "#!/usr/bin/env bash" in content
    assert "sglang" in content
    Path(path).unlink()


# ── Model serialization ──────────────────────────────────────────


def test_model_dump():
    result = _make_gen(3).generate()
    d = result[0].model_dump()
    assert "server" in d
    assert "benchmarks" in d
    assert "script_snippet" in d

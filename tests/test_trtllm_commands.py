"""Tests for TensorRT-LLM Benchmark Command Generator (M113)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from xpyd_plan.trtllm_commands import (
    TRTLLMBenchmarkCommand,
    TRTLLMCommandConfig,
    TRTLLMCommandGenerator,
    TRTLLMCommandSet,
    TRTLLMServerCommand,
    generate_trtllm_commands,
)

# --- Config model tests ---


class TestTRTLLMCommandConfig:
    def test_valid_config(self):
        cfg = TRTLLMCommandConfig(
            model="meta-llama/Llama-2-7b",
            total_instances=4,
            qps_levels=[1.0, 2.0],
        )
        assert cfg.model == "meta-llama/Llama-2-7b"
        assert cfg.total_instances == 4
        assert cfg.qps_levels == [1.0, 2.0]

    def test_defaults(self):
        cfg = TRTLLMCommandConfig(
            model="m", total_instances=2, qps_levels=[1.0]
        )
        assert cfg.tp_size == 1
        assert cfg.pp_size == 1
        assert cfg.max_batch_size == 256
        assert cfg.max_input_len == 2048
        assert cfg.max_output_len == 2048
        assert cfg.kv_cache_free_gpu_mem_fraction == 0.9
        assert cfg.dtype == "float16"
        assert cfg.host == "localhost"
        assert cfg.port == 8000
        assert cfg.engine_dir == "./engines"
        assert cfg.num_prompts == 1000

    def test_invalid_total_instances(self):
        with pytest.raises(Exception):
            TRTLLMCommandConfig(
                model="m", total_instances=1, qps_levels=[1.0]
            )

    def test_empty_model(self):
        with pytest.raises(Exception):
            TRTLLMCommandConfig(
                model="", total_instances=2, qps_levels=[1.0]
            )

    def test_empty_qps(self):
        with pytest.raises(Exception):
            TRTLLMCommandConfig(
                model="m", total_instances=2, qps_levels=[]
            )

    def test_invalid_kv_cache_fraction(self):
        with pytest.raises(Exception):
            TRTLLMCommandConfig(
                model="m",
                total_instances=2,
                qps_levels=[1.0],
                kv_cache_free_gpu_mem_fraction=0.0,
            )


# --- Generator tests ---


class TestTRTLLMCommandGenerator:
    def _make_config(self, **kwargs):
        defaults = dict(
            model="meta-llama/Llama-2-7b",
            total_instances=4,
            qps_levels=[1.0, 2.0],
        )
        defaults.update(kwargs)
        return TRTLLMCommandConfig(**defaults)

    def test_generates_correct_ratio_count(self):
        gen = TRTLLMCommandGenerator(self._make_config(total_instances=4))
        result = gen.generate()
        assert len(result) == 3  # 1P:3D, 2P:2D, 3P:1D

    def test_ratio_strings(self):
        gen = TRTLLMCommandGenerator(self._make_config(total_instances=3))
        result = gen.generate()
        ratios = [cs.server.ratio for cs in result]
        assert ratios == ["1P:2D", "2P:1D"]

    def test_server_command_contains_model(self):
        gen = TRTLLMCommandGenerator(self._make_config())
        result = gen.generate()
        assert "meta-llama/Llama-2-7b" in result[0].server.engine_build_command

    def test_server_command_contains_engine_dir(self):
        gen = TRTLLMCommandGenerator(self._make_config(engine_dir="/my/engines"))
        result = gen.generate()
        assert "/my/engines" in result[0].server.engine_build_command
        assert "/my/engines" in result[0].server.server_command

    def test_benchmark_commands_per_qps(self):
        gen = TRTLLMCommandGenerator(
            self._make_config(qps_levels=[1.0, 2.0, 4.0])
        )
        result = gen.generate()
        for cs in result:
            assert len(cs.benchmarks) == 3

    def test_benchmark_command_contains_qps(self):
        gen = TRTLLMCommandGenerator(self._make_config(qps_levels=[5.5]))
        result = gen.generate()
        assert "5.5" in result[0].benchmarks[0].command

    def test_benchmark_output_file(self):
        gen = TRTLLMCommandGenerator(self._make_config(qps_levels=[1.0]))
        result = gen.generate()
        assert "bench_1P_3D_qps1.0.json" in result[0].benchmarks[0].command

    def test_dataset_in_command(self):
        gen = TRTLLMCommandGenerator(self._make_config(dataset="/data/test.json"))
        result = gen.generate()
        assert "/data/test.json" in result[0].benchmarks[0].command

    def test_no_dataset_by_default(self):
        gen = TRTLLMCommandGenerator(self._make_config())
        result = gen.generate()
        assert "--dataset" not in result[0].benchmarks[0].command

    def test_script_snippet_contains_engine_build(self):
        gen = TRTLLMCommandGenerator(self._make_config())
        result = gen.generate()
        assert "trtllm-build" in result[0].script_snippet

    def test_script_snippet_contains_kill(self):
        gen = TRTLLMCommandGenerator(self._make_config())
        result = gen.generate()
        assert "kill $SERVER_PID" in result[0].script_snippet

    def test_custom_tp_pp(self):
        gen = TRTLLMCommandGenerator(self._make_config(tp_size=2, pp_size=4))
        result = gen.generate()
        assert "--tp_size 2" in result[0].server.engine_build_command
        assert "--pp_size 4" in result[0].server.engine_build_command

    def test_custom_dtype(self):
        gen = TRTLLMCommandGenerator(self._make_config(dtype="bfloat16"))
        result = gen.generate()
        assert "--dtype bfloat16" in result[0].server.engine_build_command

    def test_kv_cache_fraction_in_server(self):
        gen = TRTLLMCommandGenerator(
            self._make_config(kv_cache_free_gpu_mem_fraction=0.85)
        )
        result = gen.generate()
        assert "0.85" in result[0].server.server_command

    def test_two_instances_gives_one_ratio(self):
        gen = TRTLLMCommandGenerator(self._make_config(total_instances=2))
        result = gen.generate()
        assert len(result) == 1
        assert result[0].server.ratio == "1P:1D"


# --- Programmatic API tests ---


class TestGenerateTRTLLMCommands:
    def test_basic_api(self):
        result = generate_trtllm_commands(
            model="test-model",
            total_instances=3,
            qps_levels=[1.0],
        )
        assert len(result) == 2
        assert isinstance(result[0], TRTLLMCommandSet)

    def test_api_with_options(self):
        result = generate_trtllm_commands(
            model="test-model",
            total_instances=4,
            qps_levels=[1.0, 2.0],
            tp_size=2,
            pp_size=2,
            max_batch_size=128,
            dtype="bfloat16",
            engine_dir="/tmp/engines",
        )
        assert len(result) == 3
        assert "--tp_size 2" in result[0].server.engine_build_command
        assert "--max_batch_size 128" in result[0].server.engine_build_command


# --- Model tests ---


class TestModels:
    def test_server_command_fields(self):
        cmd = TRTLLMServerCommand(
            ratio="1P:2D",
            prefill_instances=1,
            decode_instances=2,
            engine_build_command="trtllm-build --model test",
            server_command="python3 -m tensorrt_llm.serve --engine_dir test",
        )
        assert cmd.ratio == "1P:2D"
        assert cmd.prefill_instances == 1
        assert cmd.decode_instances == 2

    def test_benchmark_command_fields(self):
        cmd = TRTLLMBenchmarkCommand(
            ratio="1P:2D", qps=5.0, command="trtllm-bench --qps 5"
        )
        assert cmd.qps == 5.0

    def test_command_set_fields(self):
        server = TRTLLMServerCommand(
            ratio="1P:1D",
            prefill_instances=1,
            decode_instances=1,
            engine_build_command="build",
            server_command="serve",
        )
        cs = TRTLLMCommandSet(
            server=server, benchmarks=[], script_snippet="echo hi"
        )
        assert cs.script_snippet == "echo hi"


# --- CLI integration test ---


class TestCLIIntegration:
    def test_trtllm_commands_json(self):
        """Test CLI trtllm-commands subcommand with JSON output."""

        result = subprocess.run(
            [


                "xpyd-plan",
                "trtllm-commands",
                "--model",
                "test-model",
                "--total-instances",
                "3",
                "--qps",
                "1.0,2.0",
                "--output-format",
                "json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data) == 2  # 1P:2D, 2P:1D

    def test_trtllm_commands_table(self):

        result = subprocess.run(
            [


                "xpyd-plan",
                "trtllm-commands",
                "--model",
                "test-model",
                "--total-instances",
                "3",
                "--qps",
                "1.0",
                "--output-format",
                "table",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "P:D Ratio" in result.stdout

    def test_trtllm_commands_output_script(self):

        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
            script_path = f.name

        result = subprocess.run(
            [


                "xpyd-plan",
                "trtllm-commands",
                "--model",
                "test-model",
                "--total-instances",
                "3",
                "--qps",
                "1.0",
                "--output-script",
                script_path,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        content = Path(script_path).read_text()
        assert "#!/usr/bin/env bash" in content
        assert "trtllm-build" in content
        assert "tensorrt_llm.serve" in content
        Path(script_path).unlink()

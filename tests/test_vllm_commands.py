"""Tests for vllm_commands module."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.cli import main
from xpyd_plan.vllm_commands import (
    BenchmarkCommand,
    CommandConfig,
    CommandGenerator,
    CommandSet,
    ServerCommand,
    generate_vllm_commands,
)


class TestCommandConfig:
    """Tests for CommandConfig model."""

    def test_minimal_config(self):
        cfg = CommandConfig(model="meta-llama/Llama-3-8B", total_instances=4, qps_levels=[1.0])
        assert cfg.model == "meta-llama/Llama-3-8B"
        assert cfg.tp_size == 1
        assert cfg.host == "localhost"
        assert cfg.port == 8000
        assert cfg.num_prompts == 1000

    def test_full_config(self):
        cfg = CommandConfig(
            model="my-model",
            total_instances=8,
            qps_levels=[1.0, 2.0, 4.0],
            tp_size=2,
            max_model_len=4096,
            dataset="/data/sharegpt.json",
            num_prompts=500,
            host="0.0.0.0",
            port=9000,
        )
        assert cfg.max_model_len == 4096
        assert cfg.dataset == "/data/sharegpt.json"

    def test_invalid_total_instances(self):
        with pytest.raises(Exception):
            CommandConfig(model="m", total_instances=1, qps_levels=[1.0])

    def test_empty_qps_levels(self):
        with pytest.raises(Exception):
            CommandConfig(model="m", total_instances=4, qps_levels=[])

    def test_serializable(self):
        cfg = CommandConfig(model="m", total_instances=4, qps_levels=[1.0, 2.0])
        data = cfg.model_dump()
        restored = CommandConfig.model_validate(data)
        assert restored.model == cfg.model


class TestCommandGenerator:
    """Tests for CommandGenerator."""

    def _default_config(self, **kwargs) -> CommandConfig:
        defaults = dict(model="meta-llama/Llama-3-8B", total_instances=4, qps_levels=[1.0, 2.0])
        defaults.update(kwargs)
        return CommandConfig(**defaults)

    def test_generates_correct_number_of_ratios(self):
        cfg = self._default_config(total_instances=4)
        result = CommandGenerator().generate(cfg)
        # total=4 -> 1P:3D, 2P:2D, 3P:1D = 3 ratios
        assert len(result) == 3

    def test_generates_correct_number_of_ratios_6(self):
        cfg = self._default_config(total_instances=6)
        result = CommandGenerator().generate(cfg)
        # 1P:5D, 2P:4D, 3P:3D, 4P:2D, 5P:1D = 5
        assert len(result) == 5

    def test_server_command_format(self):
        cfg = self._default_config(total_instances=4, tp_size=2)
        result = CommandGenerator().generate(cfg)
        cmd = result[0].server
        assert cmd.ratio == "1P:3D"
        assert cmd.prefill_instances == 1
        assert cmd.decode_instances == 3
        assert "vllm serve" in cmd.command
        assert "--tensor-parallel-size 2" in cmd.command
        assert "--disaggregated-prefill-instance 1" in cmd.command
        assert "--disaggregated-decode-instance 3" in cmd.command

    def test_server_command_with_max_model_len(self):
        cfg = self._default_config(max_model_len=4096)
        result = CommandGenerator().generate(cfg)
        assert "--max-model-len 4096" in result[0].server.command

    def test_server_command_without_max_model_len(self):
        cfg = self._default_config()
        result = CommandGenerator().generate(cfg)
        assert "--max-model-len" not in result[0].server.command

    def test_benchmark_command_format(self):
        cfg = self._default_config(qps_levels=[5.0])
        result = CommandGenerator().generate(cfg)
        bc = result[0].benchmarks[0]
        assert "benchmark_serving.py" in bc.command
        assert "--backend vllm" in bc.command
        assert "--request-rate 5.0" in bc.command
        assert "--num-prompts 1000" in bc.command
        assert "--base-url http://localhost:8000" in bc.command

    def test_benchmark_command_with_dataset(self):
        cfg = self._default_config(dataset="/data/test.json")
        result = CommandGenerator().generate(cfg)
        assert "--dataset-path /data/test.json" in result[0].benchmarks[0].command

    def test_benchmark_command_without_dataset(self):
        cfg = self._default_config()
        result = CommandGenerator().generate(cfg)
        assert "--dataset-path" not in result[0].benchmarks[0].command

    def test_cross_product_qps(self):
        cfg = self._default_config(total_instances=3, qps_levels=[1.0, 2.0, 4.0])
        result = CommandGenerator().generate(cfg)
        # total=3 -> 2 ratios, 3 QPS each = 6 benchmark commands total
        for cs in result:
            assert len(cs.benchmarks) == 3
        assert len(result) == 2

    def test_command_set_has_snippet(self):
        cfg = self._default_config()
        result = CommandGenerator().generate(cfg)
        for cs in result:
            assert cs.script_snippet
            assert "SERVER_PID" in cs.script_snippet
            assert "kill $SERVER_PID" in cs.script_snippet

    def test_shell_script_generation(self):
        cfg = self._default_config()
        gen = CommandGenerator()
        result = gen.generate(cfg)
        script = gen.to_shell_script(result)
        assert script.startswith("#!/usr/bin/env bash")
        assert "set -euo pipefail" in script
        for cs in result:
            assert cs.server.ratio in script

    def test_shell_script_empty(self):
        gen = CommandGenerator()
        script = gen.to_shell_script([])
        assert "#!/usr/bin/env bash" in script

    def test_total_instances_2(self):
        cfg = self._default_config(total_instances=2)
        result = CommandGenerator().generate(cfg)
        # Only 1P:1D
        assert len(result) == 1
        assert result[0].server.ratio == "1P:1D"


class TestConvenienceFunction:
    """Tests for generate_vllm_commands()."""

    def test_convenience_matches_class(self):
        cfg = CommandConfig(model="m", total_instances=4, qps_levels=[1.0])
        result = generate_vllm_commands(cfg)
        assert len(result) == 3
        assert isinstance(result[0], CommandSet)


class TestModels:
    """Tests for Pydantic models serialization."""

    def test_server_command_serializable(self):
        sc = ServerCommand(ratio="2P:2D", prefill_instances=2, decode_instances=2, command="test")
        data = sc.model_dump()
        restored = ServerCommand.model_validate(data)
        assert restored.ratio == "2P:2D"

    def test_command_set_json_roundtrip(self):
        cs = CommandSet(
            server=ServerCommand(
                ratio="1P:1D", prefill_instances=1, decode_instances=1, command="s",
            ),
            benchmarks=[BenchmarkCommand(ratio="1P:1D", qps=1.0, command="b")],
            script_snippet="echo test",
        )
        j = cs.model_dump_json()
        restored = CommandSet.model_validate_json(j)
        assert restored.server.ratio == "1P:1D"
        assert len(restored.benchmarks) == 1


class TestVLLMCommandsCLI:
    """Tests for the CLI vllm-commands subcommand."""

    def test_cli_json_output(self, capsys):
        main([
            "vllm-commands",
            "--model", "test-model",
            "--total-instances", "4",
            "--qps", "1,2",
            "--output-format", "json",
        ])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 3
        assert "server" in data[0]
        assert "benchmarks" in data[0]

    def test_cli_table_output(self, capsys):
        main([
            "vllm-commands",
            "--model", "test-model",
            "--total-instances", "4",
            "--qps", "1,2",
            "--output-format", "table",
        ])

    def test_cli_shell_script_output(self, tmp_path):
        out_path = tmp_path / "run.sh"
        main([
            "vllm-commands",
            "--model", "test-model",
            "--total-instances", "4",
            "--qps", "1,2",
            "--output-script", str(out_path),
        ])
        content = out_path.read_text()
        assert "#!/usr/bin/env bash" in content
        assert "vllm serve" in content

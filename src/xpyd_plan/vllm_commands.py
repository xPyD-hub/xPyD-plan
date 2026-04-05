"""vLLM Benchmark Command Generator — generate ready-to-run vLLM commands.

Given total instances, model name, and QPS levels, generate vLLM server
and benchmark_serving.py commands for each planned P:D ratio configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CommandConfig(BaseModel):
    """Configuration for command generation."""

    model: str = Field(..., min_length=1, description="HuggingFace model name")
    total_instances: int = Field(..., ge=2, description="Total instances (P+D)")
    qps_levels: list[float] = Field(
        ..., min_length=1, description="QPS levels to benchmark"
    )
    tp_size: int = Field(1, ge=1, description="Tensor parallel size")
    max_model_len: int | None = Field(None, ge=1, description="Max model length")
    dataset: str | None = Field(None, description="Dataset path")
    num_prompts: int = Field(1000, ge=1, description="Number of prompts per run")
    host: str = Field("localhost", description="Server host")
    port: int = Field(8000, ge=1, le=65535, description="Server port")


class ServerCommand(BaseModel):
    """A vLLM server launch command for one P:D ratio."""

    ratio: str = Field(..., description="e.g. '2P:3D'")
    prefill_instances: int = Field(..., ge=1)
    decode_instances: int = Field(..., ge=1)
    command: str = Field(..., description="Shell command")


class BenchmarkCommand(BaseModel):
    """A benchmark_serving.py invocation command."""

    ratio: str = Field(..., description="P:D ratio string")
    qps: float = Field(..., gt=0)
    command: str = Field(..., description="Shell command")


class CommandSet(BaseModel):
    """Complete command set for one P:D ratio."""

    server: ServerCommand
    benchmarks: list[BenchmarkCommand]
    script_snippet: str = Field("", description="Combined shell script snippet")


class CommandGenerator:
    """Generate vLLM serve and benchmark commands for P:D ratio exploration."""

    def generate(self, config: CommandConfig) -> list[CommandSet]:
        """Generate command sets for all valid P:D ratios."""
        total = config.total_instances
        results: list[CommandSet] = []

        for p in range(1, total):
            d = total - p
            if d < 1:
                break
            ratio_str = f"{p}P:{d}D"

            server_cmd = self._build_server_command(config, p, d, ratio_str)
            bench_cmds = [
                self._build_benchmark_command(config, ratio_str, qps)
                for qps in config.qps_levels
            ]
            snippet = self._build_snippet(server_cmd, bench_cmds)

            results.append(
                CommandSet(
                    server=server_cmd,
                    benchmarks=bench_cmds,
                    script_snippet=snippet,
                )
            )

        return results

    def _build_server_command(
        self,
        config: CommandConfig,
        prefill: int,
        decode: int,
        ratio_str: str,
    ) -> ServerCommand:
        parts = [
            f"vllm serve {config.model}",
            f"--tensor-parallel-size {config.tp_size}",
            f"--disaggregated-prefill-instance {prefill}",
            f"--disaggregated-decode-instance {decode}",
            f"--host {config.host}",
            f"--port {config.port}",
        ]
        if config.max_model_len is not None:
            parts.append(f"--max-model-len {config.max_model_len}")
        return ServerCommand(
            ratio=ratio_str,
            prefill_instances=prefill,
            decode_instances=decode,
            command=" ".join(parts),
        )

    def _build_benchmark_command(
        self,
        config: CommandConfig,
        ratio_str: str,
        qps: float,
    ) -> BenchmarkCommand:
        parts = [
            "python benchmark_serving.py",
            "--backend vllm",
            f"--model {config.model}",
            f"--base-url http://{config.host}:{config.port}",
            f"--request-rate {qps}",
            f"--num-prompts {config.num_prompts}",
        ]
        if config.dataset is not None:
            parts.append(f"--dataset-path {config.dataset}")
        return BenchmarkCommand(
            ratio=ratio_str,
            qps=qps,
            command=" ".join(parts),
        )

    def _build_snippet(
        self, server: ServerCommand, benchmarks: list[BenchmarkCommand]
    ) -> str:
        lines = [
            f"# --- {server.ratio} ---",
            f"{server.command} &",
            "SERVER_PID=$!",
            "sleep 30  # wait for server startup",
            "",
        ]
        for bc in benchmarks:
            lines.append(bc.command)
            lines.append("")
        lines.append("kill $SERVER_PID")
        lines.append("wait $SERVER_PID 2>/dev/null || true")
        lines.append("")
        return "\n".join(lines)

    def to_shell_script(self, command_sets: list[CommandSet]) -> str:
        """Combine all command sets into a single shell script."""
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
        ]
        for cs in command_sets:
            lines.append(cs.script_snippet)
        return "\n".join(lines)


def generate_vllm_commands(config: CommandConfig) -> list[CommandSet]:
    """Convenience function — generate vLLM commands for all P:D ratios."""
    return CommandGenerator().generate(config)

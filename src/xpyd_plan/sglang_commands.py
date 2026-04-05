"""SGLang Benchmark Command Generator — generate ready-to-run SGLang commands.

Given total instances, model name, and QPS levels, generate SGLang server
and bench_serving commands for each planned P:D ratio configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SGLangCommandConfig(BaseModel):
    """Configuration for SGLang command generation."""

    model: str = Field(..., min_length=1, description="HuggingFace model name")
    total_instances: int = Field(..., ge=2, description="Total instances (P+D)")
    qps_levels: list[float] = Field(
        ..., min_length=1, description="QPS levels to benchmark"
    )
    tp_size: int = Field(1, ge=1, description="Tensor parallel size")
    dp_size: int = Field(1, ge=1, description="Data parallel size")
    max_model_len: int | None = Field(None, ge=1, description="Max model length")
    chunked_prefill: bool = Field(False, description="Enable chunked prefill")
    dataset: str | None = Field(None, description="Dataset path")
    num_prompts: int = Field(1000, ge=1, description="Number of prompts per run")
    host: str = Field("localhost", description="Server host")
    port: int = Field(30000, ge=1, le=65535, description="Server port")


class SGLangServerCommand(BaseModel):
    """An SGLang server launch command for one P:D ratio."""

    ratio: str = Field(..., description="e.g. '2P:3D'")
    prefill_instances: int = Field(..., ge=1)
    decode_instances: int = Field(..., ge=1)
    command: str = Field(..., description="Shell command")


class SGLangBenchmarkCommand(BaseModel):
    """A bench_serving invocation command."""

    ratio: str = Field(..., description="P:D ratio string")
    qps: float = Field(..., gt=0)
    command: str = Field(..., description="Shell command")


class SGLangCommandSet(BaseModel):
    """Complete command set for one P:D ratio."""

    server: SGLangServerCommand
    benchmarks: list[SGLangBenchmarkCommand]
    script_snippet: str = Field("", description="Combined shell script snippet")


class SGLangCommandGenerator:
    """Generate SGLang server and benchmark commands for P:D ratio exploration."""

    def __init__(self, config: SGLangCommandConfig) -> None:
        self._config = config

    def generate(self) -> list[SGLangCommandSet]:
        """Generate command sets for all valid P:D ratios."""
        total = self._config.total_instances
        results: list[SGLangCommandSet] = []

        for p in range(1, total):
            d = total - p
            if d < 1:
                continue
            ratio_str = f"{p}P:{d}D"
            server_cmd = self._build_server_command(p, d, ratio_str)
            bench_cmds = [
                self._build_benchmark_command(ratio_str, qps)
                for qps in self._config.qps_levels
            ]
            snippet = self._build_script_snippet(server_cmd, bench_cmds)
            results.append(
                SGLangCommandSet(
                    server=server_cmd,
                    benchmarks=bench_cmds,
                    script_snippet=snippet,
                )
            )

        return results

    def _build_server_command(
        self, prefill: int, decode: int, ratio: str
    ) -> SGLangServerCommand:
        cfg = self._config
        parts = [
            "python3 -m sglang.launch_server",
            f"--model-path {cfg.model}",
            f"--tp {cfg.tp_size}",
            f"--dp {cfg.dp_size}",
            f"--host {cfg.host}",
            f"--port {cfg.port}",
        ]
        if cfg.max_model_len is not None:
            parts.append(f"--context-length {cfg.max_model_len}")
        if cfg.chunked_prefill:
            parts.append("--chunked-prefill-size 8192")
        return SGLangServerCommand(
            ratio=ratio,
            prefill_instances=prefill,
            decode_instances=decode,
            command=" \\\n  ".join(parts),
        )

    def _build_benchmark_command(
        self, ratio: str, qps: float
    ) -> SGLangBenchmarkCommand:
        cfg = self._config
        parts = [
            "python3 -m sglang.bench_serving",
            "--backend sglang",
            f"--host {cfg.host}",
            f"--port {cfg.port}",
            f"--num-prompts {cfg.num_prompts}",
            f"--request-rate {qps}",
        ]
        if cfg.dataset is not None:
            parts.append(f"--dataset-path {cfg.dataset}")
        output_file = f"bench_{ratio.replace(':', '_')}_qps{qps}.json"
        parts.append(f"--output-file {output_file}")
        return SGLangBenchmarkCommand(
            ratio=ratio,
            qps=qps,
            command=" \\\n  ".join(parts),
        )

    def _build_script_snippet(
        self,
        server: SGLangServerCommand,
        benchmarks: list[SGLangBenchmarkCommand],
    ) -> str:
        lines = [
            f"# --- {server.ratio} ---",
            f"{server.command} &",
            "SERVER_PID=$!",
            "sleep 120  # wait for SGLang server to load model",
            "",
        ]
        for bench in benchmarks:
            lines.append(f"{bench.command}")
            lines.append("")
        lines.append("kill $SERVER_PID")
        lines.append(f"echo 'Done with {server.ratio}'")
        lines.append("")
        return "\n".join(lines)


def generate_sglang_commands(
    model: str,
    total_instances: int,
    qps_levels: list[float],
    *,
    tp_size: int = 1,
    dp_size: int = 1,
    max_model_len: int | None = None,
    chunked_prefill: bool = False,
    dataset: str | None = None,
    num_prompts: int = 1000,
    host: str = "localhost",
    port: int = 30000,
) -> list[SGLangCommandSet]:
    """Programmatic API: generate SGLang benchmark commands."""
    config = SGLangCommandConfig(
        model=model,
        total_instances=total_instances,
        qps_levels=qps_levels,
        tp_size=tp_size,
        dp_size=dp_size,
        max_model_len=max_model_len,
        chunked_prefill=chunked_prefill,
        dataset=dataset,
        num_prompts=num_prompts,
        host=host,
        port=port,
    )
    return SGLangCommandGenerator(config).generate()

"""TensorRT-LLM Benchmark Command Generator — generate ready-to-run TRT-LLM commands.

Given total instances, model name, and QPS levels, generate TensorRT-LLM
engine build and trtllm-bench commands for each planned P:D ratio configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TRTLLMCommandConfig(BaseModel):
    """Configuration for TensorRT-LLM command generation."""

    model: str = Field(..., min_length=1, description="HuggingFace model name")
    total_instances: int = Field(..., ge=2, description="Total instances (P+D)")
    qps_levels: list[float] = Field(
        ..., min_length=1, description="QPS levels to benchmark"
    )
    tp_size: int = Field(1, ge=1, description="Tensor parallel size")
    pp_size: int = Field(1, ge=1, description="Pipeline parallel size")
    max_batch_size: int = Field(256, ge=1, description="Max batch size")
    max_input_len: int = Field(2048, ge=1, description="Max input sequence length")
    max_output_len: int = Field(2048, ge=1, description="Max output sequence length")
    kv_cache_free_gpu_mem_fraction: float = Field(
        0.9, gt=0.0, le=1.0, description="KV cache GPU memory fraction"
    )
    dtype: str = Field("float16", description="Data type (float16, bfloat16, float32)")
    dataset: str | None = Field(None, description="Dataset path")
    num_prompts: int = Field(1000, ge=1, description="Number of prompts per run")
    host: str = Field("localhost", description="Server host")
    port: int = Field(8000, ge=1, le=65535, description="Server port")
    engine_dir: str = Field("./engines", description="Engine output directory")


class TRTLLMServerCommand(BaseModel):
    """A TensorRT-LLM server launch command for one P:D ratio."""

    ratio: str = Field(..., description="e.g. '2P:3D'")
    prefill_instances: int = Field(..., ge=1)
    decode_instances: int = Field(..., ge=1)
    engine_build_command: str = Field(..., description="Engine build command")
    server_command: str = Field(..., description="Server launch command")


class TRTLLMBenchmarkCommand(BaseModel):
    """A trtllm-bench invocation command."""

    ratio: str = Field(..., description="P:D ratio string")
    qps: float = Field(..., gt=0)
    command: str = Field(..., description="Shell command")


class TRTLLMCommandSet(BaseModel):
    """Complete command set for one P:D ratio."""

    server: TRTLLMServerCommand
    benchmarks: list[TRTLLMBenchmarkCommand]
    script_snippet: str = Field("", description="Combined shell script snippet")


class TRTLLMCommandGenerator:
    """Generate TensorRT-LLM engine build, server, and benchmark commands."""

    def __init__(self, config: TRTLLMCommandConfig) -> None:
        self._config = config

    def generate(self) -> list[TRTLLMCommandSet]:
        """Generate command sets for all valid P:D ratios."""
        total = self._config.total_instances
        results: list[TRTLLMCommandSet] = []

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
                TRTLLMCommandSet(
                    server=server_cmd,
                    benchmarks=bench_cmds,
                    script_snippet=snippet,
                )
            )

        return results

    def _build_server_command(
        self, prefill: int, decode: int, ratio: str
    ) -> TRTLLMServerCommand:
        cfg = self._config
        engine_path = f"{cfg.engine_dir}/{ratio.replace(':', '_')}"

        build_parts = [
            "trtllm-build",
            f"--model_dir {cfg.model}",
            f"--output_dir {engine_path}",
            f"--tp_size {cfg.tp_size}",
            f"--pp_size {cfg.pp_size}",
            f"--max_batch_size {cfg.max_batch_size}",
            f"--max_input_len {cfg.max_input_len}",
            f"--max_output_len {cfg.max_output_len}",
            f"--dtype {cfg.dtype}",
        ]

        server_parts = [
            "python3 -m tensorrt_llm.serve",
            f"--engine_dir {engine_path}",
            f"--host {cfg.host}",
            f"--port {cfg.port}",
            f"--kv_cache_free_gpu_mem_fraction {cfg.kv_cache_free_gpu_mem_fraction}",
        ]

        return TRTLLMServerCommand(
            ratio=ratio,
            prefill_instances=prefill,
            decode_instances=decode,
            engine_build_command=" \\\n  ".join(build_parts),
            server_command=" \\\n  ".join(server_parts),
        )

    def _build_benchmark_command(
        self, ratio: str, qps: float
    ) -> TRTLLMBenchmarkCommand:
        cfg = self._config
        parts = [
            "trtllm-bench",
            f"--host {cfg.host}",
            f"--port {cfg.port}",
            f"--num-prompts {cfg.num_prompts}",
            f"--request-rate {qps}",
        ]
        if cfg.dataset is not None:
            parts.append(f"--dataset {cfg.dataset}")
        output_file = f"bench_{ratio.replace(':', '_')}_qps{qps}.json"
        parts.append(f"--output-file {output_file}")
        return TRTLLMBenchmarkCommand(
            ratio=ratio,
            qps=qps,
            command=" \\\n  ".join(parts),
        )

    def _build_script_snippet(
        self,
        server: TRTLLMServerCommand,
        benchmarks: list[TRTLLMBenchmarkCommand],
    ) -> str:
        lines = [
            f"# --- {server.ratio} ---",
            f"echo 'Building engine for {server.ratio}...'",
            f"{server.engine_build_command}",
            "",
            f"{server.server_command} &",
            "SERVER_PID=$!",
            "sleep 60  # wait for TRT-LLM server to load engine",
            "",
        ]
        for bench in benchmarks:
            lines.append(f"{bench.command}")
            lines.append("")
        lines.append("kill $SERVER_PID")
        lines.append(f"echo 'Done with {server.ratio}'")
        lines.append("")
        return "\n".join(lines)


def generate_trtllm_commands(
    model: str,
    total_instances: int,
    qps_levels: list[float],
    *,
    tp_size: int = 1,
    pp_size: int = 1,
    max_batch_size: int = 256,
    max_input_len: int = 2048,
    max_output_len: int = 2048,
    kv_cache_free_gpu_mem_fraction: float = 0.9,
    dtype: str = "float16",
    dataset: str | None = None,
    num_prompts: int = 1000,
    host: str = "localhost",
    port: int = 8000,
    engine_dir: str = "./engines",
) -> list[TRTLLMCommandSet]:
    """Programmatic API: generate TensorRT-LLM benchmark commands."""
    config = TRTLLMCommandConfig(
        model=model,
        total_instances=total_instances,
        qps_levels=qps_levels,
        tp_size=tp_size,
        pp_size=pp_size,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        kv_cache_free_gpu_mem_fraction=kv_cache_free_gpu_mem_fraction,
        dtype=dtype,
        dataset=dataset,
        num_prompts=num_prompts,
        host=host,
        port=port,
        engine_dir=engine_dir,
    )
    return TRTLLMCommandGenerator(config).generate()

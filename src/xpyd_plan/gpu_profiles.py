"""Built-in GPU profile library."""

from __future__ import annotations

from xpyd_plan.models import GPUProfile

# Built-in GPU profiles with realistic throughput numbers.
# prefill_tokens_per_sec: tokens/s per GPU for prompt processing (compute-bound)
# decode_tokens_per_sec: tokens/s per GPU for autoregressive decode (memory-bandwidth-bound)
_BUILTIN_PROFILES: dict[str, GPUProfile] = {
    "A100-40G": GPUProfile(
        name="A100-40G",
        prefill_tokens_per_sec=40_000,
        decode_tokens_per_sec=1_600,
        memory_gb=40.0,
        cost_per_hour=1.80,
    ),
    "A100-80G": GPUProfile(
        name="A100-80G",
        prefill_tokens_per_sec=50_000,
        decode_tokens_per_sec=2_000,
        memory_gb=80.0,
        cost_per_hour=2.50,
    ),
    "H100-80G": GPUProfile(
        name="H100-80G",
        prefill_tokens_per_sec=120_000,
        decode_tokens_per_sec=5_000,
        memory_gb=80.0,
        cost_per_hour=4.50,
    ),
    "H200-141G": GPUProfile(
        name="H200-141G",
        prefill_tokens_per_sec=150_000,
        decode_tokens_per_sec=7_500,
        memory_gb=141.0,
        cost_per_hour=5.50,
    ),
    "L40S-48G": GPUProfile(
        name="L40S-48G",
        prefill_tokens_per_sec=35_000,
        decode_tokens_per_sec=1_400,
        memory_gb=48.0,
        cost_per_hour=1.50,
    ),
    "B200-192G": GPUProfile(
        name="B200-192G",
        prefill_tokens_per_sec=250_000,
        decode_tokens_per_sec=12_000,
        memory_gb=192.0,
        cost_per_hour=8.00,
    ),
    "A10G-24G": GPUProfile(
        name="A10G-24G",
        prefill_tokens_per_sec=18_000,
        decode_tokens_per_sec=800,
        memory_gb=24.0,
        cost_per_hour=0.75,
    ),
}


def get_gpu_profile(name: str) -> GPUProfile:
    """Get a built-in GPU profile by name.

    Raises:
        KeyError: If the profile name is not found.
    """
    if name not in _BUILTIN_PROFILES:
        available = ", ".join(sorted(_BUILTIN_PROFILES.keys()))
        raise KeyError(f"Unknown GPU profile '{name}'. Available: {available}")
    return _BUILTIN_PROFILES[name].model_copy()


def list_gpu_profiles() -> list[str]:
    """Return names of all built-in GPU profiles."""
    return sorted(_BUILTIN_PROFILES.keys())

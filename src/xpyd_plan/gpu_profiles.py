"""Built-in GPU profile library."""

from __future__ import annotations

from xpyd_plan.models import GPUProfile

# Built-in GPU profiles with realistic throughput numbers.
# prefill_tokens_per_sec: tokens/s per GPU for prompt processing (compute-bound)
# decode_tokens_per_sec: tokens/s per GPU for autoregressive decode (memory-bandwidth-bound)
_BUILTIN_PROFILES: dict[str, GPUProfile] = {
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

"""Built-in GPU profile library with optional YAML custom profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

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


def load_gpu_profiles(path: Union[str, Path]) -> dict[str, GPUProfile]:
    """Load custom GPU profiles from a YAML file.

    YAML format::

        profiles:
          - name: MI300X-192G
            prefill_tokens_per_sec: 180000
            decode_tokens_per_sec: 9000
            memory_gb: 192.0
            cost_per_hour: 6.50

    Args:
        path: Path to YAML file containing GPU profile definitions.

    Returns:
        Dictionary mapping profile name to GPUProfile.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML structure is invalid or missing required fields.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"GPU profiles file not found: {filepath}")

    with open(filepath) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "profiles" not in data:
        raise ValueError(
            "Invalid GPU profiles YAML: expected top-level 'profiles' key"
        )

    profiles_list = data["profiles"]
    if not isinstance(profiles_list, list):
        raise ValueError("Invalid GPU profiles YAML: 'profiles' must be a list")

    result: dict[str, GPUProfile] = {}
    for i, entry in enumerate(profiles_list):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid GPU profiles YAML: entry {i} must be a mapping")
        if "name" not in entry:
            raise ValueError(
                f"Invalid GPU profiles YAML: entry {i} missing required 'name' field"
            )
        try:
            profile = GPUProfile(**entry)
        except Exception as exc:
            raise ValueError(
                f"Invalid GPU profiles YAML: entry {i} ({entry.get('name', '?')}): {exc}"
            ) from exc
        result[profile.name] = profile

    return result


def get_all_profiles(
    custom_profiles_path: Union[str, Path, None] = None,
) -> dict[str, GPUProfile]:
    """Get all GPU profiles (built-in + custom).

    Custom profiles override built-in profiles with the same name.

    Args:
        custom_profiles_path: Optional path to YAML file with custom profiles.

    Returns:
        Merged dictionary of all GPU profiles.
    """
    merged = {k: v.model_copy() for k, v in _BUILTIN_PROFILES.items()}
    if custom_profiles_path is not None:
        custom = load_gpu_profiles(custom_profiles_path)
        merged.update(custom)
    return merged


def get_gpu_profile(
    name: str,
    custom_profiles_path: Union[str, Path, None] = None,
) -> GPUProfile:
    """Get a GPU profile by name.

    Searches custom profiles (if provided) first, then built-in profiles.

    Args:
        name: GPU profile name.
        custom_profiles_path: Optional path to YAML file with custom profiles.

    Raises:
        KeyError: If the profile name is not found.
    """
    all_profiles = get_all_profiles(custom_profiles_path)
    if name not in all_profiles:
        available = ", ".join(sorted(all_profiles.keys()))
        raise KeyError(f"Unknown GPU profile '{name}'. Available: {available}")
    return all_profiles[name]


def list_gpu_profiles(
    custom_profiles_path: Union[str, Path, None] = None,
) -> list[str]:
    """Return names of all available GPU profiles.

    Args:
        custom_profiles_path: Optional path to YAML file with custom profiles.
    """
    return sorted(get_all_profiles(custom_profiles_path).keys())

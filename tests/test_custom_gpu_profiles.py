"""Tests for custom GPU profile loading from YAML."""

from __future__ import annotations

import pytest
import yaml

from xpyd_plan.gpu_profiles import (
    get_all_profiles,
    get_gpu_profile,
    list_gpu_profiles,
    load_gpu_profiles,
)


@pytest.fixture()
def valid_yaml(tmp_path):
    """Create a valid custom GPU profiles YAML file."""
    data = {
        "profiles": [
            {
                "name": "MI300X-192G",
                "prefill_tokens_per_sec": 180_000,
                "decode_tokens_per_sec": 9_000,
                "memory_gb": 192.0,
                "cost_per_hour": 6.50,
            },
            {
                "name": "Gaudi3-128G",
                "prefill_tokens_per_sec": 100_000,
                "decode_tokens_per_sec": 4_500,
                "memory_gb": 128.0,
                "cost_per_hour": 3.80,
            },
        ]
    }
    path = tmp_path / "gpus.yaml"
    path.write_text(yaml.dump(data))
    return path


@pytest.fixture()
def override_yaml(tmp_path):
    """YAML that overrides a built-in profile."""
    data = {
        "profiles": [
            {
                "name": "A100-80G",
                "prefill_tokens_per_sec": 55_000,
                "decode_tokens_per_sec": 2_200,
                "memory_gb": 80.0,
                "cost_per_hour": 2.00,
            },
        ]
    }
    path = tmp_path / "override.yaml"
    path.write_text(yaml.dump(data))
    return path


def test_load_valid_yaml(valid_yaml):
    """Load custom profiles from valid YAML."""
    profiles = load_gpu_profiles(valid_yaml)
    assert len(profiles) == 2
    assert "MI300X-192G" in profiles
    assert "Gaudi3-128G" in profiles
    assert profiles["MI300X-192G"].prefill_tokens_per_sec == 180_000
    assert profiles["Gaudi3-128G"].cost_per_hour == 3.80


def test_load_file_not_found(tmp_path):
    """Raise FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_gpu_profiles(tmp_path / "nonexistent.yaml")


def test_load_invalid_no_profiles_key(tmp_path):
    """Raise ValueError when 'profiles' key missing."""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump({"gpus": []}))
    with pytest.raises(ValueError, match="profiles"):
        load_gpu_profiles(path)


def test_load_invalid_profiles_not_list(tmp_path):
    """Raise ValueError when profiles is not a list."""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump({"profiles": "not-a-list"}))
    with pytest.raises(ValueError, match="must be a list"):
        load_gpu_profiles(path)


def test_load_invalid_entry_not_mapping(tmp_path):
    """Raise ValueError when an entry is not a mapping."""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump({"profiles": ["just-a-string"]}))
    with pytest.raises(ValueError, match="must be a mapping"):
        load_gpu_profiles(path)


def test_load_missing_name(tmp_path):
    """Raise ValueError when entry missing 'name'."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.dump(
            {
                "profiles": [
                    {
                        "prefill_tokens_per_sec": 100,
                        "decode_tokens_per_sec": 50,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="missing required 'name'"):
        load_gpu_profiles(path)


def test_load_missing_required_field(tmp_path):
    """Raise ValueError when required model fields missing."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.dump(
            {
                "profiles": [
                    {
                        "name": "Bad-GPU",
                        # missing prefill_tokens_per_sec and decode_tokens_per_sec
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="Bad-GPU"):
        load_gpu_profiles(path)


def test_get_all_profiles_without_custom():
    """get_all_profiles returns only built-in when no custom path."""
    profiles = get_all_profiles()
    assert "A100-80G" in profiles
    assert "H100-80G" in profiles


def test_get_all_profiles_with_custom(valid_yaml):
    """get_all_profiles merges built-in and custom."""
    profiles = get_all_profiles(valid_yaml)
    assert "A100-80G" in profiles  # built-in
    assert "MI300X-192G" in profiles  # custom


def test_get_all_profiles_override(override_yaml):
    """Custom profile overrides built-in with same name."""
    profiles = get_all_profiles(override_yaml)
    a100 = profiles["A100-80G"]
    assert a100.prefill_tokens_per_sec == 55_000
    assert a100.cost_per_hour == 2.00


def test_get_gpu_profile_custom(valid_yaml):
    """get_gpu_profile can retrieve custom profile."""
    profile = get_gpu_profile("MI300X-192G", custom_profiles_path=valid_yaml)
    assert profile.decode_tokens_per_sec == 9_000


def test_get_gpu_profile_unknown_with_custom(valid_yaml):
    """KeyError includes custom profiles in available list."""
    with pytest.raises(KeyError, match="MI300X-192G"):
        get_gpu_profile("Nonexistent-GPU", custom_profiles_path=valid_yaml)


def test_list_gpu_profiles_with_custom(valid_yaml):
    """list_gpu_profiles includes custom profiles."""
    names = list_gpu_profiles(custom_profiles_path=valid_yaml)
    assert "MI300X-192G" in names
    assert "Gaudi3-128G" in names
    assert "A100-80G" in names  # built-in still present


def test_backward_compat_get_gpu_profile():
    """get_gpu_profile still works without custom path (backward compat)."""
    profile = get_gpu_profile("H100-80G")
    assert profile.prefill_tokens_per_sec == 120_000


def test_backward_compat_list_gpu_profiles():
    """list_gpu_profiles still works without custom path."""
    names = list_gpu_profiles()
    assert "H100-80G" in names


def test_empty_profiles_list(tmp_path):
    """Empty profiles list is valid, returns empty dict."""
    path = tmp_path / "empty.yaml"
    path.write_text(yaml.dump({"profiles": []}))
    profiles = load_gpu_profiles(path)
    assert profiles == {}


def test_load_string_path(valid_yaml):
    """load_gpu_profiles accepts string path."""
    profiles = load_gpu_profiles(str(valid_yaml))
    assert len(profiles) == 2

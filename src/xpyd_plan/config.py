"""Configuration profile support for xpyd-plan.

Loads defaults from YAML config files so users don't need to repeat CLI flags.
Resolution order: --config flag → ./xpyd-plan.yaml → ~/.config/xpyd-plan/config.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

_DEFAULT_CONFIG_FILENAME = "xpyd-plan.yaml"
_USER_CONFIG_DIR = Path.home() / ".config" / "xpyd-plan"

STARTER_CONFIG = """\
# xpyd-plan configuration profile
# CLI flags always override these defaults.

sla:
  # ttft_ms: 200.0       # Max TTFT P95 (ms)
  # tpot_ms: 50.0        # Max TPOT P95 (ms)
  # max_latency_ms: 5000.0  # Max total latency P95 (ms)

cost:
  # gpu_hourly_rate: 2.50
  # currency: USD
  # budget_ceiling: 100.0

output:
  format: table          # table | json | csv
  top: 5

defaults:
  # total_instances: 16
  benchmark_format: auto  # auto | native | xpyd-bench
"""


class SLAProfile(BaseModel):
    """SLA defaults from config file."""

    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None


class CostProfile(BaseModel):
    """Cost defaults from config file."""

    gpu_hourly_rate: Optional[float] = None
    currency: str = "USD"
    budget_ceiling: Optional[float] = None


class OutputProfile(BaseModel):
    """Output preferences from config file."""

    format: str = Field(default="table", pattern="^(table|json|csv)$")
    top: int = Field(default=5, ge=1)


class DefaultsProfile(BaseModel):
    """Default values for common parameters."""

    total_instances: Optional[int] = Field(default=None, ge=2)
    benchmark_format: str = Field(default="auto", pattern="^(auto|native|xpyd-bench)$")


class ConfigProfile(BaseModel):
    """Top-level configuration profile."""

    sla: SLAProfile = Field(default_factory=SLAProfile)
    cost: CostProfile = Field(default_factory=CostProfile)
    output: OutputProfile = Field(default_factory=OutputProfile)
    defaults: DefaultsProfile = Field(default_factory=DefaultsProfile)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ConfigProfile:
        """Load config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            return cls()
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serialize config to YAML string."""
        data = self.model_dump(exclude_none=True)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)


def resolve_config_path(explicit: str | None = None) -> Path | None:
    """Find the config file using the resolution chain.

    Order: explicit --config → ./xpyd-plan.yaml → ~/.config/xpyd-plan/config.yaml
    Returns None if no config file is found.
    """
    if explicit is not None:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Config file not found: {p}")

    # Check current directory
    local = Path.cwd() / _DEFAULT_CONFIG_FILENAME
    if local.exists():
        return local

    # Check user config directory
    user = _USER_CONFIG_DIR / "config.yaml"
    if user.exists():
        return user

    return None


def load_config(explicit: str | None = None) -> ConfigProfile:
    """Load configuration profile, falling back to defaults if no file found."""
    path = resolve_config_path(explicit)
    if path is None:
        return ConfigProfile()
    return ConfigProfile.from_yaml(path)


def init_config(path: str | Path | None = None) -> Path:
    """Write a starter config file. Returns the path written."""
    if path is None:
        path = Path.cwd() / _DEFAULT_CONFIG_FILENAME
    else:
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STARTER_CONFIG)
    return path

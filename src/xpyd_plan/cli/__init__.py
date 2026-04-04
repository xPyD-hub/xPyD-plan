"""CLI package for xpyd-plan."""

from xpyd_plan.cli._config import _apply_config_defaults
from xpyd_plan.cli._main import main

__all__ = ["main", "_apply_config_defaults"]

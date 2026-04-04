"""Plugin architecture for xPyD-plan.

Allows third-party packages to register custom analyzers and CLI subcommands
via Python entry points (group: ``xpyd_plan.plugins``).

Plugin authors should:
1. Implement a class that satisfies :class:`PluginSpec`.
2. Register it as an entry point in their ``pyproject.toml``::

       [project.entry-points."xpyd_plan.plugins"]
       my_plugin = "my_package.plugin:MyPlugin"

3. Install the package — xPyD-plan will discover and load it automatically.
"""

from __future__ import annotations

import importlib.metadata
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "xpyd_plan.plugins"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PluginType(str, Enum):
    """Type of capability a plugin provides."""

    ANALYZER = "analyzer"
    EXPORTER = "exporter"
    GENERAL = "general"


class PluginMetadata(BaseModel):
    """Metadata describing an installed plugin."""

    name: str = Field(..., description="Unique plugin name")
    version: str = Field(default="0.0.0", description="Plugin version")
    description: str = Field(default="", description="Short description")
    author: str = Field(default="", description="Author name or email")
    plugin_type: PluginType = Field(
        default=PluginType.GENERAL,
        description="Category of plugin",
    )


class PluginInfo(BaseModel):
    """Summary information about a discovered plugin (for listing)."""

    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    entry_point: str = Field(
        default="", description="Entry point string (package.module:Class)"
    )
    loaded: bool = Field(default=False, description="Whether the plugin loaded OK")
    error: Optional[str] = Field(
        default=None, description="Error message if loading failed"
    )


class PluginListReport(BaseModel):
    """Report returned by ``list_plugins()``."""

    plugins: List[PluginInfo] = Field(default_factory=list)
    total: int = 0
    loaded: int = 0
    failed: int = 0


# ---------------------------------------------------------------------------
# Plugin interface
# ---------------------------------------------------------------------------

class PluginSpec(ABC):
    """Abstract base class that every xPyD-plan plugin must implement.

    Minimal example::

        class MyPlugin(PluginSpec):
            @staticmethod
            def metadata() -> PluginMetadata:
                return PluginMetadata(name="my-plugin", version="1.0.0",
                                      description="Demo plugin")

            def analyze(self, data: dict) -> dict:
                return {"result": "hello"}
    """

    @staticmethod
    @abstractmethod
    def metadata() -> PluginMetadata:
        """Return metadata describing this plugin."""
        ...

    def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the plugin's analysis on *data* and return results.

        Override this in analyzer plugins.  The default implementation
        returns an empty dict (suitable for non-analyzer plugins).
        """
        return {}

    def register_cli(self, subparsers: Any) -> None:
        """Register CLI subcommand(s) on the given *subparsers* object.

        Override this if the plugin provides CLI commands.
        """
        return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class PluginRegistry:
    """Discover, validate, and manage plugins."""

    def __init__(self) -> None:
        self._plugins: Dict[str, PluginSpec] = {}
        self._errors: Dict[str, str] = {}
        self._entry_points: Dict[str, str] = {}

    # -- discovery ----------------------------------------------------------

    def discover(self) -> None:
        """Scan entry points and load all available plugins."""
        eps = importlib.metadata.entry_points()
        # Python 3.12+ returns a SelectableGroups; 3.10/3.11 differ
        if hasattr(eps, "select"):
            group = eps.select(group=ENTRY_POINT_GROUP)
        else:
            group = eps.get(ENTRY_POINT_GROUP, [])

        for ep in group:
            self._load_entry_point(ep)

    def _load_entry_point(self, ep: importlib.metadata.EntryPoint) -> None:
        ep_str = f"{ep.value}"
        try:
            cls = ep.load()
            self._register_class(cls, ep_str)
        except Exception as exc:  # noqa: BLE001
            name = ep.name
            err = f"Failed to load plugin '{name}': {exc}"
            logger.warning(err)
            self._errors[name] = err
            self._entry_points[name] = ep_str

    def _register_class(self, cls: type, ep_str: str) -> None:
        """Instantiate and validate a plugin class."""
        if not (isinstance(cls, type) and issubclass(cls, PluginSpec)):
            raise TypeError(
                f"{cls!r} does not subclass PluginSpec"
            )
        instance = cls()
        meta = instance.metadata()
        if not isinstance(meta, PluginMetadata):
            raise TypeError(
                f"metadata() must return PluginMetadata, got {type(meta)!r}"
            )
        name = meta.name
        if name in self._plugins:
            logger.warning("Duplicate plugin name '%s' — skipping", name)
            return
        self._plugins[name] = instance
        self._entry_points[name] = ep_str

    # -- manual registration (for testing / embedded use) -------------------

    def register(self, plugin: PluginSpec) -> None:
        """Manually register a plugin instance."""
        meta = plugin.metadata()
        if not isinstance(meta, PluginMetadata):
            raise TypeError("metadata() must return PluginMetadata")
        name = meta.name
        if name in self._plugins:
            logger.warning("Duplicate plugin name '%s' — skipping", name)
            return
        self._plugins[name] = plugin
        self._entry_points[name] = "(manual)"

    # -- queries ------------------------------------------------------------

    def get_plugin(self, name: str) -> Optional[PluginSpec]:
        """Return a loaded plugin by name, or ``None``."""
        return self._plugins.get(name)

    def list_plugins(self) -> PluginListReport:
        """Return a report of all discovered plugins."""
        infos: List[PluginInfo] = []

        for name, instance in self._plugins.items():
            meta = instance.metadata()
            infos.append(
                PluginInfo(
                    name=meta.name,
                    version=meta.version,
                    description=meta.description,
                    author=meta.author,
                    plugin_type=meta.plugin_type,
                    entry_point=self._entry_points.get(name, ""),
                    loaded=True,
                )
            )

        for name, err in self._errors.items():
            if name not in self._plugins:
                infos.append(
                    PluginInfo(
                        name=name,
                        version="?",
                        description="",
                        author="",
                        plugin_type=PluginType.GENERAL,
                        entry_point=self._entry_points.get(name, ""),
                        loaded=False,
                        error=err,
                    )
                )

        return PluginListReport(
            plugins=infos,
            total=len(infos),
            loaded=len(self._plugins),
            failed=len([i for i in infos if not i.loaded]),
        )

    def register_all_cli(self, subparsers: Any) -> None:
        """Let every loaded plugin register its CLI subcommands."""
        for instance in self._plugins.values():
            try:
                instance.register_cli(subparsers)
            except Exception as exc:  # noqa: BLE001
                meta = instance.metadata()
                logger.warning(
                    "Plugin '%s' failed to register CLI: %s", meta.name, exc
                )

    @property
    def names(self) -> List[str]:
        """Names of all loaded plugins."""
        return list(self._plugins.keys())


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Return (and lazily create) the default global plugin registry."""
    global _default_registry  # noqa: PLW0603
    if _default_registry is None:
        _default_registry = PluginRegistry()
        _default_registry.discover()
    return _default_registry


def list_plugins() -> PluginListReport:
    """List all discovered plugins (convenience wrapper)."""
    return get_registry().list_plugins()


def get_plugin(name: str) -> Optional[PluginSpec]:
    """Get a plugin by name (convenience wrapper)."""
    return get_registry().get_plugin(name)

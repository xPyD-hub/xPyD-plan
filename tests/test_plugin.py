"""Tests for plugin architecture (M52)."""

from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from xpyd_plan.plugin import (
    PluginInfo,
    PluginListReport,
    PluginMetadata,
    PluginRegistry,
    PluginSpec,
    PluginType,
    get_plugin,
    get_registry,
    list_plugins,
)

# ---------------------------------------------------------------------------
# Fixtures: sample plugins
# ---------------------------------------------------------------------------

class _GoodPlugin(PluginSpec):
    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="good-plugin",
            version="1.0.0",
            description="A well-behaved plugin",
            author="tester",
            plugin_type=PluginType.ANALYZER,
        )

    def analyze(self, data: dict) -> dict:
        return {"doubled": data.get("value", 0) * 2}


class _ExporterPlugin(PluginSpec):
    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="exporter-plugin",
            version="0.5.0",
            description="Exports stuff",
            author="exporter-dev",
            plugin_type=PluginType.EXPORTER,
        )


class _CLIPlugin(PluginSpec):
    """Plugin that registers a CLI subcommand."""

    registered = False

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(name="cli-plugin", version="2.0.0")

    def register_cli(self, subparsers):
        _CLIPlugin.registered = True
        subparsers.add_parser("my-custom-cmd", help="custom")


class _BadMetadataPlugin(PluginSpec):
    @staticmethod
    def metadata():
        return "not-a-PluginMetadata"


class _DuplicatePlugin(PluginSpec):
    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(name="good-plugin", version="9.9.9")


# ---------------------------------------------------------------------------
# PluginMetadata
# ---------------------------------------------------------------------------

class TestPluginMetadata:
    def test_defaults(self):
        m = PluginMetadata(name="x")
        assert m.version == "0.0.0"
        assert m.description == ""
        assert m.author == ""
        assert m.plugin_type == PluginType.GENERAL

    def test_full(self):
        m = PluginMetadata(
            name="foo",
            version="1.2.3",
            description="desc",
            author="me",
            plugin_type=PluginType.ANALYZER,
        )
        assert m.name == "foo"
        assert m.plugin_type == PluginType.ANALYZER


# ---------------------------------------------------------------------------
# PluginSpec ABC
# ---------------------------------------------------------------------------

class TestPluginSpec:
    def test_good_plugin_metadata(self):
        p = _GoodPlugin()
        m = p.metadata()
        assert m.name == "good-plugin"
        assert m.version == "1.0.0"

    def test_analyze_default_returns_empty(self):
        p = _ExporterPlugin()
        assert p.analyze({}) == {}

    def test_analyze_override(self):
        p = _GoodPlugin()
        assert p.analyze({"value": 5}) == {"doubled": 10}

    def test_register_cli_default_is_noop(self):
        p = _GoodPlugin()
        assert p.register_cli(None) is None


# ---------------------------------------------------------------------------
# PluginRegistry
# ---------------------------------------------------------------------------

class TestPluginRegistry:
    def test_register_and_get(self):
        reg = PluginRegistry()
        reg.register(_GoodPlugin())
        assert reg.get_plugin("good-plugin") is not None
        assert reg.get_plugin("nonexistent") is None

    def test_names(self):
        reg = PluginRegistry()
        reg.register(_GoodPlugin())
        reg.register(_ExporterPlugin())
        assert set(reg.names) == {"good-plugin", "exporter-plugin"}

    def test_duplicate_skipped(self):
        reg = PluginRegistry()
        reg.register(_GoodPlugin())
        reg.register(_DuplicatePlugin())
        p = reg.get_plugin("good-plugin")
        assert p.metadata().version == "1.0.0"  # first one wins

    def test_bad_metadata_raises(self):
        reg = PluginRegistry()
        with pytest.raises(TypeError, match="PluginMetadata"):
            reg.register(_BadMetadataPlugin())

    def test_list_plugins_empty(self):
        reg = PluginRegistry()
        report = reg.list_plugins()
        assert report.total == 0
        assert report.loaded == 0
        assert report.failed == 0
        assert report.plugins == []

    def test_list_plugins_with_loaded(self):
        reg = PluginRegistry()
        reg.register(_GoodPlugin())
        reg.register(_ExporterPlugin())
        report = reg.list_plugins()
        assert report.total == 2
        assert report.loaded == 2
        assert report.failed == 0
        names = {p.name for p in report.plugins}
        assert names == {"good-plugin", "exporter-plugin"}

    def test_list_plugins_shows_errors(self):
        reg = PluginRegistry()
        reg._errors["broken"] = "some error"
        reg._entry_points["broken"] = "pkg:Cls"
        report = reg.list_plugins()
        assert report.total == 1
        assert report.failed == 1
        assert report.plugins[0].loaded is False
        assert report.plugins[0].error == "some error"

    def test_register_all_cli(self):
        reg = PluginRegistry()
        _CLIPlugin.registered = False
        reg.register(_CLIPlugin())
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        reg.register_all_cli(subs)
        assert _CLIPlugin.registered

    def test_register_non_plugin_raises(self):
        reg = PluginRegistry()

        class NotAPlugin:
            pass

        with pytest.raises(TypeError, match="PluginSpec"):
            reg._register_class(NotAPlugin, "test:NotAPlugin")

    def test_discover_with_mocked_entry_points(self):
        """Test that discover() loads entry points from the correct group."""
        reg = PluginRegistry()
        mock_ep = MagicMock()
        mock_ep.name = "good-plugin"
        mock_ep.value = "test_module:_GoodPlugin"
        mock_ep.load.return_value = _GoodPlugin

        with patch("xpyd_plan.plugin.importlib.metadata.entry_points") as mock_eps:
            mock_result = MagicMock()
            mock_result.select.return_value = [mock_ep]
            mock_eps.return_value = mock_result
            reg.discover()

        assert "good-plugin" in reg.names

    def test_discover_handles_load_failure(self):
        reg = PluginRegistry()
        mock_ep = MagicMock()
        mock_ep.name = "broken-plugin"
        mock_ep.value = "bad:Plugin"
        mock_ep.load.side_effect = ImportError("no such module")

        with patch("xpyd_plan.plugin.importlib.metadata.entry_points") as mock_eps:
            mock_result = MagicMock()
            mock_result.select.return_value = [mock_ep]
            mock_eps.return_value = mock_result
            reg.discover()

        assert "broken-plugin" not in reg.names
        report = reg.list_plugins()
        assert report.failed == 1


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    def test_list_plugins_returns_report(self):
        import xpyd_plan.plugin as mod

        old = mod._default_registry
        try:
            mod._default_registry = None
            with patch("xpyd_plan.plugin.importlib.metadata.entry_points") as mock_eps:
                mock_result = MagicMock()
                mock_result.select.return_value = []
                mock_eps.return_value = mock_result
                report = list_plugins()
            assert isinstance(report, PluginListReport)
            assert report.total == 0
        finally:
            mod._default_registry = old

    def test_get_plugin_returns_none_for_missing(self):
        import xpyd_plan.plugin as mod

        old = mod._default_registry
        try:
            mod._default_registry = None
            with patch("xpyd_plan.plugin.importlib.metadata.entry_points") as mock_eps:
                mock_result = MagicMock()
                mock_result.select.return_value = []
                mock_eps.return_value = mock_result
                assert get_plugin("nope") is None
        finally:
            mod._default_registry = old

    def test_get_registry_caches(self):
        import xpyd_plan.plugin as mod

        old = mod._default_registry
        try:
            mod._default_registry = None
            with patch("xpyd_plan.plugin.importlib.metadata.entry_points") as mock_eps:
                mock_result = MagicMock()
                mock_result.select.return_value = []
                mock_eps.return_value = mock_result
                r1 = get_registry()
                r2 = get_registry()
            assert r1 is r2
        finally:
            mod._default_registry = old


# ---------------------------------------------------------------------------
# PluginInfo / PluginListReport models
# ---------------------------------------------------------------------------

class TestModels:
    def test_plugin_info_defaults(self):
        info = PluginInfo(
            name="x", version="1", description="", author="",
            plugin_type=PluginType.GENERAL,
        )
        assert info.loaded is False
        assert info.error is None

    def test_plugin_list_report_json_roundtrip(self):
        report = PluginListReport(
            plugins=[
                PluginInfo(
                    name="a",
                    version="1.0",
                    description="test",
                    author="dev",
                    plugin_type=PluginType.ANALYZER,
                    loaded=True,
                )
            ],
            total=1,
            loaded=1,
            failed=0,
        )
        data = json.loads(report.model_dump_json())
        assert data["total"] == 1
        assert data["plugins"][0]["name"] == "a"

    def test_plugin_type_values(self):
        assert PluginType.ANALYZER.value == "analyzer"
        assert PluginType.EXPORTER.value == "exporter"
        assert PluginType.GENERAL.value == "general"

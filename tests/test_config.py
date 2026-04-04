"""Tests for configuration profile support (M12)."""

from __future__ import annotations

import pytest
import yaml

from xpyd_plan.config import (
    STARTER_CONFIG,
    ConfigProfile,
    CostProfile,
    DefaultsProfile,
    OutputProfile,
    SLAProfile,
    init_config,
    load_config,
    resolve_config_path,
)

# --- ConfigProfile model tests ---


class TestConfigProfile:
    def test_default_profile(self):
        cfg = ConfigProfile()
        assert cfg.sla.ttft_ms is None
        assert cfg.sla.tpot_ms is None
        assert cfg.output.format == "table"
        assert cfg.output.top == 5
        assert cfg.defaults.benchmark_format == "auto"
        assert cfg.cost.currency == "USD"

    def test_full_profile(self):
        cfg = ConfigProfile(
            sla=SLAProfile(ttft_ms=200.0, tpot_ms=50.0, max_latency_ms=5000.0),
            cost=CostProfile(gpu_hourly_rate=2.5, currency="EUR", budget_ceiling=100.0),
            output=OutputProfile(format="json", top=10),
            defaults=DefaultsProfile(total_instances=16, benchmark_format="xpyd-bench"),
        )
        assert cfg.sla.ttft_ms == 200.0
        assert cfg.cost.gpu_hourly_rate == 2.5
        assert cfg.output.format == "json"
        assert cfg.defaults.total_instances == 16

    def test_from_yaml(self, tmp_path):
        config_data = {
            "sla": {"ttft_ms": 150.0, "tpot_ms": 40.0},
            "output": {"format": "csv", "top": 8},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config_data))

        cfg = ConfigProfile.from_yaml(path)
        assert cfg.sla.ttft_ms == 150.0
        assert cfg.sla.tpot_ms == 40.0
        assert cfg.sla.max_latency_ms is None
        assert cfg.output.format == "csv"
        assert cfg.output.top == 8

    def test_from_yaml_empty(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        cfg = ConfigProfile.from_yaml(path)
        assert cfg.sla.ttft_ms is None

    def test_from_yaml_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ConfigProfile.from_yaml(tmp_path / "nope.yaml")

    def test_to_yaml(self):
        cfg = ConfigProfile(
            sla=SLAProfile(ttft_ms=200.0),
            output=OutputProfile(format="json", top=10),
        )
        text = cfg.to_yaml()
        loaded = yaml.safe_load(text)
        assert loaded["sla"]["ttft_ms"] == 200.0
        assert loaded["output"]["format"] == "json"

    def test_partial_sections(self, tmp_path):
        """Only some sections present in YAML."""
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.dump({"cost": {"gpu_hourly_rate": 3.0}}))
        cfg = ConfigProfile.from_yaml(path)
        assert cfg.cost.gpu_hourly_rate == 3.0
        assert cfg.sla.ttft_ms is None  # default


class TestSLAProfile:
    def test_all_none(self):
        sla = SLAProfile()
        assert sla.ttft_ms is None
        assert sla.tpot_ms is None
        assert sla.max_latency_ms is None

    def test_partial(self):
        sla = SLAProfile(ttft_ms=100.0)
        assert sla.ttft_ms == 100.0
        assert sla.tpot_ms is None


class TestOutputProfile:
    def test_defaults(self):
        out = OutputProfile()
        assert out.format == "table"
        assert out.top == 5

    def test_validation_format(self):
        with pytest.raises(Exception):
            OutputProfile(format="xml")

    def test_validation_top(self):
        with pytest.raises(Exception):
            OutputProfile(top=0)


class TestDefaultsProfile:
    def test_validation_instances(self):
        with pytest.raises(Exception):
            DefaultsProfile(total_instances=1)

    def test_validation_format(self):
        with pytest.raises(Exception):
            DefaultsProfile(benchmark_format="parquet")


# --- resolve_config_path tests ---


class TestResolveConfigPath:
    def test_explicit_found(self, tmp_path):
        path = tmp_path / "my.yaml"
        path.write_text("sla: {}")
        assert resolve_config_path(str(path)) == path

    def test_explicit_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_config_path(str(tmp_path / "nope.yaml"))

    def test_no_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Ensure no user config either
        monkeypatch.setattr(
            "xpyd_plan.config._USER_CONFIG_DIR", tmp_path / "nonexistent"
        )
        assert resolve_config_path() is None

    def test_local_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        local = tmp_path / "xpyd-plan.yaml"
        local.write_text("sla: {}")
        assert resolve_config_path() == local

    def test_user_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        user_dir = tmp_path / "user_config"
        user_dir.mkdir()
        user_cfg = user_dir / "config.yaml"
        user_cfg.write_text("sla: {}")
        monkeypatch.setattr("xpyd_plan.config._USER_CONFIG_DIR", user_dir)
        assert resolve_config_path() == user_cfg

    def test_local_takes_priority(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        local = tmp_path / "xpyd-plan.yaml"
        local.write_text("output: {format: json}")
        user_dir = tmp_path / "user_config"
        user_dir.mkdir()
        (user_dir / "config.yaml").write_text("output: {format: csv}")
        monkeypatch.setattr("xpyd_plan.config._USER_CONFIG_DIR", user_dir)
        assert resolve_config_path() == local


# --- load_config tests ---


class TestLoadConfig:
    def test_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "xpyd_plan.config._USER_CONFIG_DIR", tmp_path / "nonexistent"
        )
        cfg = load_config()
        assert cfg.sla.ttft_ms is None

    def test_explicit_file(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump({"sla": {"ttft_ms": 300.0}}))
        cfg = load_config(str(path))
        assert cfg.sla.ttft_ms == 300.0


# --- init_config tests ---


class TestInitConfig:
    def test_default_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = init_config()
        assert result == tmp_path / "xpyd-plan.yaml"
        assert result.exists()
        content = result.read_text()
        assert "sla:" in content

    def test_custom_path(self, tmp_path):
        custom = tmp_path / "sub" / "my-config.yaml"
        result = init_config(custom)
        assert result == custom
        assert custom.exists()

    def test_starter_is_valid_yaml(self):
        data = yaml.safe_load(STARTER_CONFIG)
        assert data is not None
        assert "sla" in data or "output" in data


# --- CLI integration tests ---


class TestCLIConfigIntegration:
    def test_config_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from xpyd_plan.cli import main

        main(["config", "init"])
        assert (tmp_path / "xpyd-plan.yaml").exists()

    def test_config_init_custom_path(self, tmp_path):
        from xpyd_plan.cli import main

        out = tmp_path / "custom.yaml"
        main(["config", "init", "--output-path", str(out)])
        assert out.exists()

    def test_config_show(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "xpyd-plan.yaml").write_text(
            yaml.dump({"sla": {"ttft_ms": 250.0}})
        )
        from xpyd_plan.cli import main

        main(["config", "show"])
        captured = capsys.readouterr()
        assert "250" in captured.out

    def test_analyze_picks_up_config_sla(self, tmp_path, monkeypatch):
        """Config SLA defaults are applied when CLI flags not given."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "xpyd-plan.yaml").write_text(
            yaml.dump({"sla": {"ttft_ms": 999.0, "tpot_ms": 88.0}})
        )

        import argparse

        from xpyd_plan.cli import _apply_config_defaults

        args = argparse.Namespace(
            config_file=None,
            sla_ttft=None,
            sla_tpot=None,
            sla_max_latency=None,
            output_format="table",
            top=5,
            total_instances=None,
            format="auto",
            budget_ceiling=None,
        )
        _apply_config_defaults(args)
        assert args.sla_ttft == 999.0
        assert args.sla_tpot == 88.0

    def test_cli_flags_override_config(self, tmp_path, monkeypatch):
        """CLI flags take priority over config file."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "xpyd-plan.yaml").write_text(
            yaml.dump({"sla": {"ttft_ms": 999.0}})
        )

        import argparse

        from xpyd_plan.cli import _apply_config_defaults

        args = argparse.Namespace(
            config_file=None,
            sla_ttft=100.0,  # CLI explicitly set
            sla_tpot=None,
            sla_max_latency=None,
            output_format="table",
            top=5,
            total_instances=None,
            format="auto",
            budget_ceiling=None,
        )
        _apply_config_defaults(args)
        assert args.sla_ttft == 100.0  # CLI wins

    def test_explicit_config_flag(self, tmp_path):
        """--config flag points to specific file."""
        import argparse

        from xpyd_plan.cli import _apply_config_defaults

        cfg_path = tmp_path / "special.yaml"
        cfg_path.write_text(yaml.dump({"defaults": {"total_instances": 32}}))

        args = argparse.Namespace(
            config_file=str(cfg_path),
            sla_ttft=None,
            sla_tpot=None,
            sla_max_latency=None,
            output_format="table",
            top=5,
            total_instances=None,
            format="auto",
            budget_ceiling=None,
        )
        _apply_config_defaults(args)
        assert args.total_instances == 32

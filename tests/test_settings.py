"""Settings use temporary files and the real feature/pipeline registry."""
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utils import PipelineBuilder, State
from agent_zoo.runtime.settings import Setting, SettingError, SettingsContext, SettingsRegistry
import better_communication as plugin


def document():
    raw = json.loads((Path(plugin.__file__).parents[1] / "better_communication.json").read_text())
    raw["enabled"] = False
    raw["unknown"] = {"prompt": "do not change", "ratio": 0.125}
    return raw


def dump(raw):
    return json.dumps(raw)


def load(path):
    return json.loads(path.read_text())


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text(dump(document()))
    path.chmod(0o640)
    monkeypatch.setenv("AZO_BETTER_COMMUNICATION_CONFIG", str(path))
    return path


def registry(config=None):
    session = SimpleNamespace(state=State())
    builder = PipelineBuilder(session)
    plugin.register_features(builder, session=session, config=config or {})
    # Build the normal registration, including its settings while disabled.
    components = builder.build()
    settings = [component for component in components if isinstance(component, Setting)]
    result = SettingsRegistry(settings)
    assert result.error is None
    assert len(settings) == 1
    return result, SettingsContext(session), builder


def edit(registry, ctx, value):
    return registry.handle({"type": "plugin_settings", "request_id": "test", "action": "set",
                           "binding": registry.binding,
                           "key": "better_communication.enabled", "value": value}, ctx)


def test_disabled_registration_round_trip(settings_file):
    original = load(settings_file)
    reg, ctx, builder = registry()
    row, = reg.catalog(ctx)
    assert row["value"] == False
    assert row["kind"] == "bool"
    assert row["apply"] == "reload" and row["writable"]
    assert row["error"] is None
    assert not builder.has("guarded_message_composition")
    assert "private prompt" not in json.dumps(reg.catalog(ctx))
    assert edit(reg, ctx, True)["ok"]
    expected = original.copy()
    expected["enabled"] = True
    assert load(settings_file) == expected
    assert stat.S_IMODE(settings_file.stat().st_mode) == 0o640
    # The existing builder is not live-mutated by a reload-scoped setting.
    assert not builder.has("guarded_message_composition")
    session = SimpleNamespace(state=State())
    fresh = PipelineBuilder(session)
    plugin.register_features(fresh, session=session, config={})
    assert fresh.has("guarded_message_composition")
    assert reg.catalog(ctx)[0]["value"] == True
    assert list(settings_file.parent.iterdir()) == [settings_file]


@pytest.mark.parametrize("broken", ["[", "[]", "null", ""])
def test_bad_file_is_never_overwritten(settings_file, broken):
    reg, ctx, _ = registry()
    settings_file.write_text(broken)
    before = settings_file.read_bytes()
    assert reg.catalog(ctx)[0]["error"] is not None
    assert not edit(reg, ctx, True)["ok"]
    with pytest.raises(SettingError):
        plugin._save_enabled(True)
    assert settings_file.read_bytes() == before


def test_failed_replace_preserves_file_and_cleans_temp(settings_file, monkeypatch):
    reg, ctx, _ = registry()
    before = settings_file.read_bytes()
    def fail(*args):
        raise OSError("private path must not leak")
    monkeypatch.setattr(plugin.os, "replace", fail)
    result = edit(reg, ctx, True)
    assert not result["ok"]
    assert "private path" not in json.dumps(result)
    assert settings_file.read_bytes() == before
    assert list(settings_file.parent.iterdir()) == [settings_file]


def test_invalid_type_does_not_write(settings_file):
    reg, ctx, _ = registry()
    before = settings_file.read_bytes()
    assert not edit(reg, ctx, 1)["ok"]
    assert settings_file.read_bytes() == before


def test_unreadable_file_is_safe(settings_file, monkeypatch):
    reg, ctx, _ = registry()
    def fail(*args, **kwargs):
        raise PermissionError("private path")
    monkeypatch.setattr(Path, "read_text", fail)
    row, = reg.catalog(ctx)
    assert row["error"] is not None
    assert "private path" not in json.dumps(row)
    with pytest.raises(SettingError):
        plugin._save_enabled(True)


def test_missing_user_config_never_writes_bundled_source(tmp_path, monkeypatch):
    monkeypatch.delenv("AZO_BETTER_COMMUNICATION_CONFIG", raising=False)
    monkeypatch.setenv("AGENT_ZOO_INSTALL_ROOT", str(tmp_path))
    bundled = plugin._config_path()
    before = bundled.read_bytes()
    target = plugin._config_path(writable=True)
    assert target != bundled and not target.exists()
    session = SimpleNamespace(state=State())
    builder = PipelineBuilder(session)
    plugin._register_settings(builder)
    reg = SettingsRegistry([c for c in builder.build() if isinstance(c, Setting)])
    ctx = SettingsContext(session)
    assert reg.catalog(ctx)[0]["value"] is bool(load(bundled).get("enabled", True))
    assert not edit(reg, ctx, False)["ok"]
    assert bundled.read_bytes() == before
    assert not target.exists()


def test_saved_default_ignores_registration_and_session_state(settings_file):
    reg, ctx, builder = registry({"better_communication": {"enabled": True}})
    ctx.state.better_communication_enabled = True
    assert reg.catalog(ctx)[0]["value"] is False
    assert edit(reg, ctx, True)["ok"]
    assert not builder.has("guarded_message_composition")
    assert ctx.state.better_communication_enabled is True

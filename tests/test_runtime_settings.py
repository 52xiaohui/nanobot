from nanobot.agent.commands import (
    build_request_options,
    clear_runtime_setting,
    get_runtime_setting,
    resolve_effective_model,
    set_runtime_setting,
)
from nanobot.session.manager import Session


def test_runtime_setting_roundtrip() -> None:
    session = Session(key="cli:test")

    assert get_runtime_setting(session, "model") is None

    set_runtime_setting(session, "model", "openai-responses/gpt-5")
    assert get_runtime_setting(session, "model") == "openai-responses/gpt-5"

    clear_runtime_setting(session, "model")
    assert get_runtime_setting(session, "model") is None
    assert "runtime" not in session.metadata


def test_resolve_effective_model_priority() -> None:
    session = Session(key="cli:test")

    model, source = resolve_effective_model(
        session,
        config_default_model="anthropic/claude-opus-4-5",
        provider_default_model="provider/default",
    )
    assert model == "anthropic/claude-opus-4-5"
    assert source == "config"

    set_runtime_setting(session, "model", "openai-responses/gpt-5")
    model, source = resolve_effective_model(
        session,
        config_default_model="anthropic/claude-opus-4-5",
        provider_default_model="provider/default",
    )
    assert model == "openai-responses/gpt-5"
    assert source == "session"

    clear_runtime_setting(session, "model")
    model, source = resolve_effective_model(
        session,
        config_default_model="",
        provider_default_model="provider/default",
    )
    assert model == "provider/default"
    assert source == "provider"


def test_build_request_options() -> None:
    session = Session(key="cli:test")
    assert build_request_options(session) is None

    set_runtime_setting(session, "reasoning_effort", "HIGH")
    assert build_request_options(session) == {"reasoning_effort": "high"}

    set_runtime_setting(session, "reasoning_effort", "invalid")
    assert build_request_options(session) is None


def test_session_clear_keeps_runtime_overrides() -> None:
    session = Session(key="cli:test")
    session.messages = [{"role": "user", "content": "hello"}]
    session.last_consolidated = 1
    set_runtime_setting(session, "model", "openai-responses/gpt-5")

    session.clear()

    assert session.messages == []
    assert session.last_consolidated == 0
    assert session.metadata["runtime"]["model"] == "openai-responses/gpt-5"

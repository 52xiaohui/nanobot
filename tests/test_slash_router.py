from pathlib import Path

import pytest

from nanobot.agent.commands import RuntimeSettings, SlashCommandRouter, get_runtime_setting
from nanobot.agent.commands.types import build_request_options
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.config.schema import Config
from nanobot.providers.factory import ProviderCreateError
from nanobot.session.manager import Session, SessionManager


class StubProvider:
    def __init__(self, model: str):
        self._model = model

    def get_default_model(self) -> str:
        return self._model


class StubFactory:
    def __init__(self) -> None:
        self.created: list[str] = []

    def create(self, config: Config, model: str) -> StubProvider:
        self.created.append(model)
        if model == "invalid/model":
            raise ProviderCreateError("provider create failed")
        return StubProvider(model)


def _build_router(tmp_path: Path) -> tuple[SlashCommandRouter, SessionManager]:
    config = Config()
    config.agents.defaults.model = "anthropic/claude-opus-4-5"
    sessions = SessionManager(tmp_path)
    factory = StubFactory()

    async def _handle_new(msg: InboundMessage, session: Session) -> OutboundMessage:
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="new-called")

    def _resolve_runtime(session: Session) -> RuntimeSettings:
        runtime_model = get_runtime_setting(session, "model")
        effective_model = runtime_model or config.agents.defaults.model
        source = "session" if runtime_model else "config"
        provider = factory.create(config, effective_model)
        return RuntimeSettings(
            effective_model=effective_model,
            model_source=source,
            provider=provider,
            request_options=build_request_options(session),
        )

    router = SlashCommandRouter(
        config=config,
        sessions=sessions,
        provider_factory=factory,
        resolve_runtime_settings=_resolve_runtime,
        handle_new=_handle_new,
    )
    return router, sessions


def _msg(content: str) -> InboundMessage:
    return InboundMessage(
        channel="cli",
        sender_id="user",
        chat_id="direct",
        content=content,
        metadata={},
    )


@pytest.mark.asyncio
async def test_non_slash_returns_none(tmp_path: Path) -> None:
    router, sessions = _build_router(tmp_path)
    session = sessions.get_or_create("cli:direct")

    result = await router.handle(_msg("hello"), session)

    assert result is None


@pytest.mark.asyncio
async def test_help_command_returns_full_help(tmp_path: Path) -> None:
    router, sessions = _build_router(tmp_path)
    session = sessions.get_or_create("cli:direct")

    result = await router.handle(_msg("/help"), session)

    assert result is not None
    assert "/model <model_name>" in result.content
    assert "/think <low|medium|high>" in result.content


@pytest.mark.asyncio
async def test_model_set_and_reset(tmp_path: Path) -> None:
    router, sessions = _build_router(tmp_path)
    session = sessions.get_or_create("cli:direct")

    result_set = await router.handle(_msg("/model openai-responses/gpt-5"), session)
    assert result_set is not None
    assert "Model updated:" in result_set.content
    assert session.metadata["runtime"]["model"] == "openai-responses/gpt-5"

    result_reset = await router.handle(_msg("/model reset"), session)
    assert result_reset is not None
    assert "Model override cleared." in result_reset.content
    assert "runtime" not in session.metadata or "model" not in session.metadata.get("runtime", {})


@pytest.mark.asyncio
async def test_model_set_failure_keeps_existing_value(tmp_path: Path) -> None:
    router, sessions = _build_router(tmp_path)
    session = sessions.get_or_create("cli:direct")
    session.metadata["runtime"] = {"model": "openai-responses/gpt-5"}

    result = await router.handle(_msg("/model invalid/model"), session)

    assert result is not None
    assert "Failed to set model" in result.content
    assert session.metadata["runtime"]["model"] == "openai-responses/gpt-5"


@pytest.mark.asyncio
async def test_think_set_invalid_reset(tmp_path: Path) -> None:
    router, sessions = _build_router(tmp_path)
    session = sessions.get_or_create("cli:direct")

    invalid = await router.handle(_msg("/think super"), session)
    assert invalid is not None
    assert "Invalid thinking effort" in invalid.content
    assert "runtime" not in session.metadata

    set_result = await router.handle(_msg("/think high"), session)
    assert set_result is not None
    assert "Thinking effort set to high." in set_result.content
    assert session.metadata["runtime"]["reasoning_effort"] == "high"

    reset_result = await router.handle(_msg("/think reset"), session)
    assert reset_result is not None
    assert "override cleared" in reset_result.content
    assert "runtime" not in session.metadata


@pytest.mark.asyncio
async def test_new_command_dispatches_callback(tmp_path: Path) -> None:
    router, sessions = _build_router(tmp_path)
    session = sessions.get_or_create("cli:direct")

    result = await router.handle(_msg("/new"), session)

    assert result is not None
    assert result.content == "new-called"

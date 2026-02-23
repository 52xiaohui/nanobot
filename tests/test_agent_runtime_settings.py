from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.factory import ProviderCreateError


class RecordingProvider(LLMProvider):
    def __init__(self, default_model: str, scripted: list[LLMResponse] | None = None) -> None:
        super().__init__(api_key=None, api_base=None)
        self._default_model = default_model
        self._scripted = list(scripted or [])
        self.calls: list[dict] = []

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        request_options: dict | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "request_options": dict(request_options) if request_options else None,
            }
        )
        if self._scripted:
            return self._scripted.pop(0)
        return LLMResponse(content=f"ok:{model or self._default_model}")

    def get_default_model(self) -> str:
        return self._default_model


class StubFactory:
    def __init__(self, mapping: dict[str, RecordingProvider]) -> None:
        self.mapping = mapping

    def create(self, config: Config, model: str) -> RecordingProvider:
        if model not in self.mapping:
            raise ProviderCreateError(f"unknown model: {model}")
        return self.mapping[model]


def _msg(content: str) -> InboundMessage:
    return InboundMessage(
        channel="cli",
        sender_id="user",
        chat_id="direct",
        content=content,
        metadata={},
    )


@pytest.mark.asyncio
async def test_think_runtime_applies_request_options(tmp_path: Path) -> None:
    config = Config()
    config.agents.defaults.model = "model-a"

    provider_a = RecordingProvider(default_model="model-a")
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider_a,
        workspace=tmp_path,
        config=config,
        provider_factory=StubFactory({"model-a": provider_a}),
        model="model-a",
    )

    await loop._process_message(_msg("/think high"))
    out = await loop._process_message(_msg("hello"))

    assert out is not None
    assert provider_a.calls[-1]["request_options"] == {"reasoning_effort": "high"}


@pytest.mark.asyncio
async def test_model_command_switches_provider_per_session(tmp_path: Path) -> None:
    config = Config()
    config.agents.defaults.model = "model-a"

    provider_a = RecordingProvider(default_model="model-a")
    provider_b = RecordingProvider(default_model="model-b")
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider_a,
        workspace=tmp_path,
        config=config,
        provider_factory=StubFactory({"model-a": provider_a, "model-b": provider_b}),
        model="model-a",
    )

    result = await loop._process_message(_msg("/model model-b"))
    assert result is not None
    assert "Model updated:" in result.content

    out = await loop._process_message(_msg("use switched model"))
    assert out is not None
    assert provider_b.calls, "model-b provider should have been used"
    assert provider_b.calls[-1]["model"] == "model-b"
    assert not provider_a.calls, "default provider should not receive chat calls after switch"


@pytest.mark.asyncio
async def test_spawn_receives_runtime_snapshot(tmp_path: Path) -> None:
    config = Config()
    config.agents.defaults.model = "model-a"

    provider_a = RecordingProvider(default_model="model-a")
    provider_b = RecordingProvider(
        default_model="model-b",
        scripted=[
            LLMResponse(
                content="starting",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="spawn",
                        arguments={"task": "do thing", "label": "bg"},
                    )
                ],
            ),
            LLMResponse(content="done"),
        ],
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider_a,
        workspace=tmp_path,
        config=config,
        provider_factory=StubFactory({"model-a": provider_a, "model-b": provider_b}),
        model="model-a",
    )
    loop.subagents.spawn = AsyncMock(return_value="spawned")

    await loop._process_message(_msg("/model model-b"))
    await loop._process_message(_msg("/think high"))
    out = await loop._process_message(_msg("run task in background"))

    assert out is not None
    assert out.content == "done"
    assert loop.subagents.spawn.await_count == 1
    kwargs = loop.subagents.spawn.await_args.kwargs
    assert kwargs["provider"] is provider_b
    assert kwargs["model"] == "model-b"
    assert kwargs["request_options"] == {"reasoning_effort": "high"}

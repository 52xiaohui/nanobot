"""Slash command router."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Awaitable, Callable

from nanobot.agent.commands.types import (
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORT_LEVELS,
    RuntimeSettings,
    clear_runtime_setting,
    get_runtime_setting,
    set_runtime_setting,
)
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.providers.factory import ProviderCreateError
from nanobot.session.manager import Session

if TYPE_CHECKING:
    from nanobot.config.schema import Config
    from nanobot.providers.factory import ProviderFactory
    from nanobot.session.manager import SessionManager


FULL_HELP_TEXT = (
    "nanobot commands:\n"
    "/new - Start a new conversation (keeps /model and /think overrides)\n"
    "/model - Show current session model and provider\n"
    "/model <model_name> - Set current session model\n"
    "/model reset - Clear session model override\n"
    "/think - Show current session reasoning effort\n"
    "/think <low|medium|high> - Set current session reasoning effort\n"
    "/think reset - Clear session reasoning override\n"
    "/help - Show this help"
)


PUBLIC_HELP_TEXT = (
    "nanobot is online.\n"
    "Public command:\n"
    "/help - Show this help\n"
    "If you need full access, contact the bot administrator."
)


def render_public_help() -> str:
    """Public help text for users outside channel allowlist."""
    return PUBLIC_HELP_TEXT


def _source_label(source: str) -> str:
    if source == "session":
        return "session override"
    if source == "config":
        return "config default"
    return "provider default"


@dataclass(frozen=True)
class ParsedCommand:
    """Parsed slash command."""

    name: str
    arg: str


class SlashCommandRouter:
    """Parse and execute slash commands."""

    def __init__(
        self,
        *,
        config: "Config",
        sessions: "SessionManager",
        provider_factory: "ProviderFactory",
        resolve_runtime_settings: Callable[[Session], RuntimeSettings],
        handle_new: Callable[[InboundMessage, Session], Awaitable[OutboundMessage]],
    ) -> None:
        self._config = config
        self._sessions = sessions
        self._provider_factory = provider_factory
        self._resolve_runtime_settings = resolve_runtime_settings
        self._handle_new = handle_new

    async def handle(self, msg: InboundMessage, session: Session) -> OutboundMessage | None:
        """Handle slash command. Return None when input is not a slash command."""
        parsed = self._parse(msg.content)
        if parsed is None:
            return None

        if parsed.name == "help":
            return self._outbound(msg, FULL_HELP_TEXT)
        if parsed.name == "new":
            return await self._handle_new(msg, session)
        if parsed.name == "model":
            return await self._handle_model(msg, session, parsed.arg)
        if parsed.name == "think":
            return self._handle_think(msg, session, parsed.arg)

        return self._outbound(
            msg,
            "Unknown command. Use /help to view available commands.",
        )

    @staticmethod
    def _parse(content: str) -> ParsedCommand | None:
        text = content.strip()
        if not text.startswith("/"):
            return None
        parts = text[1:].split(None, 1)
        if not parts:
            return None
        command = parts[0].split("@", 1)[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        return ParsedCommand(name=command, arg=arg)

    @staticmethod
    def _outbound(msg: InboundMessage, content: str) -> OutboundMessage:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=content,
            metadata=msg.metadata or {},
        )

    def _save_session(self, session: Session) -> None:
        session.updated_at = datetime.now()
        self._sessions.save(session)

    async def _handle_model(self, msg: InboundMessage, session: Session, arg: str) -> OutboundMessage:
        if not arg:
            settings = self._resolve_runtime_settings(session)
            return self._outbound(
                msg,
                (
                    f"Current model: {settings.effective_model}\n"
                    f"Provider: {settings.provider.__class__.__name__}\n"
                    f"Source: {_source_label(settings.model_source)}"
                ),
            )

        arg_lower = arg.lower()
        if arg_lower == "reset":
            clear_runtime_setting(session, "model")
            self._save_session(session)
            settings = self._resolve_runtime_settings(session)
            return self._outbound(
                msg,
                (
                    f"Model override cleared.\n"
                    f"Current model: {settings.effective_model}\n"
                    f"Provider: {settings.provider.__class__.__name__}\n"
                    f"Source: {_source_label(settings.model_source)}"
                ),
            )

        model_name = arg.strip()
        if not model_name:
            return self._outbound(msg, "Usage: /model <model_name> or /model reset")

        previous = self._resolve_runtime_settings(session)
        try:
            validated_provider = self._provider_factory.create(self._config, model_name)
        except ProviderCreateError as e:
            return self._outbound(msg, f"Failed to set model '{model_name}': {e}")

        set_runtime_setting(session, "model", model_name)
        self._save_session(session)
        return self._outbound(
            msg,
            (
                f"Model updated: {previous.effective_model} -> {model_name}\n"
                f"Provider: {validated_provider.__class__.__name__}"
            ),
        )

    def _handle_think(self, msg: InboundMessage, session: Session, arg: str) -> OutboundMessage:
        if not arg:
            value = get_runtime_setting(session, "reasoning_effort")
            if isinstance(value, str) and value in REASONING_EFFORT_LEVELS:
                return self._outbound(msg, f"Current thinking effort: {value} (session override)")
            return self._outbound(msg, f"Current thinking effort: {DEFAULT_REASONING_EFFORT} (default)")

        value = arg.lower().strip()
        if value == "reset":
            clear_runtime_setting(session, "reasoning_effort")
            self._save_session(session)
            return self._outbound(
                msg,
                f"Thinking effort override cleared. Current value: {DEFAULT_REASONING_EFFORT} (default)",
            )

        if value not in REASONING_EFFORT_LEVELS:
            levels = "|".join(REASONING_EFFORT_LEVELS)
            return self._outbound(
                msg,
                f"Invalid thinking effort '{arg}'. Allowed values: {levels}",
            )

        set_runtime_setting(session, "reasoning_effort", value)
        self._save_session(session)
        return self._outbound(msg, f"Thinking effort set to {value}.")

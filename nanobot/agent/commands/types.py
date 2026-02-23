"""Types and helpers for slash commands and session runtime settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session

REASONING_EFFORT_LEVELS = ("low", "medium", "high")
DEFAULT_REASONING_EFFORT = "medium"


@dataclass
class RuntimeSettings:
    """Resolved runtime settings used for a single model request."""

    effective_model: str
    model_source: str  # session | config | provider
    provider: LLMProvider
    request_options: dict[str, Any] | None = None


def get_runtime_setting(session: Session, key: str, default: Any = None) -> Any:
    """Read a runtime setting from session metadata."""
    runtime = session.metadata.get("runtime")
    if not isinstance(runtime, dict):
        return default
    return runtime.get(key, default)


def set_runtime_setting(session: Session, key: str, value: Any) -> None:
    """Write a runtime setting in session metadata."""
    runtime = session.metadata.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        session.metadata["runtime"] = runtime
    runtime[key] = value


def clear_runtime_setting(session: Session, key: str) -> None:
    """Clear a runtime setting and prune empty runtime namespace."""
    runtime = session.metadata.get("runtime")
    if not isinstance(runtime, dict):
        return
    runtime.pop(key, None)
    if not runtime:
        session.metadata.pop("runtime", None)


def resolve_effective_model(
    session: Session,
    config_default_model: str,
    provider_default_model: str,
) -> tuple[str, str]:
    """Resolve effective model and its source priority."""
    runtime_model = get_runtime_setting(session, "model")
    if isinstance(runtime_model, str) and runtime_model.strip():
        return runtime_model.strip(), "session"
    if config_default_model:
        return config_default_model, "config"
    return provider_default_model, "provider"


def build_request_options(session: Session) -> dict[str, Any] | None:
    """Build provider request options from session runtime metadata."""
    reasoning_effort = get_runtime_setting(session, "reasoning_effort")
    if isinstance(reasoning_effort, str):
        value = reasoning_effort.lower().strip()
        if value in REASONING_EFFORT_LEVELS:
            return {"reasoning_effort": value}
    return None

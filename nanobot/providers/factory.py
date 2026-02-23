"""Provider creation and caching."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from nanobot.providers.base import LLMProvider
from nanobot.providers.custom_provider import CustomProvider
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.openai_codex_provider import OpenAICodexProvider
from nanobot.providers.openai_responses_provider import OpenAIResponsesProvider
from nanobot.providers.registry import find_by_name

if TYPE_CHECKING:
    from nanobot.config.schema import Config


class ProviderCreateError(RuntimeError):
    """Raised when a provider cannot be constructed for a target model."""


class ProviderFactory:
    """Create providers from config with instance reuse."""

    def __init__(self) -> None:
        self._cache: dict[tuple, LLMProvider] = {}

    def create(self, config: "Config", model: str) -> LLMProvider:
        """Create (or reuse) a provider for the given model."""
        selected_model = (model or "").strip()
        if not selected_model:
            raise ProviderCreateError("Model cannot be empty.")

        model_lower = selected_model.lower()

        # OpenAI Responses API (API key based)
        if model_lower.startswith("openai-responses/") or model_lower.startswith("openai_responses/"):
            openai_cfg = config.providers.openai
            if not openai_cfg.api_key:
                raise ProviderCreateError(
                    "No OpenAI API key configured. Set providers.openai.apiKey in ~/.nanobot/config.json."
                )
            api_base = openai_cfg.api_base
            key = ("OpenAIResponsesProvider", "openai", openai_cfg.api_key, api_base, selected_model)
            return self._cache_get_or_create(
                key,
                lambda: OpenAIResponsesProvider(
                    api_key=openai_cfg.api_key,
                    api_base=api_base,
                    default_model=selected_model,
                ),
            )

        provider_name = self._resolve_provider_name(config, selected_model)
        p = config.get_provider(selected_model)

        # OpenAI Codex (OAuth)
        if (
            provider_name == "openai_codex"
            or model_lower.startswith("openai-codex/")
            or model_lower.startswith("openai_codex/")
        ):
            key = ("OpenAICodexProvider", "openai_codex", selected_model)
            return self._cache_get_or_create(
                key,
                lambda: OpenAICodexProvider(default_model=selected_model),
            )

        # Custom OpenAI-compatible endpoint (bypasses LiteLLM)
        if provider_name == "custom":
            api_base = config.get_api_base(selected_model) or "http://localhost:8000/v1"
            api_key = p.api_key if p else "no-key"
            key = ("CustomProvider", "custom", api_key, api_base, selected_model)
            return self._cache_get_or_create(
                key,
                lambda: CustomProvider(
                    api_key=api_key,
                    api_base=api_base,
                    default_model=selected_model,
                ),
            )

        spec = find_by_name(provider_name) if provider_name else None
        if not selected_model.startswith("bedrock/") and not (p and p.api_key) and not (spec and spec.is_oauth):
            raise ProviderCreateError(
                f"No API key configured for model '{selected_model}'. "
                "Set provider credentials in ~/.nanobot/config.json."
            )

        api_key = p.api_key if p else None
        api_base = config.get_api_base(selected_model)
        extra_headers = p.extra_headers if p else None
        cache_headers = json.dumps(extra_headers, ensure_ascii=True, sort_keys=True) if extra_headers else None
        key = ("LiteLLMProvider", provider_name, api_key, api_base, selected_model, cache_headers)
        return self._cache_get_or_create(
            key,
            lambda: LiteLLMProvider(
                api_key=api_key,
                api_base=api_base,
                default_model=selected_model,
                extra_headers=extra_headers,
                provider_name=provider_name,
            ),
        )

    @staticmethod
    def _resolve_provider_name(config: "Config", model: str) -> str | None:
        model_lower = model.lower()
        if model_lower.startswith("github-copilot/") or model_lower.startswith("github_copilot/"):
            return "github_copilot"
        return config.get_provider_name(model)

    def _cache_get_or_create(self, key: tuple, builder) -> LLMProvider:
        if key in self._cache:
            return self._cache[key]
        provider = builder()
        self._cache[key] = provider
        return provider

import pytest

from nanobot.config.schema import Config
from nanobot.providers.factory import ProviderCreateError, ProviderFactory
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.openai_responses_provider import OpenAIResponsesProvider


def test_factory_creates_openai_responses_provider() -> None:
    config = Config()
    config.providers.openai.api_key = "sk-test"

    factory = ProviderFactory()
    provider = factory.create(config, "openai-responses/gpt-5")

    assert isinstance(provider, OpenAIResponsesProvider)


def test_factory_reuses_cached_provider_instances() -> None:
    config = Config()
    config.providers.openai.api_key = "sk-test"

    factory = ProviderFactory()
    p1 = factory.create(config, "openai-responses/gpt-5")
    p2 = factory.create(config, "openai-responses/gpt-5")

    assert p1 is p2


def test_factory_raises_when_required_api_key_missing() -> None:
    config = Config()
    factory = ProviderFactory()

    with pytest.raises(ProviderCreateError):
        factory.create(config, "openai-responses/gpt-5")


def test_factory_routes_github_copilot_through_litellm() -> None:
    config = Config()
    factory = ProviderFactory()

    provider = factory.create(config, "github-copilot/gpt-5.3-codex")

    assert isinstance(provider, LiteLLMProvider)

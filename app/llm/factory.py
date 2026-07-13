"""
LLM Provider Factory

Reads LLM_PROVIDER from settings and returns the correct provider.
To switch from Gemini to Bedrock: change LLM_PROVIDER=bedrock in .env
"""
from app.llm.base import LLMProvider
from app.config.settings import get_settings
import structlog

logger = structlog.get_logger()


def create_llm_provider() -> LLMProvider:
    """Factory function that creates the configured LLM provider."""
    settings = get_settings()
    provider_name = settings.llm_provider

    logger.info("Creating LLM provider", provider=provider_name)

    if provider_name == "gemini":
        from app.llm.providers.gemini_provider import GeminiProvider
        return GeminiProvider()

    elif provider_name == "bedrock":
        from app.llm.providers.bedrock_provider import BedrockProvider
        return BedrockProvider()
    elif provider_name == "openai":
        from app.llm.providers.local_provider import OpenAIProvider
        return OpenAIProvider()
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Supported: ['gemini', 'bedrock', 'openai']"
        )
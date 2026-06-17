from pi_mono.ai.api_registry import register_api_provider
from pi_mono.ai.providers.mistral import stream_mistral, stream_simple_mistral
from pi_mono.ai.providers.anthropic import stream_anthropic, stream_simple_anthropic
from pi_mono.ai.providers.openai_completions import (
    stream_openai_completions,
    stream_simple_openai_completions,
)
from pi_mono.ai.providers.openai_responses import (
    stream_openai_responses,
    stream_simple_openai_responses,
)
from pi_mono.ai.providers.azure_openai_responses import (
    stream_azure_openai_responses,
    stream_simple_azure_openai_responses,
)
from pi_mono.ai.providers.google import stream_google, stream_simple_google
from pi_mono.ai.providers.google_vertex import (
    stream_google_vertex,
    stream_simple_google_vertex,
)
from pi_mono.ai.providers.amazon_bedrock import stream_bedrock, stream_simple_bedrock
from pi_mono.ai.providers.openai_codex_responses import (
    stream_openai_codex_responses,
    stream_simple_openai_codex_responses,
)


class AnthropicProviderRegistration:
    api = "anthropic-messages"

    def stream(self, model, context, options=None):
        return stream_anthropic(model, context, options)

    def stream_simple(self, model, context, options=None):
        return stream_simple_anthropic(model, context, options)


class OpenAICompletionsProviderRegistration:
    api = "openai-completions"

    def stream(self, model, context, options=None):
        return stream_openai_completions(model, context, options)

    def stream_simple(self, model, context, options=None):
        return stream_simple_openai_completions(model, context, options)


class MistralProviderRegistration:
    api = "mistral-conversations"

    def stream(self, model, context, options=None):
        return stream_mistral(model, context, options)

    def stream_simple(self, model, context, options=None):
        return stream_simple_mistral(model, context, options)


class OpenAIResponsesProviderRegistration:
    api = "openai-responses"

    def stream(self, model, context, options=None):
        return stream_openai_responses(model, context, options)

    def stream_simple(self, model, context, options=None):
        return stream_simple_openai_responses(model, context, options)


class AzureOpenAIResponsesProviderRegistration:
    api = "azure-openai-responses"

    def stream(self, model, context, options=None):
        return stream_azure_openai_responses(model, context, options)

    def stream_simple(self, model, context, options=None):
        return stream_simple_azure_openai_responses(model, context, options)


class GoogleProviderRegistration:
    api = "google-generative-ai"

    def stream(self, model, context, options=None):
        return stream_google(model, context, options)

    def stream_simple(self, model, context, options=None):
        return stream_simple_google(model, context, options)


class GoogleVertexProviderRegistration:
    api = "google-vertex"

    def stream(self, model, context, options=None):
        return stream_google_vertex(model, context, options)

    def stream_simple(self, model, context, options=None):
        return stream_simple_google_vertex(model, context, options)


class BedrockProviderRegistration:
    api = "bedrock-converse-stream"

    def stream(self, model, context, options=None):
        return stream_bedrock(model, context, options)

    def stream_simple(self, model, context, options=None):
        return stream_simple_bedrock(model, context, options)


class OpenAICodexResponsesProviderRegistration:
    api = "openai-codex-responses"

    def stream(self, model, context, options=None):
        return stream_openai_codex_responses(model, context, options)

    def stream_simple(self, model, context, options=None):
        return stream_simple_openai_codex_responses(model, context, options)


def reset_api_providers() -> None:
    """Clear and re-register built-in API providers."""
    from pi_mono.ai.api_registry import clear_api_providers

    clear_api_providers()
    register_built_in_api_providers()


def register_built_in_api_providers() -> None:
    """Register built-in API providers."""
    register_api_provider(AnthropicProviderRegistration())  # type: ignore
    register_api_provider(OpenAICompletionsProviderRegistration())  # type: ignore
    register_api_provider(MistralProviderRegistration())  # type: ignore
    register_api_provider(OpenAIResponsesProviderRegistration())  # type: ignore
    register_api_provider(AzureOpenAIResponsesProviderRegistration())  # type: ignore
    register_api_provider(GoogleProviderRegistration())  # type: ignore
    register_api_provider(GoogleVertexProviderRegistration())  # type: ignore
    register_api_provider(BedrockProviderRegistration())  # type: ignore
    register_api_provider(OpenAICodexResponsesProviderRegistration())  # type: ignore


# Automatically register on import, matching TypeScript side-effects loading
register_built_in_api_providers()

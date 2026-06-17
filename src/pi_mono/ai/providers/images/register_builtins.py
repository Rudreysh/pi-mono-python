from pi_mono.ai.images_api_registry import register_images_api_provider
from pi_mono.ai.providers.images.openrouter import generate_images_openrouter


class OpenRouterImagesProviderRegistration:
    api = "openrouter-images"

    async def generate_images(self, model, context, options=None):
        return await generate_images_openrouter(model, context, options)


def register_built_in_images_api_providers() -> None:
    """Register built-in image API providers."""
    register_images_api_provider(OpenRouterImagesProviderRegistration())  # type: ignore


# Automatically register on import
register_built_in_images_api_providers()

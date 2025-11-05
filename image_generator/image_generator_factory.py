"""
image_generator_factory.py

Factory for creating image generators with different backends.
Supports local Stable Diffusion (via diffusers) and API-based services.
"""

from typing import Protocol, Dict, Any, List, Literal, Optional


class ImageGenerator(Protocol):
    """
    Protocol defining the interface for image generators.

    All image generator implementations must provide the generate_images method.
    """

    def generate_images(
        self,
        prompts_file: str,
        output_dir: str = "generated_images",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images from prompts file.

        Args:
            prompts_file: Path to JSON file with image prompts
            output_dir: Directory to save generated images
            **kwargs: Additional backend-specific parameters

        Returns:
            Dictionary containing:
                - images: List of generated image paths
                - prompts_file: Original prompts file path
                - num_images: Number of images generated
                - metadata: Additional generation metadata
        """
        ...


# Supported backends
ImageBackend = Literal["diffusers", "openai", "stability"]


def get_generator(
    backend: ImageBackend = "diffusers",
    model: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
) -> ImageGenerator:
    """
    Factory function to create image generators.

    Args:
        backend: Which backend to use (diffusers, openai, stability)
        model: Model name/path (backend-specific)
        device: Device to use (cuda, mps, cpu) - only for local backends
        **kwargs: Additional backend-specific parameters

    Returns:
        ImageGenerator instance

    Examples:
        # Local Stable Diffusion (default)
        generator = get_generator("diffusers")

        # OpenAI DALL-E
        generator = get_generator("openai", model="dall-e-3")

        # Stability AI
        generator = get_generator("stability", model="sd3-large")
    """
    if backend == "diffusers":
        from .diffusers_generator import DiffusersGenerator
        return DiffusersGenerator(model=model, device=device, **kwargs)
    elif backend == "openai":
        from .openai_generator import OpenAIImageGenerator
        return OpenAIImageGenerator(model=model, **kwargs)
    elif backend == "stability":
        from .stability_generator import StabilityGenerator
        return StabilityGenerator(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: diffusers, openai, stability")

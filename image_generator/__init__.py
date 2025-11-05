"""
image_generator

Generate images from text prompts using various backends:
- diffusers: Local Stable Diffusion (free, requires GPU)
- openai: DALL-E API (paid)
- stability: Stability AI API (paid)

Usage:
    from image_generator import get_generator

    # Local Stable Diffusion (default)
    generator = get_generator("diffusers")
    result = generator.generate_images("prompts.json")

    # OpenAI DALL-E
    generator = get_generator("openai", model="dall-e-3")
    result = generator.generate_images("prompts.json")

    # Stability AI
    generator = get_generator("stability", model="sd3-large")
    result = generator.generate_images("prompts.json")
"""

from .image_generator_factory import (
    get_generator,
    ImageGenerator,
    ImageBackend,
)

__all__ = [
    "get_generator",
    "ImageGenerator",
    "ImageBackend",
]

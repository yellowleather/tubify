"""
image_prompt_generator_factory.py

Factory for creating image prompt generators with different LLM backends.
"""

from typing import Protocol, Dict, Any, List, Literal, Optional


class ImagePromptGenerator(Protocol):
    """
    Protocol for image prompt generators.

    Implementations should analyze transcripts and generate image prompts
    for each semantic section of the content.
    """

    def generate_prompts(
        self,
        transcript_file: str,
        output_dir: str = "image_prompts"
    ) -> Dict[str, Any]:
        """
        Generate image prompts from transcript file.

        Args:
            transcript_file: Path to aligned JSON transcript from transcriber
            output_dir: Directory to save generated prompts

        Returns:
            Dictionary with:
              - prompts_file: Path to generated prompts JSON file
              - sections: List of section objects with prompts
              - video_name: Name of the video
        """
        ...


# Supported LLM backends
LLMBackend = Literal["openai", "anthropic", "ollama"]


def get_generator(
    backend: LLMBackend = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> ImagePromptGenerator:
    """
    Factory function to get the appropriate image prompt generator.

    Args:
        backend: LLM backend to use ("openai", "anthropic", "ollama")
        model: Model name (e.g., "gpt-4", "claude-3-5-sonnet", "llama3")
        api_key: API key for the service (optional, can use env var)
        **kwargs: Additional backend-specific parameters

    Returns:
        An ImagePromptGenerator instance for the specified backend

    Raises:
        ValueError: If the backend is not supported

    Examples:
        # OpenAI GPT-4
        generator = get_generator("openai", model="gpt-4")

        # Anthropic Claude
        generator = get_generator("anthropic", model="claude-3-5-sonnet-20241022")

        # Local Ollama
        generator = get_generator("ollama", model="llama3")
    """
    if backend == "openai":
        from .openai_generator import OpenAIGenerator
        return OpenAIGenerator(model=model, api_key=api_key, **kwargs)

    elif backend == "anthropic":
        from .anthropic_generator import AnthropicGenerator
        return AnthropicGenerator(model=model, api_key=api_key, **kwargs)

    elif backend == "ollama":
        from .ollama_generator import OllamaGenerator
        return OllamaGenerator(model=model, **kwargs)

    else:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Supported backends: openai, anthropic, ollama"
        )

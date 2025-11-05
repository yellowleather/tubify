"""
model_downloader_factory.py

Factory for creating model downloaders for different platforms.
Currently supports HuggingFace Hub.
"""

from typing import Dict, List, Literal, Optional, Protocol

from .huggingface_downloader import HuggingFaceDownloader


class ModelDownloader(Protocol):
    """Protocol for model downloader implementations."""

    def download(
        self,
        models: Optional[List[str]] = None,
        force: bool = False,
        skip_space_check: bool = False
    ) -> Dict[str, List[str]]:
        """
        Download one or more models.

        Args:
            models: List of model names to download (None = all models)
            force: Force fresh downloads even if files exist
            skip_space_check: Skip disk space validation

        Returns:
            Dictionary with 'successful' and 'failed' lists of model names
        """
        ...


PlatformType = Literal["huggingface"]


def get_downloader(platform: PlatformType = "huggingface", models_dir: str = "models") -> ModelDownloader:
    """
    Factory function to get the appropriate model downloader.

    Args:
        platform: The model platform ("huggingface" currently supported)
        models_dir: Directory to store downloaded models

    Returns:
        A ModelDownloader instance for the specified platform

    Raises:
        ValueError: If the platform is not supported

    Examples:
        >>> downloader = get_downloader("huggingface")
        >>> downloader.download(models=["tiny", "base"])
    """
    if platform == "huggingface":
        return HuggingFaceDownloader(models_dir=models_dir)
    else:
        raise ValueError(
            f"Unsupported platform: {platform}. "
            f"Supported platforms: huggingface"
        )

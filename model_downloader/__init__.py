"""
model_downloader package

Factory-based model downloader supporting multiple platforms.
"""

from .model_downloader_factory import (
    get_downloader,
    ModelDownloader,
    PlatformType,
)
from .huggingface_downloader import HuggingFaceDownloader, WHISPER_MODELS

__all__ = [
    "get_downloader",
    "HuggingFaceDownloader",
    "ModelDownloader",
    "PlatformType",
    "WHISPER_MODELS",
]

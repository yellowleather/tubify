"""
video_downloader package

Factory-based video downloader supporting multiple platforms.
"""

from .video_downloader_factory import (
    get_downloader,
    VideoDownloader,
    PlatformType,
)
from .youtube_downloader import YouTubeDownloader

__all__ = [
    "get_downloader",
    "YouTubeDownloader",
    "VideoDownloader",
    "PlatformType",
]

"""
video_downloader_factory.py

Factory for creating video downloaders for different platforms.
Currently supports YouTube (including Shorts).
"""

from typing import Literal, Protocol

from .youtube_downloader import YouTubeDownloader


class VideoDownloader(Protocol):
    """Protocol for video downloader implementations."""

    def download(self, url: str, output_dir: str) -> int:
        """
        Download a video from the given URL to output_dir.
        Returns 0 on success, non-zero on failure.
        """
        ...


PlatformType = Literal["youtube"]


def get_downloader(platform: PlatformType = "youtube") -> VideoDownloader:
    """
    Factory function to get the appropriate video downloader.

    Args:
        platform: The video platform ("youtube" currently supported)

    Returns:
        A VideoDownloader instance for the specified platform

    Raises:
        ValueError: If the platform is not supported

    Examples:
        >>> downloader = get_downloader("youtube")
        >>> downloader.download("https://www.youtube.com/watch?v=...", "downloads/")
    """
    if platform == "youtube":
        return YouTubeDownloader()
    else:
        raise ValueError(
            f"Unsupported platform: {platform}. "
            f"Supported platforms: youtube"
        )

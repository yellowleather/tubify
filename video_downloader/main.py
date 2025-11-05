#!/usr/bin/env -S uv run python
"""
main.py

CLI entry point for video downloader.
Downloads videos from various platforms (YouTube, etc.) via command line.
"""

import argparse
import sys
import os

# Add parent directory to path for imports when run as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from video_downloader.video_downloader_factory import get_downloader, PlatformType
except ImportError:
    from video_downloader_factory import get_downloader, PlatformType


def download_video(url: str, output_dir: str = "inputs/", platform: PlatformType = "youtube") -> int:
    """
    Download a video from the specified platform.

    Args:
        url: Video URL
        output_dir: Directory to save the downloaded file
        platform: Video platform (default: "youtube")

    Returns:
        0 on success, non-zero on failure
    """
    downloader = get_downloader(platform)
    return downloader.download(url, output_dir)


def main() -> int:
    """
    Main CLI entry point for video downloader.

    Returns:
        Exit code: 0 on success, non-zero on failure
    """
    parser = argparse.ArgumentParser(
        description="Download videos from various platforms using the factory pattern.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a YouTube video
  python main.py --url "https://www.youtube.com/watch?v=..."

  # Download a YouTube Short
  python main.py --url "https://www.youtube.com/shorts/..." --output_dir downloads/

  # Specify platform explicitly
  python main.py --url "https://www.youtube.com/watch?v=..." --platform youtube
        """
    )

    parser.add_argument(
        "--url",
        required=True,
        help="Video URL to download"
    )
    parser.add_argument(
        "--output_dir",
        default="inputs/",
        help="Directory to save the downloaded file (default: inputs/)"
    )
    parser.add_argument(
        "--platform",
        default="youtube",
        choices=["youtube"],
        help="Video platform (default: youtube)"
    )

    args = parser.parse_args()

    exit_code = download_video(args.url, args.output_dir, args.platform)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

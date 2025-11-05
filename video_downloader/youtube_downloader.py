"""
youtube_downloader.py

YouTube video downloader implementation using yt-dlp.

Downloads YouTube videos (including Shorts) to a target directory.
- Normalizes /shorts URLs to regular watch URLs automatically.
- Uses a robust yt-dlp configuration that works well for Shorts:
  * Android player client (no cookies)
  * IPv4 only (avoids some CDN issues)
  * Progressive MP4 first (itag 18), then 1080p+audio, then best fallback

Requirements:
  - Python 3.8+
  - yt-dlp >= 2024.10 (recommended latest)
  - ffmpeg in PATH (for merging DASH formats)

Note: This is a library module. Use main.py for CLI access.
"""

import os
import re
import sys
from urllib.parse import urlparse, parse_qs

try:
    import yt_dlp  # type: ignore
except ImportError as e:
    print("Error: yt-dlp is not installed. Install with:\n  pip install -U yt-dlp", file=sys.stderr)
    sys.exit(1)


# Constants
ANDROID_UA = "com.google.android.youtube/19.39.37 (Linux; U; Android 13)"
MOBILE_REF = "https://m.youtube.com/"
# Format selection priority:
# 1) 18: progressive MP4 (video+audio) – most reliable for Shorts
# 2) 137+140: 1080p video + m4a audio
# 3) bv*+ba: best separate video+audio
# 4) b: best overall fallback
FORMAT_SELECTOR = "18/137+140/bv*+ba/b"


# Helper functions
def extract_video_id_from_shorts(url: str) -> str | None:
    """
    Extract the video ID from a /shorts/<ID> URL.
    Returns the ID as a string if matched, else None.
    """
    # Patterns like:
    # https://www.youtube.com/shorts/QMJAUg2snas
    # https://youtube.com/shorts/QMJAUg2snas?feature=share
    m = re.search(r"(?:youtube\.com|youtu\.be)/shorts/([A-Za-z0-9_-]{5,})", url)
    if m:
        return m.group(1)
    return None


def normalize_youtube_url(url: str) -> str:
    """
    Convert Shorts URLs to standard watch URLs.
    Leaves watch URLs unchanged. Pass-through for already-normalized forms.
    """
    # If it's a shorts URL, replace with watch?v=ID
    vid_from_shorts = extract_video_id_from_shorts(url)
    if vid_from_shorts:
        return f"https://www.youtube.com/watch?v={vid_from_shorts}"

    # If it's youtu.be/<ID>, convert to watch URL
    parsed = urlparse(url)
    if parsed.netloc in {"youtu.be"}:
        video_id = parsed.path.lstrip("/")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"

    # If it's already a watch URL with ?v=
    if "youtube.com/watch" in url:
        return url

    # Try to salvage unknown forms by pulling v= param if any
    qs = parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        return f"https://www.youtube.com/watch?v={qs['v'][0]}"

    # Otherwise return original
    return url


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


class YouTubeDownloader:
    """
    YouTube video downloader using yt-dlp.

    Implements the VideoDownloader protocol with a download() method.
    Handles YouTube videos and Shorts with optimized settings.
    """

    def __init__(self):
        """Initialize the YouTube downloader."""
        # Verify yt-dlp is available (already checked at module level)
        pass

    def download(self, url: str, output_dir: str = "inputs/") -> int:
        """
        Download a YouTube video to the specified directory.
        Skips download if file already exists.

        Args:
            url: YouTube video URL (supports /shorts/ links)
            output_dir: Directory to save the downloaded file

        Returns:
            0 on success, non-zero on failure
        """
        # Ensure output directory exists
        ensure_dir(output_dir)

        # Normalize URL (convert shorts to watch URLs)
        norm_url = normalize_youtube_url(url)

        # Output template
        outtmpl = os.path.join(output_dir, "%(title)s.%(ext)s")

        # yt-dlp options optimized for YouTube/Shorts
        ydl_opts: dict = {
            # Network hardening
            "force_ipv4": True,
            "quiet": False,
            "no_warnings": False,
            "nocheckcertificate": False,

            # Player/headers: impersonate Android client (no cookies needed)
            "http_headers": {
                "User-Agent": ANDROID_UA,
                "Referer": MOBILE_REF,
            },
            "extractor_args": {"youtube": {"player_client": ["android"]}},

            # Formats and merging
            "format": FORMAT_SELECTOR,
            "merge_output_format": "mp4",  # produce mp4 when merging DASH

            # Output
            "outtmpl": outtmpl,
            "restrictfilenames": True,
            "writethumbnail": False,

            # Caching
            "cachedir": False,   # similar effect to --rm-cache-dir per-run
        }

        try:
            # First, extract video info to check if file already exists
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(norm_url, download=False)
                expected_filename = ydl.prepare_filename(info)

                # Check if file already exists
                if os.path.exists(expected_filename):
                    print(f"[info] File already exists: {expected_filename}")
                    print(f"[info] Skipping download")
                    return 0

                # File doesn't exist, proceed with download
                print(f"[info] Downloading:\n  URL: {norm_url}\n  Output dir: {output_dir}\n  Format: {FORMAT_SELECTOR}")
                result = ydl.download([norm_url])
                return 0 if result == 0 else 2

        except yt_dlp.utils.DownloadError as e:
            # Commonly 403s or restricted content errors
            print(f"[error] Download failed: {e}", file=sys.stderr)
            return 3
        except Exception as e:
            print(f"[error] Unexpected failure: {e}", file=sys.stderr)
            return 4

# Video Downloader Package

Factory-based video downloader supporting multiple platforms (currently YouTube).

## Features

- Download YouTube videos and Shorts
- Automatic URL normalization (converts /shorts/ to /watch URLs)
- Skip download if file already exists
- Optimized for YouTube with Android player client
- Factory pattern for easy extensibility

## Architecture

```
video_downloader/
├── __init__.py                    # Package exports
├── youtube_downloader.py          # YouTubeDownloader class (implementation)
├── video_downloader_factory.py   # Factory pattern
└── main.py                        # CLI entry point
```

## Usage

### 1. CLI Usage (via main.py)

```bash
# Basic download
./video_downloader/main.py --url "https://www.youtube.com/shorts/QMJAUg2snas"

# Custom output directory
./video_downloader/main.py --url "https://www.youtube.com/watch?v=..." --output_dir downloads/

# With platform specification
./video_downloader/main.py --url "..." --platform youtube
```

### 2. Library Usage (Programmatic)

#### Using the Factory Pattern

```python
from video_downloader import get_downloader

# Get a YouTube downloader
downloader = get_downloader("youtube")
result = downloader.download(
    url="https://www.youtube.com/shorts/QMJAUg2snas",
    output_dir="inputs/"
)

if result == 0:
    print("Download successful!")
else:
    print(f"Download failed with code: {result}")
```

#### Direct Class Instantiation

```python
from video_downloader import YouTubeDownloader

# Create downloader instance
yt = YouTubeDownloader()
result = yt.download(
    url="https://www.youtube.com/watch?v=...",
    output_dir="videos/"
)
```

## API Reference

### VideoDownloader Protocol

```python
class VideoDownloader(Protocol):
    def download(self, url: str, output_dir: str) -> int:
        """
        Download a video from the given URL to output_dir.

        Args:
            url: Video URL
            output_dir: Directory to save the file

        Returns:
            0 on success, non-zero on failure
        """
```

### YouTubeDownloader Class

```python
class YouTubeDownloader:
    def __init__(self):
        """Initialize the YouTube downloader."""

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
```

### Factory Function

```python
def get_downloader(platform: PlatformType = "youtube") -> VideoDownloader:
    """
    Factory function to get the appropriate video downloader.

    Args:
        platform: The video platform ("youtube" currently supported)

    Returns:
        A VideoDownloader instance for the specified platform

    Raises:
        ValueError: If the platform is not supported
    """
```

## Return Codes

- `0` - Success
- `2` - Download completed but yt-dlp returned non-zero
- `3` - Download error (403, restricted content, etc.)
- `4` - Unexpected error

## Features in Detail

### URL Normalization

Automatically converts various YouTube URL formats:
- `/shorts/<ID>` → `/watch?v=<ID>`
- `youtu.be/<ID>` → `/watch?v=<ID>`
- Handles query parameters

### Skip Existing Files

Before downloading:
1. Extracts video metadata (no download)
2. Determines expected filename using yt-dlp's naming
3. Checks if file exists
4. Skips download if present, proceeds otherwise

### Android Player Client

Uses Android player client for better compatibility:
- No cookies required
- Bypasses some restrictions
- Works well with Shorts

## Dependencies

- `yt-dlp >= 2024.10` - YouTube video downloading
- `python >= 3.8` - Modern Python features

## Extending to New Platforms

To add a new platform (e.g., Vimeo):

1. Create `vimeo_downloader.py`:
```python
class VimeoDownloader:
    def download(self, url: str, output_dir: str = "inputs/") -> int:
        # Implementation here
        pass
```

2. Update `video_downloader_factory.py`:
```python
from .vimeo_downloader import VimeoDownloader

PlatformType = Literal["youtube", "vimeo"]

def get_downloader(platform: PlatformType = "youtube") -> VideoDownloader:
    if platform == "youtube":
        return YouTubeDownloader()
    elif platform == "vimeo":
        return VimeoDownloader()
    # ...
```

3. Export in `__init__.py`:
```python
from .vimeo_downloader import VimeoDownloader

__all__ = [..., "VimeoDownloader"]
```

## Examples

### Download Multiple Videos

```python
from video_downloader import get_downloader

downloader = get_downloader("youtube")

urls = [
    "https://www.youtube.com/shorts/ABC123",
    "https://www.youtube.com/watch?v=XYZ789",
]

for url in urls:
    result = downloader.download(url, "videos/")
    print(f"Downloaded {url}: {'Success' if result == 0 else 'Failed'}")
```

### Error Handling

```python
from video_downloader import get_downloader

downloader = get_downloader("youtube")
result = downloader.download(url, "outputs/")

if result == 0:
    print("Success!")
elif result == 3:
    print("Download error - may be restricted or deleted")
elif result == 4:
    print("Unexpected error occurred")
```

## License

Part of the Tubify project. See root LICENSE file.

# Model Downloader Package

Factory-based model downloader for Whisper models from HuggingFace Hub.

## Features

- Download Whisper models (tiny to large-v3)
- Automatic skip if model already exists
- Disk space validation before download
- Beautiful progress display with Rich
- HuggingFace cache format compatible with faster-whisper
- Fast downloads via hf_transfer

## Architecture

```
model_downloader/
├── __init__.py                    # Package exports
├── huggingface_downloader.py     # HuggingFaceDownloader class (implementation)
├── model_downloader_factory.py   # Factory pattern
└── main.py                        # CLI entry point
```

## Available Models

| Model | Size | Description |
|-------|------|-------------|
| tiny | ~75MB | Fastest, least accurate |
| base | ~145MB | Good speed/accuracy balance |
| small | ~483MB | Better accuracy |
| medium | ~1.53GB | High accuracy |
| large-v3 | ~3.09GB | Best accuracy (recommended) |

## Usage

### 1. CLI Usage (via main.py)

```bash
# Download all models
./model_downloader/main.py

# Download specific models
./model_downloader/main.py --models tiny base

# Download to custom directory
./model_downloader/main.py --models large-v3 --models-dir /path/to/models

# Force redownload
./model_downloader/main.py --models base --force

# Skip disk space check
./model_downloader/main.py --models medium --skip-space-check
```

### 2. Library Usage (Programmatic)

#### Using the Factory Pattern

```python
from model_downloader import get_downloader

# Get a HuggingFace downloader
downloader = get_downloader("huggingface")

# Download specific models
result = downloader.download(
    models=["tiny", "base"],
    force=False,
    skip_space_check=False
)

print(f"Successful: {result['successful']}")
print(f"Failed: {result['failed']}")
```

#### Direct Class Instantiation

```python
from model_downloader import HuggingFaceDownloader

# Create downloader with custom directory
downloader = HuggingFaceDownloader(models_dir="custom/path")

# Download all models
result = downloader.download()

# Download a single model
path = downloader.download_single_model("large-v3", force=False)
```

#### Get Available Models

```python
from model_downloader import WHISPER_MODELS

# Print all available models
for name, info in WHISPER_MODELS.items():
    print(f"{name}: {info['description']} ({info['size_gb']:.2f}GB)")
```

## API Reference

### ModelDownloader Protocol

```python
class ModelDownloader(Protocol):
    def download(
        self,
        models: Optional[List[str]] = None,
        force: bool = False,
        skip_space_check: bool = False
    ) -> Dict[str, List[str]]:
        """
        Download one or more models.

        Args:
            models: List of model names (None = all models)
            force: Force fresh downloads even if files exist
            skip_space_check: Skip disk space validation

        Returns:
            Dictionary with 'successful' and 'failed' lists
        """
```

### HuggingFaceDownloader Class

```python
class HuggingFaceDownloader:
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the HuggingFace downloader.

        Args:
            models_dir: Directory to store models (default: "models")
        """

    def get_available_models(self) -> Dict[str, Dict]:
        """Get dictionary of available Whisper models."""

    def download_single_model(
        self,
        model_name: str,
        force: bool = False
    ) -> Optional[str]:
        """
        Download a single model.

        Args:
            model_name: Name of model ("tiny", "base", etc.)
            force: Force fresh download even if exists

        Returns:
            Path to downloaded model on success, None on failure
        """

    def download(
        self,
        models: Optional[List[str]] = None,
        force: bool = False,
        skip_space_check: bool = False
    ) -> Dict[str, List[str]]:
        """Download one or more models."""
```

### Factory Function

```python
def get_downloader(
    platform: PlatformType = "huggingface",
    models_dir: str = "models"
) -> ModelDownloader:
    """
    Factory function to get the appropriate model downloader.

    Args:
        platform: The model platform ("huggingface" currently supported)
        models_dir: Directory to store downloaded models

    Returns:
        A ModelDownloader instance for the specified platform

    Raises:
        ValueError: If the platform is not supported
    """
```

## Return Values

The `download()` method returns a dictionary:

```python
{
    "successful": ["tiny", "base"],  # List of successfully downloaded models
    "failed": []                      # List of failed downloads
}
```

## Features in Detail

### Skip Existing Models

Before downloading:
1. Checks for `model.bin` in the cache directory
2. If exists and `force=False`, skips download
3. Reports that model already exists
4. Returns success immediately

### Disk Space Validation

Before downloading multiple models:
1. Calculates total required space
2. Checks available disk space
3. Warns if insufficient (with 1GB buffer)
4. Prompts user to continue or cancel
5. Can be skipped with `skip_space_check=True`

### HuggingFace Cache Format

Models are stored in the format that faster-whisper expects:
```
models/
└── models--Systran--faster-whisper-large-v3/
    ├── model.bin
    ├── config.json
    └── ...
```

### Fast Downloads

Uses `hf_transfer` for accelerated downloads if available:
- Set via `HF_HUB_ENABLE_HF_TRANSFER=1` environment variable
- Automatically enabled in the module
- Falls back to standard download if not available

## Dependencies

- `huggingface_hub >= 0.23` - Model downloads from HuggingFace
- `hf_transfer >= 0.1.4` - Fast downloads (optional but recommended)
- `rich >= 13.7` - Beautiful terminal output
- `python >= 3.8` - Modern Python features

## Environment Variables

The package automatically sets:

```python
os.environ["HF_HOME"] = "./.hf_home"
os.environ["HUGGINGFACE_HUB_CACHE"] = "./.hf_home/hub"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
```

This keeps all caches self-contained within the project.

## Examples

### Download All Models

```python
from model_downloader import get_downloader

downloader = get_downloader("huggingface")
result = downloader.download()  # Downloads all 5 models

if result["successful"]:
    print(f"Downloaded: {', '.join(result['successful'])}")
if result["failed"]:
    print(f"Failed: {', '.join(result['failed'])}")
```

### Download with Progress Tracking

```python
from model_downloader import HuggingFaceDownloader

downloader = HuggingFaceDownloader(models_dir="models")

models_to_download = ["tiny", "base", "large-v3"]
result = downloader.download(
    models=models_to_download,
    force=False,
    skip_space_check=True
)

# Result includes rich table output automatically
success_rate = len(result["successful"]) / len(models_to_download) * 100
print(f"Success rate: {success_rate:.1f}%")
```

### Check Model Availability

```python
from model_downloader import HuggingFaceDownloader
from pathlib import Path

downloader = HuggingFaceDownloader()

model_name = "large-v3"
cache_dir = Path("models") / f"models--Systran--faster-whisper-{model_name}"

if (cache_dir / "model.bin").exists():
    print(f"{model_name} is already downloaded")
else:
    print(f"{model_name} needs to be downloaded")
    downloader.download_single_model(model_name)
```

### Force Redownload

```python
from model_downloader import get_downloader

# Force redownload of specific model
downloader = get_downloader("huggingface")
result = downloader.download(
    models=["base"],
    force=True  # Redownload even if exists
)
```

## Extending to New Platforms

To add a new model source (e.g., local files, S3):

1. Create `local_downloader.py`:
```python
class LocalDownloader:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)

    def download(self, models=None, force=False, skip_space_check=False):
        # Implementation for local copy
        pass
```

2. Update `model_downloader_factory.py`:
```python
from .local_downloader import LocalDownloader

PlatformType = Literal["huggingface", "local"]

def get_downloader(platform: PlatformType = "huggingface", models_dir: str = "models"):
    if platform == "huggingface":
        return HuggingFaceDownloader(models_dir)
    elif platform == "local":
        return LocalDownloader(models_dir)
```

## Integration with Tubify

Used in the main Tubify pipeline:

```python
from model_downloader import get_downloader

# Download models before transcription
model_downloader = get_downloader("huggingface", models_dir="models")
result = model_downloader.download(
    models=["large-v3"],
    force=False,
    skip_space_check=True
)

if result["successful"]:
    # Proceed with video transcription
    pass
```

## Troubleshooting

### Slow Downloads

If downloads are slow:
1. Install `hf_transfer`: `pip install hf_transfer`
2. Verify environment variable: `HF_HUB_ENABLE_HF_TRANSFER=1`
3. Check network connection

### Disk Space Issues

If running out of space:
1. Download models individually
2. Use smaller models (tiny, base)
3. Clean old model versions from cache

### Model Not Found

If model not recognized:
```python
from model_downloader import WHISPER_MODELS

# List all available models
print("Available models:", list(WHISPER_MODELS.keys()))
```

## License

Part of the Tubify project. See root LICENSE file.

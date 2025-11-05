# Transcriber Package

Factory-based transcriber supporting multiple backends (currently Whisper with faster-whisper + WhisperX).

## Features

- Transcribe audio/video files with word-level timestamps
- Automatic device detection (Metal/CPU/CUDA)
- Smart compute type selection for Apple Silicon (float16 with Metal)
- WhisperX alignment for precise word-level timing
- Multiple output formats: JSON, SRT, VTT, CSV
- Audio normalization (16kHz mono)
- Factory pattern for easy extensibility
- Automatic language detection

## Architecture

```
transcriber/
├── __init__.py                  # Package exports
├── whisper_transcriber.py       # WhisperTranscriber class (implementation)
├── transcriber_factory.py       # Factory pattern
└── main.py                      # CLI entry point
```

## Available Backends

| Backend | Description | Models |
|---------|-------------|--------|
| whisper | faster-whisper + WhisperX | tiny, base, small, medium, large-v3 |

## Usage

### 1. CLI Usage (via main.py)

```bash
# Basic transcription
./transcriber/main.py inputs/video.mp4

# Use specific model
./transcriber/main.py inputs/video.mp4 --model base

# Specify language
./transcriber/main.py inputs/video.mp4 --language en

# Custom output directory
./transcriber/main.py inputs/video.mp4 --output-dir results/

# Force CPU
./transcriber/main.py inputs/video.mp4 --device cpu

# Custom model directory
./transcriber/main.py inputs/video.mp4 --model-dir /path/to/models
```

### 2. Library Usage (Programmatic)

#### Using the Factory Pattern

```python
from transcriber import get_transcriber

# Get a Whisper transcriber
transcriber = get_transcriber(
    backend="whisper",
    model_name="large-v3",
    device="auto"
)

# Transcribe a file
result = transcriber.transcribe(
    input_file="inputs/video.mp4",
    output_dir="outputs"
)

# Result is a dictionary with paths to all generated files
print(f"Aligned JSON: {result['aligned_json']}")
print(f"SRT: {result['srt']}")
print(f"VTT: {result['vtt']}")
print(f"CSV: {result['clips_csv']}")
print(f"Normalized WAV: {result['wav']}")
```

#### Direct Class Instantiation

```python
from transcriber import WhisperTranscriber

# Create transcriber with custom settings
transcriber = WhisperTranscriber(
    model_name="large-v3",
    compute_type="float16",
    language="en",
    model_dir="models",
    device="auto"
)

# Transcribe
result = transcriber.transcribe(
    input_file="inputs/video.mp4",
    output_dir="outputs"
)
```

## API Reference

### Transcriber Protocol

```python
class Transcriber(Protocol):
    def transcribe(self, input_file: str, output_dir: str = "outputs") -> Dict[str, str]:
        """
        Transcribe audio/video file.

        Args:
            input_file: Path to audio/video file
            output_dir: Output directory for results

        Returns:
            Dictionary with paths to generated files
        """
```

### WhisperTranscriber Class

```python
class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "large-v3",
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        model_dir: str = "models",
        device: str = "auto"
    ):
        """
        Initialize Whisper transcriber.

        Args:
            model_name: Whisper model (tiny, base, small, medium, large-v3)
            compute_type: Compute type (float16, int8, etc.) or None for auto
            language: Force language code or None for auto-detect
            model_dir: Directory for model cache
            device: Device backend (metal, cpu, cuda, auto)
        """

    def transcribe(
        self,
        input_file: str,
        output_dir: str = "outputs"
    ) -> Dict[str, str]:
        """
        Transcribe audio/video file with word-level timestamps.

        Args:
            input_file: Path to audio/video file
            output_dir: Output directory for results

        Returns:
            Dictionary with keys:
              - aligned_json: Path to JSON with word-level timestamps
              - srt: Path to SRT subtitle file
              - vtt: Path to WebVTT subtitle file
              - clips_csv: Path to CSV with segments
              - wav: Path to normalized WAV file
        """
```

### Factory Function

```python
def get_transcriber(
    backend: BackendType = "whisper",
    model_name: str = "large-v3",
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
    model_dir: str = "models",
    device: str = "auto"
) -> Transcriber:
    """
    Factory function to get the appropriate transcriber.

    Args:
        backend: Transcription backend ("whisper" currently supported)
        model_name: Model name (for Whisper: "tiny", "base", "small", "medium", "large-v3")
        compute_type: Compute type ("float16", "int8", etc.)
        language: Force language code or None for auto-detect
        model_dir: Directory for model cache
        device: Device backend ("metal", "cpu", "cuda", "auto")

    Returns:
        A Transcriber instance for the specified backend

    Raises:
        ValueError: If the backend is not supported
    """
```

## Output Files

The `transcribe()` method generates multiple output files:

| File | Format | Description |
|------|--------|-------------|
| `*.aligned.json` | JSON | Word-level timestamps with full transcript |
| `*.srt` | SubRip | Standard subtitle format |
| `*.vtt` | WebVTT | Web Video Text Tracks format |
| `*.clips.csv` | CSV | Segment-based clips (start, end, text) |
| `*_16k.wav` | WAV | Normalized audio (16kHz mono) |

## Features in Detail

### Automatic Device Selection

The transcriber automatically selects the best device:

1. **Apple Silicon (macOS)**: Uses Metal acceleration via CTranslate2
   - Device: `auto` (enables Metal backend)
   - Compute type: `float16` (optimal for Metal)

2. **CPU**: Falls back to CPU if no GPU available
   - Device: `cpu`
   - Compute type: `int8` (best CPU performance)

3. **CUDA (Linux/Windows)**: Uses CUDA if available
   - Device: `cuda`
   - Compute type: `float16` or `int8_float16`

### Audio Normalization

Before transcription, audio is automatically normalized:
- Sample rate: 16kHz (Whisper requirement)
- Channels: Mono (single channel)
- Uses FFmpeg for conversion

### Two-Stage Transcription

1. **Stage 1: faster-whisper**
   - Fast initial transcription with segment-level timestamps
   - VAD (Voice Activity Detection) filtering
   - Beam search for accuracy

2. **Stage 2: WhisperX Alignment**
   - Word-level timestamp alignment
   - Uses Wav2Vec2 models
   - Forced CPU usage (avoids MPS limitations)

### Compute Type Fallback

If requested compute type is unsupported:
1. Tries requested type first
2. Falls back to `float16`
3. Falls back to `int8`
4. Warns user about the fallback

## Dependencies

- `faster-whisper >= 1.0` - Fast Whisper transcription
- `whisperx >= 3.4` - Word-level alignment
- `torch >= 2.0` - PyTorch backend
- `ffmpeg-python >= 0.2` - Audio processing
- `rich >= 13.7` - Beautiful terminal output
- `python >= 3.8` - Modern Python features

## Environment Variables

The package automatically sets local caches:

```python
os.environ["HF_HOME"] = "./.hf_home"
os.environ["HUGGINGFACE_HUB_CACHE"] = "./.hf_home/hub"
os.environ["CT2_HOME"] = "./.ct2_home"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
```

This keeps all caches self-contained within the project.

## Examples

### Transcribe with Auto Settings

```python
from transcriber import get_transcriber

# Auto-detect device and compute type
transcriber = get_transcriber(backend="whisper")
result = transcriber.transcribe("inputs/video.mp4")

print(f"Transcription complete!")
print(f"JSON: {result['aligned_json']}")
print(f"Subtitles: {result['srt']}")
```

### Force Specific Settings

```python
from transcriber import get_transcriber

# Force CPU with int8
transcriber = get_transcriber(
    backend="whisper",
    model_name="base",
    compute_type="int8",
    device="cpu",
    language="en"
)

result = transcriber.transcribe(
    input_file="inputs/audio.mp3",
    output_dir="results"
)
```

### Multiple File Processing

```python
from transcriber import get_transcriber
from pathlib import Path

transcriber = get_transcriber(backend="whisper", model_name="medium")

input_dir = Path("inputs")
for video_file in input_dir.glob("*.mp4"):
    print(f"Processing {video_file}...")
    result = transcriber.transcribe(
        input_file=str(video_file),
        output_dir="outputs"
    )
    print(f"  Generated: {result['srt']}")
```

### Error Handling

```python
from transcriber import get_transcriber

transcriber = get_transcriber(backend="whisper")

try:
    result = transcriber.transcribe("inputs/video.mp4")
    print("Success!")
except FileNotFoundError:
    print("Input file not found")
except Exception as e:
    print(f"Transcription failed: {e}")
```

## Extending to New Backends

To add a new backend (e.g., Google Cloud Speech):

1. Create `google_transcriber.py`:
```python
class GoogleTranscriber:
    def __init__(self, api_key: str, language: str = "en-US"):
        self.api_key = api_key
        self.language = language

    def transcribe(self, input_file: str, output_dir: str = "outputs") -> Dict[str, str]:
        # Implementation here
        pass
```

2. Update `transcriber_factory.py`:
```python
from .google_transcriber import GoogleTranscriber

BackendType = Literal["whisper", "google"]

def get_transcriber(
    backend: BackendType = "whisper",
    **kwargs
) -> Transcriber:
    if backend == "whisper":
        return WhisperTranscriber(**kwargs)
    elif backend == "google":
        return GoogleTranscriber(**kwargs)
```

3. Export in `__init__.py`:
```python
from .google_transcriber import GoogleTranscriber

__all__ = [..., "GoogleTranscriber"]
```

## Integration with Tubify

Used in the Tubify pipeline after video download:

```python
from transcriber import get_transcriber

# After downloading video and models
transcriber = get_transcriber(
    backend="whisper",
    model_name="large-v3",
    model_dir="models"
)

result = transcriber.transcribe(
    input_file="inputs/video.mp4",
    output_dir="outputs"
)

# Use result['aligned_json'] for next steps
# (e.g., LLM segmentation, clip generation)
```

## Troubleshooting

### Metal/MPS Issues on Apple Silicon

If you see Metal-related errors:
```bash
# Force CPU usage
./transcriber/main.py inputs/video.mp4 --device cpu
```

The alignment step automatically uses CPU due to MPS limitations with Wav2Vec2 models.

### Out of Memory Errors

If you run out of memory:
1. Use a smaller model: `--model base` or `--model tiny`
2. Use int8 compute: `--compute-type int8`
3. Force CPU: `--device cpu`

### Language Detection Issues

If language detection fails:
```bash
# Force specific language
./transcriber/main.py inputs/video.mp4 --language en
```

Available language codes: `en`, `es`, `fr`, `de`, `it`, `pt`, `ru`, `zh`, `ja`, `hi`, etc.

### Model Not Found

If model is not found:
```bash
# Download models first
./model_downloader/main.py --models large-v3

# Then transcribe
./transcriber/main.py inputs/video.mp4 --model large-v3 --model-dir models
```

## Performance Tips

### Apple Silicon (M1/M2/M3)

- Use `--device auto` (enables Metal acceleration)
- Use `--compute-type float16` (optimal for Metal)
- Model recommendation: `large-v3` for best quality

### Linux/Windows with CUDA

- Use `--device cuda`
- Use `--compute-type float16` or `--compute-type int8_float16`
- Model recommendation: `large-v3` or `medium`

### CPU Only

- Use smaller models: `base` or `small`
- Use `--compute-type int8` for faster inference
- Consider batching multiple files to amortize model load time

## License

Part of the Tubify project. See root LICENSE file.

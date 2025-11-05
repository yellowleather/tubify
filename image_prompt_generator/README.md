# Image Prompt Generator Package

Factory-based image prompt generator that analyzes video transcripts and generates detailed prompts for image generation models.

## Features

- Analyze video transcripts with word-level timestamps
- Break content into semantic sections (scenes/topics)
- Generate detailed image prompts for each section
- Support for multiple LLM backends (OpenAI, Anthropic, Ollama)
- Factory pattern for easy backend switching
- JSON output format with rich metadata
- Sentiment analysis and visual style suggestions

## Architecture

```
image_prompt_generator/
├── __init__.py                         # Package exports
├── image_prompt_generator_factory.py   # Factory + Protocol definition
├── openai_generator.py                 # OpenAI GPT implementation
├── anthropic_generator.py              # Anthropic Claude implementation
├── ollama_generator.py                 # Ollama local LLM implementation
├── main.py                             # CLI entry point
└── README.md                           # This file
```

## Available Backends

| Backend | Description | Models | API Key Required |
|---------|-------------|--------|------------------|
| openai | OpenAI GPT models | gpt-4o-mini, gpt-4o, gpt-4, gpt-3.5-turbo | Yes (OPENAI_API_KEY) |
| anthropic | Anthropic Claude models | claude-3-5-sonnet-20241022, claude-3-opus, etc. | Yes (ANTHROPIC_API_KEY) |
| ollama | Local LLMs via Ollama | llama3, mistral, mixtral, etc. | No (local) |

## Usage

### 1. CLI Usage (via main.py)

```bash
# Using local Ollama (default - no API key needed)
./image_prompt_generator/main.py outputs/video/tiny/video.aligned.json

# Using different Ollama model
./image_prompt_generator/main.py outputs/video/tiny/video.aligned.json \
  --model llama3

# Using OpenAI GPT-4o-mini
./image_prompt_generator/main.py outputs/video/tiny/video.aligned.json \
  --backend openai --model gpt-4o-mini

# Using Anthropic Claude
./image_prompt_generator/main.py outputs/video/tiny/video.aligned.json \
  --backend anthropic --model claude-3-5-sonnet-20241022

# Custom output directory
./image_prompt_generator/main.py outputs/video/tiny/video.aligned.json \
  --output-dir my_prompts/

# With API key as argument
./image_prompt_generator/main.py outputs/video/tiny/video.aligned.json \
  --backend openai --api-key sk-...
```

### 2. Library Usage (Programmatic)

#### Using the Factory Pattern

```python
from image_prompt_generator import get_generator

# OpenAI GPT-4
generator = get_generator(
    backend="openai",
    model="gpt-4o",
    temperature=0.7
)

result = generator.generate_prompts(
    transcript_file="outputs/video/tiny/video.aligned.json",
    output_dir="image_prompts"
)

print(f"Generated {len(result['sections'])} prompts")
print(f"Saved to: {result['prompts_file']}")

# Access individual sections
for section in result['sections']:
    print(f"Section: {section['scene_description']}")
    print(f"Prompt: {section['image_prompt']}")
    print(f"Time: {section['start_time']}s - {section['end_time']}s")
```

#### Anthropic Claude

```python
from image_prompt_generator import get_generator

generator = get_generator(
    backend="anthropic",
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-..."  # or use ANTHROPIC_API_KEY env var
)

result = generator.generate_prompts(
    transcript_file="transcript.aligned.json",
    output_dir="image_prompts"
)
```

#### Local Ollama

```python
from image_prompt_generator import get_generator

generator = get_generator(
    backend="ollama",
    model="llama3",
    base_url="http://localhost:11434"  # default
)

result = generator.generate_prompts(
    transcript_file="transcript.aligned.json",
    output_dir="image_prompts"
)
```

## API Reference

### ImagePromptGenerator Protocol

```python
class ImagePromptGenerator(Protocol):
    def generate_prompts(
        self,
        transcript_file: str,
        output_dir: str = "image_prompts"
    ) -> Dict[str, Any]:
        """
        Generate image prompts from transcript file.

        Args:
            transcript_file: Path to aligned JSON transcript
            output_dir: Directory to save prompts

        Returns:
            Dictionary with:
              - prompts_file: Path to generated prompts JSON
              - sections: List of section objects
              - video_name: Name of the video
        """
```

### Factory Function

```python
def get_generator(
    backend: LLMBackend = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> ImagePromptGenerator:
    """
    Factory function to get image prompt generator.

    Args:
        backend: LLM backend ("openai", "anthropic", "ollama")
        model: Model name (backend-specific)
        api_key: API key (optional, uses env var if not provided)
        **kwargs: Backend-specific parameters

    Returns:
        ImagePromptGenerator instance

    Raises:
        ValueError: If backend is not supported
    """
```

## Output Format

The generator produces a JSON file with the following structure:

```json
{
  "video_name": "video_title",
  "transcript_file": "path/to/transcript.aligned.json",
  "model": "gpt-4o-mini",
  "generated_at": "2024-11-04T19:00:00",
  "language": "en",
  "num_sections": 5,
  "sections": [
    {
      "start_time": 0.0,
      "end_time": 5.2,
      "transcript_text": "The full transcript text for this section...",
      "image_prompt": "A detailed, cinematic scene showing...",
      "sentiment": "positive",
      "scene_description": "Introduction to the topic",
      "visual_style": "cinematic"
    },
    {
      "start_time": 5.2,
      "end_time": 12.8,
      "transcript_text": "Next section transcript...",
      "image_prompt": "A vibrant illustration depicting...",
      "sentiment": "neutral",
      "scene_description": "Main content explanation",
      "visual_style": "artistic"
    }
  ]
}
```

### Section Object Fields

| Field | Type | Description |
|-------|------|-------------|
| start_time | float | Start time in seconds |
| end_time | float | End time in seconds |
| transcript_text | string | Transcript text for this section |
| image_prompt | string | Detailed prompt for image generation |
| sentiment | string | Sentiment analysis (positive/neutral/negative) |
| scene_description | string | Brief summary of the scene |
| visual_style | string | Suggested visual style (realistic/artistic/cinematic/etc) |

## Installation

### Core Dependencies

```bash
# For OpenAI backend
pip install openai

# For Anthropic backend
pip install anthropic

# For Ollama backend
pip install requests
```

### Environment Variables

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# For Ollama (if not using default localhost)
# No API key needed, just make sure Ollama is running
```

## Examples

### Complete Pipeline Integration

```python
from transcriber import get_transcriber
from image_prompt_generator import get_generator

# Step 1: Transcribe video
transcriber = get_transcriber(backend="whisper", model_name="tiny")
transcript_result = transcriber.transcribe(
    input_file="inputs/video.mp4",
    output_dir="outputs/video/tiny"
)

# Step 2: Generate image prompts from transcript
generator = get_generator(backend="openai", model="gpt-4o-mini")
prompt_result = generator.generate_prompts(
    transcript_file=transcript_result['aligned_json'],
    output_dir="image_prompts"
)

print(f"Generated {len(prompt_result['sections'])} image prompts")
```

### Multi-Backend Comparison

```python
from image_prompt_generator import get_generator

transcript_file = "outputs/video/tiny/video.aligned.json"

# Try different backends
backends = [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-sonnet-20241022"),
    ("ollama", "llama3")
]

for backend, model in backends:
    try:
        generator = get_generator(backend=backend, model=model)
        result = generator.generate_prompts(
            transcript_file=transcript_file,
            output_dir=f"image_prompts/{backend}"
        )
        print(f"{backend}/{model}: {len(result['sections'])} sections")
    except Exception as e:
        print(f"{backend}/{model}: Failed - {e}")
```

### Custom Temperature and Parameters

```python
from image_prompt_generator import get_generator

# More creative (higher temperature)
creative_generator = get_generator(
    backend="openai",
    model="gpt-4o",
    temperature=1.0
)

# More deterministic (lower temperature)
precise_generator = get_generator(
    backend="openai",
    model="gpt-4o",
    temperature=0.2
)
```

## Backend-Specific Notes

### OpenAI

**Default Model**: `gpt-4o-mini` (fast and cost-effective)

**Recommended Models**:
- `gpt-4o-mini` - Fast, cheap, good quality
- `gpt-4o` - Best quality, higher cost
- `gpt-3.5-turbo` - Cheaper alternative

**Setup**:
```bash
export OPENAI_API_KEY="sk-..."
pip install openai
```

### Anthropic Claude

**Default Model**: `claude-3-5-sonnet-20241022`

**Recommended Models**:
- `claude-3-5-sonnet-20241022` - Best balance
- `claude-3-opus-20240229` - Highest quality
- `claude-3-haiku-20240307` - Fastest, cheapest

**Setup**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
pip install anthropic
```

### Ollama (Local)

**Default Model**: `llama3`

**Recommended Models**:
- `llama3` - Good general performance
- `mistral` - Fast and capable
- `mixtral` - High quality, larger
- `phi3` - Small and fast

**Setup**:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3

# Start Ollama server (usually auto-starts)
ollama serve

# Use with Tubify
pip install requests
```

## Prompt Engineering

The image prompts are generated to be:

1. **Detailed and Descriptive**: Include visual elements, composition, lighting, mood
2. **Action-Oriented**: Describe what's happening, not just what exists
3. **Style-Aware**: Suggest appropriate visual styles (realistic, artistic, cinematic)
4. **Context-Rich**: Include relevant scene context from the transcript

Example generated prompt:
```
"A cinematic wide-angle shot of a modern classroom with natural lighting
streaming through large windows. A teacher stands at the front, gesturing
enthusiastically at a digital whiteboard displaying colorful diagrams.
Students are engaged, taking notes on tablets. The atmosphere is bright,
focused, and intellectually stimulating. Professional photography style,
high detail, warm color palette."
```

## Troubleshooting

### OpenAI Authentication Error

```
Error: Incorrect API key provided
```

**Solution**: Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Anthropic API Error

```
Error: anthropic package not installed
```

**Solution**: Install the Anthropic SDK:
```bash
pip install anthropic
```

### Ollama Connection Error

```
Warning: Ollama generation failed: Connection refused
```

**Solution**: Make sure Ollama is running:
```bash
ollama serve
```

### Model Not Found

```
Error: Model 'llama3' not found
```

**Solution**: Pull the model first:
```bash
ollama pull llama3
```

## Integration with Tubify Pipeline

This module is designed to be integrated into the main Tubify pipeline:

```
1. Download video (video_downloader)
   ↓
2. Download models (model_downloader)
   ↓
3. Transcribe video (transcriber)
   ↓
4. Generate image prompts (image_prompt_generator) ← YOU ARE HERE
   ↓
5. Generate images (coming soon)
   ↓
6. Animate images (coming soon)
```

The output prompts can be fed into image generation models like:
- DALL-E 3
- Stable Diffusion XL
- Midjourney
- Flux
- And others

## Performance

**Response Times** (approximate):

| Backend | Model | Speed | Quality |
|---------|-------|-------|---------|
| OpenAI | gpt-4o-mini | ~5-10s | Good |
| OpenAI | gpt-4o | ~10-20s | Excellent |
| Anthropic | claude-3-5-sonnet | ~8-15s | Excellent |
| Ollama | llama3 | ~15-30s | Good |
| Ollama | mixtral | ~30-60s | Very Good |

*Times for a typical 1-minute video transcript*

## License

Part of the Tubify project. See root LICENSE file.

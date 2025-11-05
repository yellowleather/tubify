# Image Generator

Generate images from text prompts using various AI models. Supports local Stable Diffusion and cloud-based APIs.

## Features

- **Multiple Backends**: Switch between local and cloud-based image generation
- **Factory Pattern**: Easy to extend with new backends
- **Batch Generation**: Process multiple prompts efficiently
- **Timestamp Sync**: Images saved with timestamps for video alignment
- **Progress Tracking**: Rich progress bars for generation status

## Supported Backends

| Backend | Description | Models | API Key Required |
|---------|-------------|--------|------------------|
| diffusers | Local Stable Diffusion | SDXL, SD 1.5, Turbo, etc. | No (local) |
| openai | OpenAI DALL-E | dall-e-2, dall-e-3 | Yes (OPENAI_API_KEY) |
| stability | Stability AI API | sd3-large, sd3-medium, etc. | Yes (STABILITY_API_KEY) |

## Installation

### For Local Generation (diffusers)

```bash
# Already included in main requirements.txt
pip install diffusers torch torchvision transformers accelerate Pillow
```

**Hardware Requirements**:
- **GPU recommended**: NVIDIA (CUDA) or Apple Silicon (MPS)
- **RAM**: 8GB+ system RAM
- **VRAM**: 6GB+ for SDXL, 4GB+ for SD 1.5
- **Storage**: ~7GB for SDXL model download

### For API-based Generation

```bash
# OpenAI (already in requirements.txt)
pip install openai

# Stability AI (uses standard requests library)
pip install requests
```

## Usage

### 1. CLI Usage (via main.py)

```bash
# Using local Stable Diffusion (default - free, requires GPU)
./image_generator/main.py image_prompts/video_prompts.json

# Use faster Turbo model (4 steps instead of 30)
./image_generator/main.py image_prompts/video_prompts.json \
  --model stabilityai/sdxl-turbo --steps 4

# Use OpenAI DALL-E 3
./image_generator/main.py image_prompts/video_prompts.json \
  --backend openai --model dall-e-3 --quality hd

# Use Stability AI
./image_generator/main.py image_prompts/video_prompts.json \
  --backend stability --model sd3-large

# Custom settings for local generation
./image_generator/main.py image_prompts/video_prompts.json \
  --steps 50 --guidance 8.0 --width 1024 --height 576 \
  --negative-prompt "blurry, distorted, low quality"

# Custom output directory
./image_generator/main.py image_prompts/video_prompts.json \
  --output-dir my_images/
```

### 2. Python API

```python
from image_generator import get_generator

# Local Stable Diffusion (default)
generator = get_generator(
    backend="diffusers",
    model="stabilityai/stable-diffusion-xl-base-1.0",
    device="auto",  # auto-detect: cuda > mps > cpu
    num_inference_steps=30,
    guidance_scale=7.5,
)

result = generator.generate_images(
    prompts_file="image_prompts/video_prompts.json",
    output_dir="generated_images",
)

print(f"Generated {result['num_images']} images")
print(f"Saved to: {result['output_dir']}")

# OpenAI DALL-E
generator = get_generator(
    backend="openai",
    model="dall-e-3",
    quality="standard",  # or "hd"
    size="1024x1024",
)

result = generator.generate_images(
    prompts_file="image_prompts/video_prompts.json",
    output_dir="generated_images",
)

# Stability AI
generator = get_generator(
    backend="stability",
    model="sd3-large",
    aspect_ratio="16:9",
)

result = generator.generate_images(
    prompts_file="image_prompts/video_prompts.json",
    output_dir="generated_images",
)
```

### 3. Integrated Pipeline

```bash
# Run complete pipeline with default settings
./main.py --video_url "https://www.youtube.com/watch?v=..."

# Use DALL-E for image generation
./main.py --video_url "https://www.youtube.com/watch?v=..." \
  --image-backend openai

# Use faster SDXL Turbo model
./main.py --video_url "https://www.youtube.com/watch?v=..." \
  --image-model stabilityai/sdxl-turbo --image-steps 4
```

## Backend Details

### Diffusers (Local Stable Diffusion)

**Pros**:
- Free, no API costs
- Privacy - runs entirely locally
- Full control over generation parameters
- No rate limits

**Cons**:
- Requires GPU for reasonable speed
- Large model downloads (~7GB)
- Slower than API-based options

**Recommended Models**:
- `stabilityai/stable-diffusion-xl-base-1.0` - Best quality (default)
- `stabilityai/sdxl-turbo` - Fastest (4 steps)
- `runwayml/stable-diffusion-v1-5` - Lighter weight

**Key Parameters**:
- `num_inference_steps`: 20-50 for quality, 4 for turbo
- `guidance_scale`: 7-8 for creative, 3-5 for realistic
- `negative_prompt`: What to avoid in images

### OpenAI DALL-E

**Pros**:
- High quality, consistent results
- Fast generation
- No local GPU needed

**Cons**:
- Costs ~$0.04 per image (1024x1024)
- Requires API key and credits
- Rate limits apply

**Models**:
- `dall-e-3` - Latest, best quality (default)
- `dall-e-2` - Cheaper, faster

**Key Parameters**:
- `quality`: "standard" or "hd" (DALL-E 3 only)
- `size`: "1024x1024", "1792x1024", "1024x1792"

### Stability AI

**Pros**:
- Competitive pricing (~$0.004-0.04 per image)
- Latest Stable Diffusion 3 models
- No local GPU needed

**Cons**:
- Requires API key and credits
- Rate limits apply

**Models**:
- `sd3-large` - Best quality (default)
- `sd3-medium` - Faster, cheaper

**Key Parameters**:
- `aspect_ratio`: "1:1", "16:9", "9:16", etc.
- `output_format`: "png" or "jpeg"

## Output Format

Generated images are saved with timestamps for video synchronization:

```
generated_images/
└── video_name/
    ├── frame_00000000.png  # Image at 0.0 seconds
    ├── frame_00003500.png  # Image at 3.5 seconds
    ├── frame_00007800.png  # Image at 7.8 seconds
    └── metadata.json       # Generation metadata
```

### Metadata File

```json
{
  "video_name": "example_video",
  "prompts_file": "image_prompts/example_video_prompts.json",
  "output_dir": "generated_images/example_video",
  "model": "stabilityai/stable-diffusion-xl-base-1.0",
  "backend": "diffusers",
  "num_images": 20,
  "generated_at": "2025-11-04T23:45:00",
  "settings": {
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "image_size": [1024, 1024],
    "negative_prompt": "blurry, bad quality..."
  },
  "images": [
    {
      "image_path": "generated_images/example_video/frame_00000000.png",
      "timestamp": 0.0,
      "prompt": "A dramatic illustration of...",
      "section_index": 0
    }
  ]
}
```

## Environment Variables

```bash
# For OpenAI DALL-E (in .env file)
OPENAI_API_KEY=sk-your-openai-api-key-here

# For Stability AI (in .env file)
STABILITY_API_KEY=sk-your-stability-api-key-here
```

## Performance Tips

### Local Generation (diffusers)

1. **Use GPU**: Ensure PyTorch detects your GPU
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"  # NVIDIA
   python -c "import torch; print(torch.backends.mps.is_available())"  # Apple
   ```

2. **Use Turbo models**: For faster generation
   ```bash
   --model stabilityai/sdxl-turbo --steps 4
   ```

3. **Reduce image size**: For faster generation
   ```bash
   --width 768 --height 768
   ```

4. **Enable optimizations**: Automatically enabled for MPS/CUDA

### API-based Generation

1. **Batch wisely**: APIs have rate limits
2. **Monitor costs**: Track API usage
3. **Cache results**: Reuse generated images when possible

## Troubleshooting

### Out of Memory (OOM)

```bash
# Use smaller model
--model runwayml/stable-diffusion-v1-5

# Reduce image size
--width 512 --height 512

# Use CPU (slower but works)
--device cpu
```

### Slow Generation

```bash
# Use Turbo model (4 steps)
--model stabilityai/sdxl-turbo --steps 4

# Or switch to API backend
--backend openai
```

### Import Errors

```bash
# Install missing packages
pip install diffusers torch torchvision transformers accelerate

# For CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### API Errors

- **401 Unauthorized**: Check API key in .env file
- **429 Rate Limit**: Wait and retry, or upgrade plan
- **402 Payment Required**: Add credits to account

## Cost Comparison

| Backend | Cost per Image | Speed | Quality | Local |
|---------|---------------|-------|---------|-------|
| diffusers (local) | Free | Slow-Medium | High | Yes |
| DALL-E 3 | ~$0.04 | Fast | Very High | No |
| Stability AI | ~$0.004-0.04 | Fast | High | No |

**Example costs for 20 images**:
- Local (diffusers): $0 (uses your GPU)
- DALL-E 3: ~$0.80
- Stability AI: ~$0.08-0.80

## Next Steps

After generating images, you can:

1. **Review images**: Check quality and consistency
2. **Regenerate specific frames**: Adjust prompts for poor results
3. **Animate images**: Create video from generated frames (coming soon)
4. **Add transitions**: Smooth transitions between frames (coming soon)

## API Reference

See [image_generator_factory.py](image_generator_factory.py) for complete API documentation.

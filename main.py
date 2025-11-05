#!/usr/bin/env -S uv run python
"""
main.py

Main entry point for Tubify - Complete video-to-image-prompt pipeline.
Downloads ML models, videos from YouTube, transcribes them, and generates image prompts.
"""

import argparse
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from video_downloader import get_downloader
from video_downloader.youtube_downloader import normalize_youtube_url
from model_downloader import get_downloader as get_model_downloader, WHISPER_MODELS
from transcriber import get_transcriber
from image_prompt_generator import get_generator as get_prompt_generator
from image_generator import get_generator as get_image_generator


def get_video_filename(video_url: str, output_dir: str) -> str:
    """
    Get the expected filename for a YouTube video.

    Args:
        video_url: YouTube video URL
        output_dir: Output directory

    Returns:
        Path to the downloaded video file
    """
    import yt_dlp

    # Normalize URLs in the same way as the downloader (handles Shorts, etc.)
    norm_url = normalize_youtube_url(video_url)

    # yt-dlp options matching those in youtube_downloader
    ydl_opts = {
        "outtmpl": str(Path(output_dir) / "%(title)s.%(ext)s"),
        "format": "best",
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(norm_url, download=False)
        return ydl.prepare_filename(info)


def main() -> int:
    """
    Main entry point for Tubify pipeline.
    1. Downloads ML models (if needed)
    2. Downloads video from YouTube (if needed)
    3. Transcribes video with word-level timestamps
    4. Generates image prompts from transcript using LLM
    5. Generates images from prompts using image generation model

    Returns:
        Exit code: 0 on success, non-zero on failure
    """
    parser = argparse.ArgumentParser(
        description="Tubify - Complete pipeline: download, transcribe, generate prompts, and create images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete pipeline with default settings (Ollama backend - no API key needed)
  python main.py --video_url "https://www.youtube.com/watch?v=..."

  # Use larger Whisper model for better transcription
  python main.py --video_url "https://www.youtube.com/shorts/QMJAUg2snas" --model large-v3

  # Use OpenAI GPT for prompt generation
  python main.py --video_url "https://www.youtube.com/watch?v=..." --prompt-backend openai

  # Use Anthropic Claude for prompt generation
  python main.py --video_url "https://www.youtube.com/watch?v=..." --prompt-backend anthropic

  # Use different Ollama model for prompt generation
  python main.py --video_url "https://www.youtube.com/watch?v=..." --prompt-model llama3

  # Use OpenAI DALL-E for image generation
  python main.py --video_url "https://www.youtube.com/watch?v=..." --image-backend openai

  # Use different Stable Diffusion model for image generation
  python main.py --video_url "https://www.youtube.com/watch?v=..." --image-model stabilityai/sdxl-turbo --image-steps 4

  # Custom model and directories
  python main.py --video_url "https://www.youtube.com/watch?v=..." --model tiny --prompt-backend anthropic --prompt-model claude-3-5-sonnet-20241022

  # Custom output directories
  python main.py --video_url "https://www.youtube.com/watch?v=..." --video-dir downloads/ --models-dir /path/to/models --prompts-dir my_prompts/ --images-dir my_images/

Available Whisper models: """ + ", ".join(WHISPER_MODELS.keys())
    )

    parser.add_argument(
        "--video_url",
        required=True,
        help="Video URL to download (YouTube, Shorts, etc.)"
    )
    parser.add_argument(
        "--video-dir",
        default="inputs/",
        help="Directory to save downloaded videos (default: inputs/)"
    )
    parser.add_argument(
        "--model",
        choices=list(WHISPER_MODELS.keys()),
        default="large-v3",
        help="Whisper model to use for transcription (default: large-v3)"
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory for ML models (default: models/)"
    )
    parser.add_argument(
        "--transcribe-dir",
        default="outputs",
        help="Base directory for transcription outputs (default: outputs/)"
    )
    parser.add_argument(
        "--prompt-backend",
        default="ollama",
        choices=["openai", "anthropic", "ollama"],
        help="LLM backend for prompt generation (default: ollama)"
    )
    parser.add_argument(
        "--prompt-model",
        default=None,
        help="LLM model for prompt generation (e.g., gpt-4o-mini, claude-3-5-sonnet-20241022, llama3)"
    )
    parser.add_argument(
        "--prompts-dir",
        default="image_prompts",
        help="Directory for generated image prompts (default: image_prompts/)"
    )
    parser.add_argument(
        "--image-backend",
        default="diffusers",
        choices=["diffusers", "openai", "stability"],
        help="Image generation backend (default: diffusers)"
    )
    parser.add_argument(
        "--image-model",
        default=None,
        help="Image model to use (e.g., stabilityai/stable-diffusion-xl-base-1.0, dall-e-3, sd3-large)"
    )
    parser.add_argument(
        "--images-dir",
        default="generated_images",
        help="Directory for generated images (default: generated_images/)"
    )
    parser.add_argument(
        "--image-steps",
        type=int,
        default=30,
        help="Number of inference steps for local generation (default: 30)"
    )

    args = parser.parse_args()

    # Validate paths
    video_dir = Path(args.video_dir)
    models_dir = Path(args.models_dir)
    transcribe_base_dir = Path(args.transcribe_dir)
    prompts_dir = Path(args.prompts_dir)
    images_dir = Path(args.images_dir)

    total_steps = 5

    print(f"[Tubify] ═══════════════════════════════════════════════")
    print(f"[Tubify] Tubify Pipeline Started (5 steps)")
    print(f"[Tubify] ═══════════════════════════════════════════════")
    print()

    # Step 1: Ensure ML model is available (download if needed)
    print(f"[Tubify] Step 1/{total_steps}: Checking ML model...")
    print(f"[Tubify] Model: {args.model}")
    print(f"[Tubify] Models directory: {models_dir.absolute()}")
    print()

    model_downloader = get_model_downloader("huggingface", models_dir=str(models_dir))
    model_result = model_downloader.download(
        models=[args.model],
        force=False,
        skip_space_check=True
    )

    if not model_result["successful"]:
        print()
        print(f"[Tubify] Model download failed!")
        return 1

    print()
    print(f"[Tubify] Model ready: {args.model}")
    print()

    # Step 2: Download video
    print(f"[Tubify] Step 2/{total_steps}: Downloading video...")
    print(f"[Tubify] URL: {args.video_url}")
    print(f"[Tubify] Video directory: {video_dir.absolute()}")
    print()

    # Get the appropriate downloader (currently only YouTube supported)
    video_downloader = get_downloader("youtube")

    # Download the video
    video_result = video_downloader.download(args.video_url, str(video_dir))

    if video_result != 0:
        print()
        print(f"[Tubify] Video download failed with exit code: {video_result}")
        return video_result

    # Get the video filename
    video_file = get_video_filename(args.video_url, str(video_dir))
    video_name = Path(video_file).stem

    print()
    print(f"[Tubify] Video ready: {video_file}")
    print()

    # Step 3: Transcribe video
    print(f"[Tubify] Step 3/{total_steps}: Transcribing video...")
    print(f"[Tubify] Model: {args.model}")
    print(f"[Tubify] Video: {video_file}")

    # Output structure: outputs/<video_name>/<model_size>/
    transcribe_output_dir = transcribe_base_dir / video_name / args.model
    print(f"[Tubify] Output directory: {transcribe_output_dir.absolute()}")
    print()

    try:
        # Get transcriber using factory pattern
        transcriber = get_transcriber(
            backend="whisper",
            model_name=args.model,
            model_dir=str(models_dir),
            device="auto"
        )

        # Perform transcription
        transcript_result = transcriber.transcribe(
            input_file=video_file,
            output_dir=str(transcribe_output_dir)
        )

        print()
        print(f"[Tubify] Transcription complete!")
        print()
        print("[Tubify] Generated files:")
        for file_path in transcript_result.values():
            print(f"  {file_path}")
        print()

    except Exception as e:
        print()
        print(f"[Tubify] Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Generate image prompts
    print(f"[Tubify] Step 4/{total_steps}: Generating image prompts...")
    print(f"[Tubify] Backend: {args.prompt_backend}")
    print(f"[Tubify] Model: {args.prompt_model or 'default'}")
    print(f"[Tubify] Output directory: {prompts_dir.absolute()}")
    print()

    try:
        # Get prompt generator using factory pattern
        prompt_generator = get_prompt_generator(
            backend=args.prompt_backend,
            model=args.prompt_model,
            temperature=0.7
        )

        # Generate prompts from transcript
        prompt_result = prompt_generator.generate_prompts(
            transcript_file=transcript_result['aligned_json'],
            output_dir=str(prompts_dir)
        )

        print()
        print(f"[Tubify] Image prompt generation complete!")
        print(f"[Tubify] Generated {len(prompt_result['sections'])} prompts")
        print(f"[Tubify] Saved to: {prompt_result['prompts_file']}")
        print()

    except ImportError as e:
        print()
        print(f"[Tubify] Error: Prompt generation failed - {e}")
        print(f"[Tubify] Install required package: pip install {args.prompt_backend}")
        print()
        return 1
    except Exception as e:
        print()
        print(f"[Tubify] Error: Prompt generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 5: Generate images from prompts
    print(f"[Tubify] Step 5/{total_steps}: Generating images...")
    print(f"[Tubify] Backend: {args.image_backend}")
    print(f"[Tubify] Model: {args.image_model or 'default'}")
    print(f"[Tubify] Output directory: {images_dir.absolute()}")
    print()

    try:
        # Get image generator using factory pattern
        image_generator = get_image_generator(
            backend=args.image_backend,
            model=args.image_model,
            device="auto",
            num_inference_steps=args.image_steps,
        )

        # Generate images from prompts
        image_result = image_generator.generate_images(
            prompts_file=prompt_result['prompts_file'],
            output_dir=str(images_dir)
        )

        print()
        print(f"[Tubify] Image generation complete!")
        print(f"[Tubify] Generated {image_result['num_images']} images")
        print(f"[Tubify] Saved to: {image_result['output_dir']}")
        print()

    except ImportError as e:
        print()
        print(f"[Tubify] Error: Image generation failed - {e}")
        if args.image_backend == "diffusers":
            print(f"[Tubify] Install required packages: pip install diffusers torch torchvision transformers accelerate")
        elif args.image_backend == "openai":
            print(f"[Tubify] Install required package: pip install openai")
        print()
        return 1
    except Exception as e:
        print()
        print(f"[Tubify] Error: Image generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Pipeline complete
    print(f"[Tubify] ═══════════════════════════════════════════════")
    print(f"[Tubify] Pipeline Complete!")
    print(f"[Tubify] ═══════════════════════════════════════════════")
    print()
    print("[Tubify] Generated outputs:")
    print(f"  Transcript: {transcript_result.get('aligned_json', 'N/A')}")
    print(f"  Prompts:    {prompt_result.get('prompts_file', 'N/A')}")
    print(f"  Images:     {image_result.get('output_dir', 'N/A')}")
    print(f"  Sections:   {len(prompt_result.get('sections', []))}")
    print(f"  Generated:  {image_result.get('num_images', 0)} images")
    print()
    print("[Tubify] Next steps:")
    print(f"  1. Review generated images: {image_result.get('output_dir', 'N/A')}")
    print(f"  2. Animate images to create video (coming soon)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

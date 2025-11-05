#!/usr/bin/env -S uv run python
"""
main.py

Main entry point for Tubify - Video downloader and transcription pipeline.
Downloads ML models and videos from YouTube for transcription.
"""

import argparse
import sys
from pathlib import Path

from video_downloader import get_downloader
from model_downloader import get_downloader as get_model_downloader, WHISPER_MODELS


def main() -> int:
    """
    Main entry point for Tubify pipeline.
    1. Downloads ML models (if needed)
    2. Downloads video from YouTube

    Returns:
        Exit code: 0 on success, non-zero on failure
    """
    parser = argparse.ArgumentParser(
        description="Tubify - Download ML models and YouTube videos for transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a YouTube video (downloads models if needed)
  python main.py --video_url "https://www.youtube.com/watch?v=..."

  # Download specific models and video
  python main.py --video_url "https://www.youtube.com/shorts/QMJAUg2snas" --models large-v3

  # Skip model download (if already downloaded)
  python main.py --video_url "https://www.youtube.com/watch?v=..." --skip-model-download

  # Custom output directory
  python main.py --video_url "https://www.youtube.com/watch?v=..." --output_dir downloads/

Available models: """ + ", ".join(WHISPER_MODELS.keys())
    )

    parser.add_argument(
        "--video_url",
        required=True,
        help="Video URL to download (YouTube, Shorts, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        default="inputs/",
        help="Directory to save the downloaded file (default: inputs/)"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(WHISPER_MODELS.keys()),
        default=["large-v3"],
        help="ML models to download (default: large-v3)"
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Skip ML model download step"
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory for ML models (default: models/)"
    )

    args = parser.parse_args()

    # Validate output directory path
    output_path = Path(args.output_dir)

    print(f"[Tubify] ═══════════════════════════════════════════════")
    print(f"[Tubify] Tubify Pipeline Started")
    print(f"[Tubify] ═══════════════════════════════════════════════")
    print()

    # Step 1: Download ML models
    if not args.skip_model_download:
        print(f"[Tubify] Step 1/2: Downloading ML models...")
        print(f"[Tubify] Models: {', '.join(args.models)}")
        print(f"[Tubify] Models directory: {Path(args.models_dir).absolute()}")
        print()

        model_downloader = get_model_downloader("huggingface", models_dir=args.models_dir)
        model_result = model_downloader.download(
            models=args.models,
            force=False,
            skip_space_check=True
        )

        if not model_result["successful"]:
            print()
            print(f"[Tubify] Model download failed!")
            return 1

        print()
        print(f"[Tubify] Models ready: {', '.join(model_result['successful'])}")
        print()
    else:
        print(f"[Tubify] Skipping model download (--skip-model-download)")
        print()

    # Step 2: Download video
    print(f"[Tubify] Step 2/2: Downloading video...")
    print(f"[Tubify] URL: {args.video_url}")
    print(f"[Tubify] Output directory: {output_path.absolute()}")
    print()

    # Get the appropriate downloader (currently only YouTube supported)
    video_downloader = get_downloader("youtube")

    # Download the video
    video_result = video_downloader.download(args.video_url, args.output_dir)

    if video_result == 0:
        print()
        print(f"[Tubify] Video download completed successfully!")
        print(f"[Tubify] Video saved to: {output_path.absolute()}")
        print()
        print(f"[Tubify] ═══════════════════════════════════════════════")
        print(f"[Tubify] Pipeline Complete!")
        print(f"[Tubify] ═══════════════════════════════════════════════")
        print()
        print("[Tubify] Next steps:")
        print(f"  1. Transcribe: python transcribe.py <video_file>")
        print(f"  2. Build clips: python cutter/build_clips.py <aligned_json>")
    else:
        print()
        print(f"[Tubify] Video download failed with exit code: {video_result}")
        return video_result

    return 0


if __name__ == "__main__":
    sys.exit(main())

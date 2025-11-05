#!/usr/bin/env -S uv run python
"""
main.py

Main entry point for Tubify - Complete video-to-transcript pipeline.
Downloads ML models, videos from YouTube, and transcribes them.
"""

import argparse
import sys
from pathlib import Path

from video_downloader import get_downloader
from model_downloader import get_downloader as get_model_downloader, WHISPER_MODELS
from transcriber import get_transcriber


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

    # Normalize Shorts URLs to /watch URLs
    norm_url = video_url.replace("/shorts/", "/watch?v=")

    # yt-dlp options matching those in youtube_downloader
    ydl_opts = {
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "format": "best",
        "quiet": True,
        "no_warnings": True,
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

    Returns:
        Exit code: 0 on success, non-zero on failure
    """
    parser = argparse.ArgumentParser(
        description="Tubify - Complete pipeline: download, transcribe, and process videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete pipeline with default settings (tiny model)
  python main.py --video_url "https://www.youtube.com/watch?v=..."

  # Use larger model for better accuracy
  python main.py --video_url "https://www.youtube.com/shorts/QMJAUg2snas" --model large-v3

  # Skip model download (if already downloaded)
  python main.py --video_url "https://www.youtube.com/watch?v=..." --skip-model-download

  # Custom directories
  python main.py --video_url "https://www.youtube.com/watch?v=..." --video-dir downloads/ --models-dir /path/to/models

Available models: """ + ", ".join(WHISPER_MODELS.keys())
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
        default="tiny",
        help="Whisper model to use for transcription (default: tiny)"
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
    parser.add_argument(
        "--transcribe-dir",
        default="outputs",
        help="Base directory for transcription outputs (default: outputs/)"
    )

    args = parser.parse_args()

    # Validate paths
    video_dir = Path(args.video_dir)
    models_dir = Path(args.models_dir)
    transcribe_base_dir = Path(args.transcribe_dir)

    print(f"[Tubify] ═══════════════════════════════════════════════")
    print(f"[Tubify] Tubify Pipeline Started")
    print(f"[Tubify] ═══════════════════════════════════════════════")
    print()

    # Step 1: Download ML models
    if not args.skip_model_download:
        print(f"[Tubify] Step 1/3: Downloading ML models...")
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
    else:
        print(f"[Tubify] Step 1/3: Skipping model download (--skip-model-download)")
        print()

    # Step 2: Download video
    print(f"[Tubify] Step 2/3: Downloading video...")
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
    print(f"[Tubify] Step 3/3: Transcribing video...")
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
        result = transcriber.transcribe(
            input_file=video_file,
            output_dir=str(transcribe_output_dir)
        )

        print()
        print(f"[Tubify] Transcription complete!")
        print()
        print("[Tubify] Generated files:")
        for file_path in result.values():
            print(f"  {file_path}")
        print()

    except Exception as e:
        print()
        print(f"[Tubify] Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Pipeline complete
    print(f"[Tubify] ═══════════════════════════════════════════════")
    print(f"[Tubify] Pipeline Complete!")
    print(f"[Tubify] ═══════════════════════════════════════════════")
    print()
    print("[Tubify] Next steps:")
    print(f"  1. Review transcript: {result.get('aligned_json', 'N/A')}")
    print(f"  2. Build clips: python cutter/build_clips.py {result.get('aligned_json', '<aligned_json>')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

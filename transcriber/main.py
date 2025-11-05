#!/usr/bin/env -S uv run python
"""
transcriber/main.py

CLI entry point for the transcriber package.
Supports transcription with faster-whisper + WhisperX alignment.
"""

import argparse
import os
import sys

from rich import print
from rich.console import Console

from transcriber import get_transcriber


def main() -> int:
    """
    Main entry point for transcription CLI.

    Returns:
        Exit code: 0 on success, non-zero on failure
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video with word-level timestamps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe with default model (large-v3)
  python transcriber/main.py input.mp4

  # Use specific model
  python transcriber/main.py input.mp4 --model base

  # Specify language
  python transcriber/main.py input.mp4 --language en

  # Custom output directory
  python transcriber/main.py input.mp4 --output-dir results/

  # Force CPU
  python transcriber/main.py input.mp4 --device cpu
"""
    )

    parser.add_argument(
        "input_file",
        help="Path to audio/video file to transcribe"
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model: large-v3 | medium | small | base | tiny (default: large-v3)"
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Compute type: float16 | int8 | int8_float16 (auto-detected if not specified)"
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g., 'en', 'hi') or None for auto-detect"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for output files (default: outputs/)"
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory for model cache (default: models/)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device backend: metal | cpu | cuda | auto (default: auto)"
    )
    parser.add_argument(
        "--backend",
        default="whisper",
        choices=["whisper"],
        help="Transcription backend (default: whisper)"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input_file):
        print(f"[red]Error:[/red] Input file not found: {args.input_file}")
        print()
        print("Hint: Make sure the file path is correct, e.g.:")
        print("  python transcriber/main.py inputs/video.mp4")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    console = Console()
    console.rule("[bold cyan]Transcription Pipeline")
    print(f"[bold]Input:[/bold] {args.input_file}")
    print(f"[bold]Model:[/bold] {args.model}")
    print(f"[bold]Backend:[/bold] {args.backend}")
    print(f"[bold]Device:[/bold] {args.device}")
    print(f"[bold]Language:[/bold] {args.language or 'auto-detect'}")
    print(f"[bold]Output:[/bold] {args.output_dir}")
    print()

    try:
        # Get transcriber using factory pattern
        transcriber = get_transcriber(
            backend=args.backend,
            model_name=args.model,
            compute_type=args.compute_type,
            language=args.language,
            model_dir=args.model_dir,
            device=args.device
        )

        # Perform transcription
        result = transcriber.transcribe(
            input_file=args.input_file,
            output_dir=args.output_dir
        )

        # Display results
        print()
        print("[bold green]Transcription complete![/bold green]")
        print()
        print("Generated files:")
        for file_type, file_path in result.items():
            print(f"  {file_type:12} → {file_path}")
        print()

        return 0

    except Exception as e:
        print()
        print(f"[red]Transcription failed:[/red] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

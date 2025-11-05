#!/usr/bin/env -S uv run python
"""
main.py

CLI entry point for model downloader.
Downloads Whisper models from HuggingFace Hub via command line.
"""

import argparse
import sys
import os

# Add parent directory to path for imports when run as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model_downloader import get_downloader, WHISPER_MODELS
except ImportError:
    from model_downloader_factory import get_downloader
    from huggingface_downloader import WHISPER_MODELS


def main() -> int:
    """
    Main CLI entry point for model downloader.

    Returns:
        Exit code: 0 on success, non-zero on failure
    """
    parser = argparse.ArgumentParser(
        description="Download Whisper models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models
  python main.py

  # Download specific models
  python main.py --models tiny base

  # Force redownload
  python main.py --models large-v3 --force

  # Custom models directory
  python main.py --models-dir /path/to/models

Available models:
  """ + "\n  ".join([f"- {name}: {info['description']}" for name, info in WHISPER_MODELS.items()])
    )

    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(WHISPER_MODELS.keys()),
        default=None,
        help="Specific models to download (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force fresh downloads even if files exist"
    )
    parser.add_argument(
        "--skip-space-check",
        action="store_true",
        help="Skip disk space check"
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store downloaded models (default: models/)"
    )

    args = parser.parse_args()

    # Get the downloader
    downloader = get_downloader("huggingface", models_dir=args.models_dir)

    # Download models
    result = downloader.download(
        models=args.models,
        force=args.force,
        skip_space_check=args.skip_space_check
    )

    # Return success if at least one model succeeded
    if result["successful"]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

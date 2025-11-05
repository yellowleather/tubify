#!/usr/bin/env -S uv run python
"""
image_prompt_generator/main.py

CLI entry point for the image prompt generator package.
Generates image prompts from video transcripts using various LLM backends.
"""

import argparse
import os
import sys

from rich import print
from rich.console import Console

from image_prompt_generator import get_generator


def main() -> int:
    """
    Main entry point for image prompt generation CLI.

    Returns:
        Exit code: 0 on success, non-zero on failure
    """
    parser = argparse.ArgumentParser(
        description="Generate image prompts from video transcripts using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate prompts using local Ollama (default - no API key needed)
  python image_prompt_generator/main.py transcript.aligned.json

  # Use different Ollama model
  python image_prompt_generator/main.py transcript.aligned.json --model llama3

  # Use OpenAI GPT-4
  python image_prompt_generator/main.py transcript.aligned.json --backend openai --model gpt-4

  # Use Anthropic Claude
  python image_prompt_generator/main.py transcript.aligned.json --backend anthropic --model claude-3-5-sonnet-20241022

  # Custom output directory
  python image_prompt_generator/main.py transcript.aligned.json --output-dir my_prompts/

Environment Variables:
  OPENAI_API_KEY     - API key for OpenAI (if using --backend openai)
  ANTHROPIC_API_KEY  - API key for Anthropic (if using --backend anthropic)
"""
    )

    parser.add_argument(
        "transcript_file",
        help="Path to aligned JSON transcript file from transcriber"
    )
    parser.add_argument(
        "--backend",
        default="ollama",
        choices=["openai", "anthropic", "ollama"],
        help="LLM backend to use (default: ollama)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (e.g., gpt-4, claude-3-5-sonnet-20241022, llama3)"
    )
    parser.add_argument(
        "--output-dir",
        default="image_prompts",
        help="Directory for output prompts (default: image_prompts/)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the backend (optional, uses env var if not provided)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0-1.0, default: 0.7)"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.transcript_file):
        print(f"[red]Error:[/red] Transcript file not found: {args.transcript_file}")
        print()
        print("Hint: Make sure you have run the transcriber first:")
        print("  python transcribe.py inputs/video.mp4")
        return 1

    console = Console()
    console.rule("[bold cyan]Image Prompt Generation")
    print(f"[bold]Transcript:[/bold] {args.transcript_file}")
    print(f"[bold]Backend:[/bold] {args.backend}")
    print(f"[bold]Model:[/bold] {args.model or 'default'}")
    print(f"[bold]Output:[/bold] {args.output_dir}")
    print()

    try:
        # Get generator using factory pattern
        generator = get_generator(
            backend=args.backend,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature
        )

        # Generate prompts
        result = generator.generate_prompts(
            transcript_file=args.transcript_file,
            output_dir=args.output_dir
        )

        # Display results
        print()
        print("[bold green]Prompt generation complete![/bold green]")
        print()
        print(f"Video: {result['video_name']}")
        print(f"Sections: {len(result['sections'])}")
        print(f"Output: {result['prompts_file']}")
        print()

        # Show first few prompts as preview
        if result['sections']:
            print("[bold]Preview of generated prompts:[/bold]")
            for i, section in enumerate(result['sections'][:3], 1):
                print(f"\n[cyan]Section {i}[/cyan] ({section.get('start_time', 0):.1f}s - {section.get('end_time', 0):.1f}s)")
                print(f"  Scene: {section.get('scene_description', 'N/A')}")
                print(f"  Prompt: {section.get('image_prompt', 'N/A')[:100]}...")
                print(f"  Style: {section.get('visual_style', 'N/A')}")

            if len(result['sections']) > 3:
                print(f"\n... and {len(result['sections']) - 3} more sections")

        print()
        return 0

    except ImportError as e:
        print()
        print(f"[red]Import error:[/red] {e}")
        print()
        print("Install required dependencies:")
        if "openai" in str(e):
            print("  pip install openai")
        elif "anthropic" in str(e):
            print("  pip install anthropic")
        elif "requests" in str(e):
            print("  pip install requests")
        return 1

    except Exception as e:
        print()
        print(f"[red]Prompt generation failed:[/red] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

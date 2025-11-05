#!/usr/bin/env python3
"""
image_generator/main.py

CLI entry point for generating images from prompts.
"""

import argparse
import sys
from pathlib import Path

from image_generator_factory import get_generator


def main() -> int:
    """
    Generate images from prompts file using various backends.

    Returns:
        Exit code: 0 on success, non-zero on failure
    """
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using various backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate using local Stable Diffusion (default - free, requires GPU)
  python image_generator/main.py prompts.json

  # Use different Stable Diffusion model
  python image_generator/main.py prompts.json --model stabilityai/sdxl-turbo

  # Use OpenAI DALL-E 3
  python image_generator/main.py prompts.json --backend openai --model dall-e-3

  # Use Stability AI API
  python image_generator/main.py prompts.json --backend stability --model sd3-large

  # Custom output directory and settings
  python image_generator/main.py prompts.json --output-dir my_images/ --steps 50 --guidance 8.0

Environment Variables:
  OPENAI_API_KEY     - API key for OpenAI (if using --backend openai)
  STABILITY_API_KEY  - API key for Stability AI (if using --backend stability)
"""
    )

    parser.add_argument(
        "prompts_file",
        help="Path to JSON file with image prompts (from image_prompt_generator)"
    )
    parser.add_argument(
        "--backend",
        default="diffusers",
        choices=["diffusers", "openai", "stability"],
        help="Image generation backend (default: diffusers)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (e.g., stabilityai/stable-diffusion-xl-base-1.0, dall-e-3, sd3-large)"
    )
    parser.add_argument(
        "--output-dir",
        default="generated_images",
        help="Directory for generated images (default: generated_images/)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for local generation (default: auto)"
    )

    # Diffusers-specific arguments
    diffusers_group = parser.add_argument_group("Diffusers (local) options")
    diffusers_group.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (default: 30, higher = better quality but slower)"
    )
    diffusers_group.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5, how closely to follow prompt)"
    )
    diffusers_group.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)"
    )
    diffusers_group.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)"
    )
    diffusers_group.add_argument(
        "--negative-prompt",
        default=None,
        help="What to avoid in images (default: blurry, bad quality, etc.)"
    )

    # OpenAI-specific arguments
    openai_group = parser.add_argument_group("OpenAI DALL-E options")
    openai_group.add_argument(
        "--quality",
        default="standard",
        choices=["standard", "hd"],
        help="Image quality for DALL-E 3 (default: standard)"
    )
    openai_group.add_argument(
        "--size",
        default="1024x1024",
        help="Image size for DALL-E (default: 1024x1024)"
    )

    # Stability AI-specific arguments
    stability_group = parser.add_argument_group("Stability AI options")
    stability_group.add_argument(
        "--aspect-ratio",
        default="1:1",
        choices=["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
        help="Image aspect ratio for Stability AI (default: 1:1)"
    )
    stability_group.add_argument(
        "--format",
        default="png",
        choices=["png", "jpeg"],
        help="Output format for Stability AI (default: png)"
    )

    args = parser.parse_args()

    # Validate prompts file
    if not Path(args.prompts_file).exists():
        print(f"Error: Prompts file not found: {args.prompts_file}")
        return 1

    print(f"Image Generator")
    print(f"Backend: {args.backend}")
    print(f"Prompts: {args.prompts_file}")
    print()

    try:
        # Create generator based on backend
        if args.backend == "diffusers":
            generator = get_generator(
                backend="diffusers",
                model=args.model,
                device=args.device,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                image_size=(args.width, args.height),
            )

            # Generate images
            result = generator.generate_images(
                prompts_file=args.prompts_file,
                output_dir=args.output_dir,
                negative_prompt=args.negative_prompt,
            )

        elif args.backend == "openai":
            generator = get_generator(
                backend="openai",
                model=args.model,
                quality=args.quality,
                size=args.size,
            )

            # Generate images
            result = generator.generate_images(
                prompts_file=args.prompts_file,
                output_dir=args.output_dir,
            )

        elif args.backend == "stability":
            generator = get_generator(
                backend="stability",
                model=args.model,
                aspect_ratio=args.aspect_ratio,
                output_format=args.format,
            )

            # Generate images
            result = generator.generate_images(
                prompts_file=args.prompts_file,
                output_dir=args.output_dir,
                negative_prompt=args.negative_prompt,
            )

        else:
            print(f"Error: Unknown backend: {args.backend}")
            return 1

        print()
        print("Image generation complete!")
        print(f"Generated {result['num_images']} images")
        print(f"Saved to: {result['output_dir']}")

        return 0

    except ImportError as e:
        print()
        print(f"Error: Missing required package - {e}")
        print()
        if args.backend == "diffusers":
            print("Install with: pip install diffusers torch torchvision transformers accelerate")
        elif args.backend == "openai":
            print("Install with: pip install openai")
        elif args.backend == "stability":
            print("Stability AI uses standard packages (requests)")
        print()
        return 1

    except Exception as e:
        print()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

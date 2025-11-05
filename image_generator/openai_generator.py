"""
openai_generator.py

Image generation using OpenAI DALL-E API.
"""

import json
import os
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


class OpenAIImageGenerator:
    """
    Image generation using OpenAI DALL-E API.

    Supports DALL-E 2 and DALL-E 3 models.
    Requires OPENAI_API_KEY environment variable or api_key parameter.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        quality: str = "standard",
        size: str = "1024x1024",
        **kwargs
    ):
        """
        Initialize OpenAI image generator.

        Args:
            model: Model to use (dall-e-2, dall-e-3). Default: dall-e-3
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            quality: Image quality - "standard" or "hd" (DALL-E 3 only)
            size: Image size - "1024x1024", "1792x1024", "1024x1792" (DALL-E 3)
                  or "256x256", "512x512", "1024x1024" (DALL-E 2)
            **kwargs: Additional API parameters
        """
        self.model = model or "dall-e-3"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.quality = quality
        self.size = size
        self.kwargs = kwargs

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def generate_images(
        self,
        prompts_file: str,
        output_dir: str = "generated_images",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images from prompts file using DALL-E.

        Args:
            prompts_file: Path to JSON file with image prompts
            output_dir: Directory to save generated images
            **kwargs: Override initialization parameters

        Returns:
            Dictionary with generation results
        """
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        # Load prompts
        with open(prompts_file, 'r') as f:
            prompts_data = json.load(f)

        video_name = prompts_data.get("video_name", "video")
        sections = prompts_data.get("sections", [])

        # Create output directory
        output_path = Path(output_dir) / video_name
        output_path.mkdir(parents=True, exist_ok=True)

        rprint(f"[cyan]Generating {len(sections)} images with {self.model}...[/cyan]")
        rprint(f"[cyan]Output directory: {output_path.absolute()}[/cyan]")
        rprint(f"[yellow]Note: DALL-E API costs apply (~$0.04 per image for 1024x1024)[/yellow]")
        rprint()

        # Override parameters if provided
        model = kwargs.get("model", self.model)
        quality = kwargs.get("quality", self.quality)
        size = kwargs.get("size", self.size)

        # Generate images
        generated_images = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Generating images...", total=len(sections))

            for i, section in enumerate(sections):
                prompt = section.get("image_prompt", "")
                start_time = section.get("start_time", 0.0)
                visual_style = section.get("visual_style", "")

                # Enhance prompt with visual style
                full_prompt = f"{prompt}, {visual_style} style" if visual_style else prompt

                # DALL-E 3 has a 4000 character limit
                if len(full_prompt) > 4000:
                    full_prompt = full_prompt[:3997] + "..."

                try:
                    # Generate image
                    response = client.images.generate(
                        model=model,
                        prompt=full_prompt,
                        size=size,
                        quality=quality,
                        n=1,
                    )

                    # Download image
                    image_url = response.data[0].url
                    image_data = requests.get(image_url).content

                    # Save with timestamp for video sync
                    filename = f"frame_{int(start_time * 1000):08d}.png"
                    image_path = output_path / filename

                    with open(image_path, 'wb') as f:
                        f.write(image_data)

                    generated_images.append({
                        "image_path": str(image_path),
                        "timestamp": start_time,
                        "prompt": full_prompt,
                        "section_index": i,
                        "revised_prompt": response.data[0].revised_prompt if hasattr(response.data[0], 'revised_prompt') else None,
                    })

                except Exception as e:
                    rprint(f"[yellow]Warning: Failed to generate image {i}: {e}[/yellow]")

                progress.update(task, advance=1)

        # Save metadata
        metadata = {
            "video_name": video_name,
            "prompts_file": prompts_file,
            "output_dir": str(output_path),
            "model": model,
            "backend": "openai",
            "num_images": len(generated_images),
            "generated_at": datetime.now().isoformat(),
            "settings": {
                "quality": quality,
                "size": size,
            },
            "images": generated_images,
        }

        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        rprint()
        rprint(f"[green]Generated {len(generated_images)} images![/green]")
        rprint(f"[cyan]Saved to: {output_path}[/cyan]")
        rprint(f"[cyan]Metadata: {metadata_file}[/cyan]")

        return metadata

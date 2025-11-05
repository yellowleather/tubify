"""
stability_generator.py

Image generation using Stability AI API.
"""

import json
import os
import requests
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


class StabilityGenerator:
    """
    Image generation using Stability AI API.

    Supports Stable Diffusion 3 and other Stability AI models.
    Requires STABILITY_API_KEY environment variable or api_key parameter.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        aspect_ratio: str = "1:1",
        output_format: str = "png",
        **kwargs
    ):
        """
        Initialize Stability AI image generator.

        Args:
            model: Model to use (sd3-large, sd3-medium, etc.). Default: sd3-large
            api_key: Stability AI API key (or set STABILITY_API_KEY env var)
            aspect_ratio: Image aspect ratio - "1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"
            output_format: Output format - "png" or "jpeg"
            **kwargs: Additional API parameters
        """
        self.model = model or "sd3-large"
        self.api_key = api_key or os.environ.get("STABILITY_API_KEY")
        self.aspect_ratio = aspect_ratio
        self.output_format = output_format
        self.kwargs = kwargs
        self.api_host = "https://api.stability.ai"

        if not self.api_key:
            raise ValueError(
                "Stability AI API key not found. Set STABILITY_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def generate_images(
        self,
        prompts_file: str,
        output_dir: str = "generated_images",
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images from prompts file using Stability AI.

        Args:
            prompts_file: Path to JSON file with image prompts
            output_dir: Directory to save generated images
            negative_prompt: What to avoid in images (optional)
            **kwargs: Override initialization parameters

        Returns:
            Dictionary with generation results
        """
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
        rprint(f"[yellow]Note: Stability AI API costs apply (~$0.004-0.04 per image)[/yellow]")
        rprint()

        # Override parameters if provided
        model = kwargs.get("model", self.model)
        aspect_ratio = kwargs.get("aspect_ratio", self.aspect_ratio)
        output_format = kwargs.get("output_format", self.output_format)

        # Default negative prompt
        if negative_prompt is None:
            negative_prompt = "blurry, bad quality, distorted, ugly, low resolution"

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

                try:
                    # API endpoint
                    url = f"{self.api_host}/v2beta/stable-image/generate/sd3"

                    # Request payload
                    payload = {
                        "prompt": full_prompt,
                        "negative_prompt": negative_prompt,
                        "aspect_ratio": aspect_ratio,
                        "output_format": output_format,
                        "model": model,
                        "mode": "text-to-image",
                    }

                    # Make API request
                    response = requests.post(
                        url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Accept": "application/json",
                        },
                        files={"none": ''},
                        data=payload,
                    )

                    if response.status_code == 200:
                        # Parse response
                        result = response.json()

                        # Decode base64 image
                        image_data = base64.b64decode(result['image'])

                        # Save with timestamp for video sync
                        filename = f"frame_{int(start_time * 1000):08d}.{output_format}"
                        image_path = output_path / filename

                        with open(image_path, 'wb') as f:
                            f.write(image_data)

                        generated_images.append({
                            "image_path": str(image_path),
                            "timestamp": start_time,
                            "prompt": full_prompt,
                            "section_index": i,
                            "seed": result.get("seed"),
                        })
                    else:
                        rprint(f"[yellow]Warning: Failed to generate image {i}: {response.text}[/yellow]")

                except Exception as e:
                    rprint(f"[yellow]Warning: Failed to generate image {i}: {e}[/yellow]")

                progress.update(task, advance=1)

        # Save metadata
        metadata = {
            "video_name": video_name,
            "prompts_file": prompts_file,
            "output_dir": str(output_path),
            "model": model,
            "backend": "stability",
            "num_images": len(generated_images),
            "generated_at": datetime.now().isoformat(),
            "settings": {
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "negative_prompt": negative_prompt,
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

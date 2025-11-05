"""
diffusers_generator.py

Local Stable Diffusion image generation using Hugging Face diffusers library.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


class DiffusersGenerator:
    """
    Local image generation using Stable Diffusion via diffusers library.

    Supports various Stable Diffusion models from Hugging Face.
    Runs entirely locally - no API keys needed.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        image_size: tuple = (1024, 1024),
        **kwargs
    ):
        """
        Initialize diffusers generator.

        Args:
            model: Model name/path (default: stabilityai/stable-diffusion-xl-base-1.0)
            device: Device to use (auto, cuda, mps, cpu)
            num_inference_steps: Number of denoising steps (higher = better quality, slower)
            guidance_scale: How closely to follow the prompt (7-15 typical)
            image_size: Output image size (width, height)
            **kwargs: Additional pipeline parameters
        """
        self.model = model or "stabilityai/stable-diffusion-xl-base-1.0"
        self.device = self._detect_device(device)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.kwargs = kwargs
        self.pipeline = None

    def _detect_device(self, device: Optional[str]) -> str:
        """Detect the best available device."""
        if device and device != "auto":
            return device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def _load_pipeline(self):
        """Lazy load the diffusion pipeline."""
        if self.pipeline is not None:
            return

        try:
            from diffusers import DiffusionPipeline
            import torch

            rprint(f"[cyan]Loading Stable Diffusion model: {self.model}[/cyan]")
            rprint(f"[cyan]Device: {self.device}[/cyan]")
            rprint(f"[dim]This may take a few minutes on first run (downloading ~7GB)[/dim]")

            # Load pipeline
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
                use_safetensors=True,
            )

            # Move to device
            self.pipeline = self.pipeline.to(self.device)

            # Enable optimizations
            if self.device == "mps":
                # MPS-specific optimizations
                self.pipeline.enable_attention_slicing()
            elif self.device == "cuda":
                # CUDA-specific optimizations
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except Exception:
                    # xformers not available, use attention slicing
                    self.pipeline.enable_attention_slicing()

            rprint("[green]Model loaded successfully![/green]")

        except ImportError as e:
            raise ImportError(
                f"Failed to import required packages: {e}\n"
                "Install with: pip install diffusers torch torchvision transformers accelerate"
            )

    def generate_images(
        self,
        prompts_file: str,
        output_dir: str = "generated_images",
        negative_prompt: Optional[str] = None,
        batch_size: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images from prompts file.

        Args:
            prompts_file: Path to JSON file with image prompts
            output_dir: Directory to save generated images
            negative_prompt: What to avoid in images (optional)
            batch_size: Number of images to generate at once
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

        rprint(f"[cyan]Generating {len(sections)} images...[/cyan]")
        rprint(f"[cyan]Output directory: {output_path.absolute()}[/cyan]")
        rprint()

        # Load pipeline
        self._load_pipeline()

        # Default negative prompt
        if negative_prompt is None:
            negative_prompt = "blurry, bad quality, distorted, ugly, low resolution, artifacts"

        # Override parameters if provided
        num_inference_steps = kwargs.get("num_inference_steps", self.num_inference_steps)
        guidance_scale = kwargs.get("guidance_scale", self.guidance_scale)
        image_size = kwargs.get("image_size", self.image_size)

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

                # Generate image
                try:
                    image = self.pipeline(
                        prompt=full_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=image_size[0],
                        height=image_size[1],
                    ).images[0]

                    # Save with timestamp for video sync
                    filename = f"frame_{int(start_time * 1000):08d}.png"
                    image_path = output_path / filename
                    image.save(image_path)

                    generated_images.append({
                        "image_path": str(image_path),
                        "timestamp": start_time,
                        "prompt": full_prompt,
                        "section_index": i,
                    })

                except Exception as e:
                    rprint(f"[yellow]Warning: Failed to generate image {i}: {e}[/yellow]")

                progress.update(task, advance=1)

        # Save metadata
        metadata = {
            "video_name": video_name,
            "prompts_file": prompts_file,
            "output_dir": str(output_path),
            "model": self.model,
            "device": self.device,
            "num_images": len(generated_images),
            "generated_at": datetime.now().isoformat(),
            "settings": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "image_size": image_size,
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

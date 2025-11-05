"""
huggingface_downloader.py

HuggingFace model downloader implementation.

Downloads Whisper models from HuggingFace Hub with local caching.
- Skips download if model already exists
- Uses HF cache format that faster-whisper expects
- Supports fast transfers via hf_transfer

Requirements:
  - huggingface_hub >= 0.23
  - hf_transfer >= 0.1.4 (optional, for faster downloads)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from rich import print as rprint
from rich.console import Console
from rich.table import Table

# Keep caches inside the project (self-contained)
os.environ.setdefault("HF_HOME", "./.hf_home")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "./.hf_home/hub")

# Enable fast downloads if hf_transfer is available
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


# Whisper model definitions
WHISPER_MODELS = {
    "tiny": {
        "repo": "Systran/faster-whisper-tiny",
        "size_gb": 0.075,  # ~75MB
        "description": "Fastest, least accurate"
    },
    "base": {
        "repo": "Systran/faster-whisper-base",
        "size_gb": 0.145,  # ~145MB
        "description": "Good speed/accuracy balance"
    },
    "small": {
        "repo": "Systran/faster-whisper-small",
        "size_gb": 0.483,  # ~483MB
        "description": "Better accuracy"
    },
    "medium": {
        "repo": "Systran/faster-whisper-medium",
        "size_gb": 1.53,   # ~1.53GB
        "description": "High accuracy"
    },
    "large-v3": {
        "repo": "Systran/faster-whisper-large-v3",
        "size_gb": 3.09,   # ~3.09GB
        "description": "Best accuracy (recommended)"
    }
}


def check_disk_space(required_gb: float) -> bool:
    """
    Check if we have enough disk space.

    Args:
        required_gb: Required space in GB

    Returns:
        True if sufficient space, False otherwise
    """
    try:
        import shutil
        free_gb = shutil.disk_usage(".").free / (1024**3)
        if free_gb < required_gb + 1:  # +1GB buffer
            rprint(f"[yellow]Warning:[/yellow] Need ~{required_gb:.1f}GB but only {free_gb:.1f}GB free")
            return False
        return True
    except Exception:
        return True  # Can't check, assume OK


class HuggingFaceDownloader:
    """
    HuggingFace model downloader for Whisper models.

    Implements the ModelDownloader protocol with a download() method.
    Downloads models from HuggingFace Hub with local caching.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the HuggingFace downloader.

        Args:
            models_dir: Directory to store downloaded models (default: "models")
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()

    def get_available_models(self) -> Dict[str, Dict]:
        """Get dictionary of available Whisper models."""
        return WHISPER_MODELS.copy()

    def download_single_model(self, model_name: str, force: bool = False) -> Optional[str]:
        """
        Download a single model.

        Args:
            model_name: Name of the model (e.g., "tiny", "base", "large-v3")
            force: Force fresh download even if file exists

        Returns:
            Path to downloaded model on success, None on failure
        """
        if model_name not in WHISPER_MODELS:
            rprint(f"[red]Unknown model: {model_name}[/red]")
            rprint(f"[yellow]Available models: {', '.join(WHISPER_MODELS.keys())}[/yellow]")
            return None

        model_info = WHISPER_MODELS[model_name]
        repo_id = model_info["repo"]

        # Use HuggingFace cache format that faster-whisper expects
        cache_dir = self.models_dir / f"models--{repo_id.replace('/', '--')}"

        rprint(f"[cyan]Downloading {model_name}[/cyan] ({model_info['size_gb']:.2f}GB) → {cache_dir}")

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check if already exists (unless force)
        if not force and (cache_dir / "model.bin").exists():
            rprint(f"[green]{model_name} already exists (use force=True to redownload)[/green]")
            return str(cache_dir)

        try:
            from huggingface_hub import snapshot_download
            result_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(cache_dir),
                force_download=force,
            )
            rprint(f"[green]{model_name} downloaded successfully[/green]")
            return result_path
        except Exception as e:
            rprint(f"[red]Failed to download {model_name}: {e}[/red]")
            return None

    def download(
        self,
        models: Optional[List[str]] = None,
        force: bool = False,
        skip_space_check: bool = False
    ) -> Dict[str, List[str]]:
        """
        Download one or more models.

        Args:
            models: List of model names to download (None = all models)
            force: Force fresh downloads even if files exist
            skip_space_check: Skip disk space validation

        Returns:
            Dictionary with 'successful' and 'failed' lists of model names
        """
        # Default to all models if none specified
        if models is None:
            models = list(WHISPER_MODELS.keys())

        # Validate model names
        invalid = [m for m in models if m not in WHISPER_MODELS]
        if invalid:
            rprint(f"[red]Invalid models: {', '.join(invalid)}[/red]")
            rprint(f"[yellow]Available: {', '.join(WHISPER_MODELS.keys())}[/yellow]")
            return {"successful": [], "failed": models}

        # Show model info table
        table = Table(title="Whisper Models to Download", show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Size", justify="right")
        table.add_column("Description")
        table.add_column("Repository", style="dim")

        total_size = 0
        for model in models:
            info = WHISPER_MODELS[model]
            table.add_row(
                model,
                f"{info['size_gb']:.2f}GB",
                info['description'],
                info['repo']
            )
            total_size += info['size_gb']

        self.console.print(table)
        rprint(f"\n[bold]Total download size: ~{total_size:.1f}GB[/bold]")

        # Check disk space
        if not skip_space_check:
            if not check_disk_space(total_size):
                response = self.console.input("\nContinue anyway? [y/N]: ")
                if not response.lower().startswith('y'):
                    return {"successful": [], "failed": models}

        rprint(f"\n[green]Starting downloads...[/green]")

        # Download each model
        successful = []
        failed = []

        for model_name in models:
            result = self.download_single_model(model_name, force=force)
            if result:
                successful.append(model_name)
            else:
                failed.append(model_name)

        # Summary
        rprint(f"\n[bold green]Download Summary:[/bold green]")
        rprint(f"Successful: {len(successful)}/{len(models)}")
        if successful:
            rprint(f"   {', '.join(successful)}")

        if failed:
            rprint(f"Failed: {len(failed)}")
            rprint(f"   {', '.join(failed)}")

        rprint(f"\n[bold]Models available in:[/bold] {self.models_dir.absolute()}/")
        rprint(f"[dim]Use with: ./transcribe.py input.mp4 --model <model_name>[/dim]")

        return {"successful": successful, "failed": failed}

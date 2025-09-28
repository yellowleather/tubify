#!/usr/bin/env -S uv run python
"""
Download all faster-whisper models (tiny, base, small, medium, large-v3) with fast transfers and local caching.

Usage:
  ./download_models.py                    # Download all models
  ./download_models.py --models tiny base # Download specific models only
  ./download_models.py --force            # Force fresh downloads
"""

import os
import argparse
from pathlib import Path
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Keep caches inside the project (self-contained)
os.environ.setdefault("HF_HOME", "./.hf_home")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "./.hf_home/hub")

# Enable fast downloads if hf_transfer is available
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# All available Whisper models with their approximate sizes
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

def check_disk_space(required_gb):
    """Check if we have enough disk space."""
    try:
        import shutil
        free_gb = shutil.disk_usage(".").free / (1024**3)
        if free_gb < required_gb + 1:  # +1GB buffer
            print(f"⚠️  [yellow]Warning:[/yellow] Need ~{required_gb:.1f}GB but only {free_gb:.1f}GB free")
            return False
        return True
    except Exception:
        return True  # Can't check, assume OK

def download_model(model_name, force=False):
    """Download a single model."""
    model_info = WHISPER_MODELS[model_name]
    repo_id = model_info["repo"]
    
    # Use HuggingFace cache format that faster-whisper expects
    cache_dir = f"models/models--{repo_id.replace('/', '--')}"
    out_dir = Path(cache_dir)
    
    print(f"[cyan]Downloading {model_name}[/cyan] ({model_info['size_gb']:.2f}GB) → {cache_dir}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists (unless force)
    if not force and (out_dir / "model.bin").exists():
        print(f"[green]✓[/green] {model_name} already exists (use --force to redownload)")
        return str(out_dir)
    
    try:
        from huggingface_hub import snapshot_download
        result_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(out_dir),
            force_download=force,
        )
        print(f"[green]✅ {model_name} downloaded successfully[/green]")
        return result_path
    except Exception as e:
        print(f"[red]❌ Failed to download {model_name}: {e}[/red]")
        return None

def main():
    ap = argparse.ArgumentParser(description="Download all faster-whisper models")
    ap.add_argument("--models", nargs="*", choices=list(WHISPER_MODELS.keys()),
                    default=list(WHISPER_MODELS.keys()),
                    help="Specific models to download (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="Force fresh downloads even if files exist")
    ap.add_argument("--skip-space-check", action="store_true",
                    help="Skip disk space check")
    args = ap.parse_args()

    console = Console()
    
    # Show model info table
    table = Table(title="Whisper Models to Download", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right") 
    table.add_column("Description")
    table.add_column("Repository", style="dim")
    
    total_size = 0
    for model in args.models:
        info = WHISPER_MODELS[model]
        table.add_row(
            model,
            f"{info['size_gb']:.2f}GB",
            info['description'], 
            info['repo']
        )
        total_size += info['size_gb']
    
    console.print(table)
    print(f"\n[bold]Total download size: ~{total_size:.1f}GB[/bold]")
    
    # Check disk space
    if not args.skip_space_check:
        if not check_disk_space(total_size):
            if not console.input("\nContinue anyway? [y/N]: ").lower().startswith('y'):
                return
    
    print(f"\n[green]Starting downloads...[/green]")
    
    # Download each model
    successful = []
    failed = []
    
    for model_name in args.models:
        result = download_model(model_name, force=args.force)
        if result:
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    # Summary
    print(f"\n[bold green]Download Summary:[/bold green]")
    print(f"✅ Successful: {len(successful)}/{len(args.models)}")
    if successful:
        print(f"   {', '.join(successful)}")
    
    if failed:
        print(f"❌ Failed: {len(failed)}")
        print(f"   {', '.join(failed)}")
    
    print(f"\n[bold]Models available in:[/bold] ./models/")
    print(f"[dim]Use with: ./run.py input.mp4 --model <model_name>[/dim]")

if __name__ == "__main__":
    main()
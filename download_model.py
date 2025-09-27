#!/usr/bin/env -S uv run python
"""
One-time model prefetch for faster-whisper (large-v3), with fast transfers and local caching.

Usage:
  ./download_model.py
  ./download_model.py --repo Systran/faster-whisper-large-v3 --out models/faster-whisper-large-v3
"""

import os
import argparse
from pathlib import Path

# Keep caches inside the project (self-contained)
os.environ.setdefault("HF_HOME", "./.hf_home")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "./.hf_home/hub")

# Enable fast downloads if hf_transfer is available
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Systran/faster-whisper-large-v3",
                    help="Hugging Face repo_id to download")
    ap.add_argument("--out", default="models/faster-whisper-large-v3",
                    help="Destination folder for the model")
    ap.add_argument("--force", action="store_true",
                    help="Force a fresh download even if files exist")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: quick disk-space sanity (~3.1 GB for large-v3)
    try:
        import shutil
        free = shutil.disk_usage(".").free
        if free < 6 * 1024**3:
            print("⚠️  Warning: Less than ~6 GB free disk space may cause issues.")
    except Exception:
        pass

    from huggingface_hub import snapshot_download
    p = snapshot_download(
        repo_id=args.repo,
        local_dir=str(out_dir),
        # resume is now default behavior; no need to set anything
        force_download=bool(args.force),  # only if user wants a fresh pull
    )
    print(f"✅ Model available at: {p}")

if __name__ == "__main__":
    main()

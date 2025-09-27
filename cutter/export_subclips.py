#!/usr/bin/env -S uv run python
import argparse, csv, os, subprocess, pathlib

def cut_clip(src, start, end, outpath):
    duration = max(0.01, end - start)
    cmd = [
        "ffmpeg","-y",
        "-ss", f"{start:.3f}",
        "-i", src,
        "-t", f"{duration:.3f}",
        "-c:v","libx264","-preset","fast","-crf","18",
        "-c:a","aac","-b:a","128k",
        outpath
    ]
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="Export subclips with ffmpeg from clips.csv")
    ap.add_argument("input", help="source media")
    ap.add_argument("csv", help="clips.csv (start,end,text)")
    ap.add_argument("--outdir", default="outputs/clips")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base = pathlib.Path(args.input).stem

    with open(args.csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows, start=1):
        start = float(row["start"]); end = float(row["end"])
        out = os.path.join(args.outdir, f"{base}_clip_{i:03}.mp4")
        cut_clip(args.input, start, end, out)
        print(f"✅ Wrote {out}")

if __name__ == "__main__":
    main()

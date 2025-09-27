#!/usr/bin/env -S uv run python
import os, sys, json, subprocess, argparse, pathlib, platform
from rich import print, box
from rich.console import Console
from rich.table import Table

# ---------- Local caches & accelerated HF downloads ----------
os.environ.setdefault("HF_HOME", "./.hf_home")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "./.hf_home/hub")
os.environ.setdefault("CT2_HOME", "./.ct2_home")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
try:
    import hf_transfer  # noqa: F401
except Exception:
    pass
# ------------------------------------------------------------

def ts_hhmmssms(t: float) -> str:
    h = int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int((t-int(t))*1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def write_srt(segments, out_path):
    with open(out_path,"w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n{ts_hhmmssms(seg['start'])} --> {ts_hhmmssms(seg['end'])}\n{seg['text'].strip()}\n\n")

def write_vtt(segments, out_path):
    with open(out_path,"w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = ts_hhmmssms(seg["start"]).replace(",", ".")
            end = ts_hhmmssms(seg["end"]).replace(",", ".")
            f.write(f"{start} --> {end}\n{seg['text'].strip()}\n\n")

def ffmpeg_norm(inp, out_wav):
    subprocess.run(["ffmpeg","-y","-i", inp, "-ac","1","-ar","16000", out_wav], check=True)

def transcribe_faster_whisper(wav, model_name, compute_type, language, download_root, device):
    from faster_whisper import WhisperModel

    # Candidate compute types: try requested first, then sensible fallbacks
    candidates = [compute_type, "float16", "int8"]
    last_err = None
    for ct in candidates:
        try:
            model = WhisperModel(
                model_name,
                device=device,
                compute_type=ct,
                download_root=download_root,
            )
            if ct != compute_type:
                print(f"[yellow]Note:[/yellow] compute_type '{compute_type}' unsupported on this backend. Using '{ct}'.")
            break
        except ValueError as e:
            last_err = e
    else:
        # If *everything* failed, raise the last error
        raise last_err

    kwargs = dict(vad_filter=True, beam_size=5, temperature=0.0)
    if language:
        kwargs["language"] = language

    segments, info = model.transcribe(wav, **kwargs)
    segs = [{"start":s.start,"end":s.end,"text":s.text} for s in segments]
    return segs, (language or info.language or "en")


def align_whisperx(segments, wav, lang):
    import torch, whisperx
    
    # Force CPU for alignment due to MPS limitations with large models
    # WhisperX Wav2Vec2 models have >65536 channels which exceed MPS limits
    device = "cpu"
    
    print(f"[yellow]Note:[/yellow] Using CPU for alignment (MPS has channel limits for Wav2Vec2)")
    
    # Fix for WhisperX 3.4.2 API
    try:
        # Try the correct parameter name for WhisperX 3.4.2
        align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
    except TypeError:
        try:
            # Try older API (positional language argument)
            align_model, metadata = whisperx.load_align_model(lang, device)
        except TypeError:
            # Fallback: try without language specification
            print(f"[yellow]Warning:[/yellow] WhisperX alignment model for '{lang}' not found. Using default.")
            align_model, metadata = whisperx.load_align_model(device=device)
    
    return whisperx.align(segments, align_model, metadata, wav, device)

def default_clips_from_segments(segments):
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in segments]


def default_device_and_compute(user_device, user_compute):
    """Auto-pick fast defaults for Apple Silicon; respect user overrides."""
    if user_device and user_compute:
        return user_device, user_compute

    is_macos = (platform.system() == "Darwin")
    # Use "auto" for faster-whisper (lets CTranslate2 choose best backend including Metal)
    # On Apple Silicon, this will automatically use Metal acceleration via CTranslate2
    device = user_device or ("auto" if is_macos else "auto")
    
    # Prefer float16 on Apple Silicon for better performance with Metal acceleration
    if user_compute:
        compute = user_compute
    else:
        compute = "float16" if is_macos else "int8"
    return device, compute

def main():
    ap = argparse.ArgumentParser(description="ASR+Alignment (local, Apple Silicon friendly)")
    ap.add_argument("input", help="video/audio file")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--model", default="large-v3", help="Whisper model: large-v3 | medium | small | base | tiny")
    ap.add_argument("--compute-type", default=None, help="Preferred compute type: float16 | int8 | int8_float16 (CUDA only)")
    ap.add_argument("--language", default=None, help="Force language code like 'en' or 'hi' (optional)")
    ap.add_argument("--model-dir", default="models", help="Local directory for model cache/downloads")
    ap.add_argument("--device", default="auto", help="Device backend: metal | cpu | cuda | auto")
    args = ap.parse_args()

    # Input sanity
    if not os.path.isfile(args.input):
        print(f"[red]Input file not found:[/red] {args.input}")
        print("Hint: put your media under ./inputs and pass the real name, e.g.:")
        print("  ./run.py 'inputs/why_hinglish_hated.mp4' --outdir outputs --model large-v3")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Smart defaults for Apple Silicon (Metal + float16)
    device, compute_type = default_device_and_compute(args.device, args.compute_type)

    base = os.path.join(args.outdir, pathlib.Path(args.input).stem)
    wav = f"{base}_16k.wav"

    console = Console()
    console.rule("[bold cyan]ASR Local Agent")
    print(f"[bold]Input:[/bold] {args.input}")
    print(f"[bold]Model:[/bold] {args.model}   [bold]Compute:[/bold] {compute_type}   [bold]Lang:[/bold] {args.language or 'auto'}")
    print(f"[bold]Device:[/bold] {device}      [bold]Model cache:[/bold] {os.path.abspath(args.model_dir)}")

    # 1) Normalize audio
    print("[green]Step 1/3:[/green] Normalize audio → 16k mono WAV")
    ffmpeg_norm(args.input, wav)

    # 2) Transcribe
    print("[green]Step 2/3:[/green] Transcribe with faster-whisper")
    segments, lang = transcribe_faster_whisper(wav, args.model, compute_type, args.language, args.model_dir, device)

    # 3) Align (word-level)
    print("[green]Step 3/3:[/green] Align with WhisperX (word-level)")
    aligned = align_whisperx(segments, wav, lang)
    aligned_segments = aligned["segments"]

    # Save outputs
    data = {"language": lang, "segments": aligned_segments}
    jpath = f"{base}.aligned.json"
    with open(jpath,"w",encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

    write_srt(aligned_segments, f"{base}.srt")
    write_vtt(aligned_segments, f"{base}.vtt")

    # Simple default clips = segments
    clips = default_clips_from_segments(aligned_segments)
    import csv
    with open(f"{base}.clips.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["start","end","text"])
        for c in clips: w.writerow([c["start"], c["end"], c["text"]])

    # Summary table
    table = Table(title="Outputs", box=box.SIMPLE)
    table.add_column("File")
    for p in [jpath, f"{base}.srt", f"{base}.vtt", f"{base}.clips.csv"]:
        table.add_row(p)
    console.print(table)
    print("[bold green]Done.[/bold green]")

if __name__ == "__main__":
    main()

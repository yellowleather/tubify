"""
whisper_transcriber.py

Whisper-based transcriber implementation using faster-whisper + WhisperX.

Handles transcription with word-level alignment for Apple Silicon and other platforms.
"""

import os
import sys
import json
import csv
import subprocess
import pathlib
import platform
from typing import Dict, List, Optional, Tuple

from rich import print as rprint

# Local caches & accelerated HF downloads
os.environ.setdefault("HF_HOME", "./.hf_home")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "./.hf_home/hub")
os.environ.setdefault("CT2_HOME", "./.ct2_home")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
try:
    import hf_transfer  # noqa: F401
except Exception:
    pass


def ts_hhmmssms(t: float) -> str:
    """Convert timestamp to HH:MM:SS,MS format."""
    h = int(t//3600)
    m = int((t%3600)//60)
    s = int(t%60)
    ms = int((t-int(t))*1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segments: List[Dict], out_path: str) -> None:
    """Write segments to SRT subtitle format."""
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n{ts_hhmmssms(seg['start'])} --> {ts_hhmmssms(seg['end'])}\n{seg['text'].strip()}\n\n")


def write_vtt(segments: List[Dict], out_path: str) -> None:
    """Write segments to VTT subtitle format."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = ts_hhmmssms(seg["start"]).replace(",", ".")
            end = ts_hhmmssms(seg["end"]).replace(",", ".")
            f.write(f"{start} --> {end}\n{seg['text'].strip()}\n\n")


def ffmpeg_norm(inp: str, out_wav: str) -> None:
    """Normalize audio to 16kHz mono WAV using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", inp, "-ac", "1", "-ar", "16000", out_wav],
        check=True
    )


class WhisperTranscriber:
    """
    Whisper-based transcriber using faster-whisper + WhisperX alignment.

    Implements the Transcriber protocol with a transcribe() method.
    Optimized for Apple Silicon (Metal acceleration).
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        model_dir: str = "models",
        device: str = "auto"
    ):
        """
        Initialize the Whisper transcriber.

        Args:
            model_name: Whisper model ("tiny", "base", "small", "medium", "large-v3")
            compute_type: Compute type ("float16", "int8", "int8_float16")
            language: Force language code ("en", "hi", etc.) or None for auto-detect
            model_dir: Directory for model cache/downloads
            device: Device backend ("metal", "cpu", "cuda", "auto")
        """
        self.model_name = model_name
        self.language = language
        self.model_dir = model_dir

        # Smart defaults for Apple Silicon
        self.device, self.compute_type = self._default_device_and_compute(device, compute_type)

        os.makedirs(self.model_dir, exist_ok=True)

    def _default_device_and_compute(
        self,
        user_device: str,
        user_compute: Optional[str]
    ) -> Tuple[str, str]:
        """Auto-pick fast defaults for Apple Silicon; respect user overrides."""
        if user_device and user_compute:
            return user_device, user_compute

        is_macos = (platform.system() == "Darwin")
        # Use "auto" for faster-whisper (lets CTranslate2 choose best backend including Metal)
        device = user_device or ("auto" if is_macos else "auto")

        # Prefer float16 on Apple Silicon for better performance with Metal acceleration
        if user_compute:
            compute = user_compute
        else:
            compute = "float16" if is_macos else "int8"

        return device, compute

    def _transcribe_faster_whisper(
        self,
        wav: str
    ) -> Tuple[List[Dict], str]:
        """
        Transcribe audio using faster-whisper.

        Returns:
            Tuple of (segments, detected_language)
        """
        from faster_whisper import WhisperModel

        # Candidate compute types: try requested first, then sensible fallbacks
        candidates = [self.compute_type, "float16", "int8"]
        last_err = None
        for ct in candidates:
            try:
                model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=ct,
                    download_root=self.model_dir,
                )
                if ct != self.compute_type:
                    rprint(f"[yellow]Note:[/yellow] compute_type '{self.compute_type}' unsupported. Using '{ct}'.")
                break
            except ValueError as e:
                last_err = e
        else:
            # If everything failed, raise the last error
            raise last_err

        kwargs = dict(vad_filter=True, beam_size=5, temperature=0.0)
        if self.language:
            kwargs["language"] = self.language

        segments, info = model.transcribe(wav, **kwargs)
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        return segs, (self.language or info.language or "en")

    def _align_whisperx(
        self,
        segments: List[Dict],
        wav: str,
        lang: str
    ) -> Dict:
        """
        Align segments with WhisperX for word-level timestamps.

        Returns:
            Aligned results with word-level timestamps
        """
        import torch
        import whisperx

        # Force CPU for alignment due to MPS limitations with large models
        device = "cpu"

        rprint(f"[yellow]Note:[/yellow] Using CPU for alignment (MPS has channel limits for Wav2Vec2)")

        # Fix for WhisperX 3.4.2 API
        try:
            align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
        except TypeError:
            try:
                align_model, metadata = whisperx.load_align_model(lang, device)
            except TypeError:
                rprint(f"[yellow]Warning:[/yellow] WhisperX alignment model for '{lang}' not found. Using default.")
                align_model, metadata = whisperx.load_align_model(device=device)

        return whisperx.align(segments, align_model, metadata, wav, device)

    def _default_clips_from_segments(self, segments: List[Dict]) -> List[Dict]:
        """Create simple clips from segments."""
        return [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in segments
        ]

    def transcribe(
        self,
        input_file: str,
        output_dir: str = "outputs"
    ) -> Dict[str, str]:
        """
        Transcribe audio/video file with word-level alignment.

        Args:
            input_file: Path to audio/video file
            output_dir: Output directory for results

        Returns:
            Dictionary with paths to generated files:
            {
                "aligned_json": "path/to/file.aligned.json",
                "srt": "path/to/file.srt",
                "vtt": "path/to/file.vtt",
                "clips_csv": "path/to/file.clips.csv",
                "wav": "path/to/file_16k.wav"
            }
        """
        # Validate input
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        os.makedirs(output_dir, exist_ok=True)

        base = os.path.join(output_dir, pathlib.Path(input_file).stem)
        wav = f"{base}_16k.wav"

        rprint(f"[bold]Input:[/bold] {input_file}")
        rprint(f"[bold]Model:[/bold] {self.model_name}   [bold]Compute:[/bold] {self.compute_type}   [bold]Lang:[/bold] {self.language or 'auto'}")
        rprint(f"[bold]Device:[/bold] {self.device}      [bold]Model cache:[/bold] {os.path.abspath(self.model_dir)}")

        # Step 1: Normalize audio
        rprint("[green]Step 1/3:[/green] Normalize audio → 16k mono WAV")
        ffmpeg_norm(input_file, wav)

        # Step 2: Transcribe
        rprint("[green]Step 2/3:[/green] Transcribe with faster-whisper")
        segments, lang = self._transcribe_faster_whisper(wav)

        # Step 3: Align (word-level)
        rprint("[green]Step 3/3:[/green] Align with WhisperX (word-level)")
        aligned = self._align_whisperx(segments, wav, lang)
        aligned_segments = aligned["segments"]

        # Save outputs
        data = {"language": lang, "segments": aligned_segments}
        jpath = f"{base}.aligned.json"
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        srt_path = f"{base}.srt"
        write_srt(aligned_segments, srt_path)

        vtt_path = f"{base}.vtt"
        write_vtt(aligned_segments, vtt_path)

        # Simple default clips = segments
        clips = self._default_clips_from_segments(aligned_segments)
        clips_path = f"{base}.clips.csv"
        with open(clips_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start", "end", "text"])
            for c in clips:
                w.writerow([c["start"], c["end"], c["text"]])

        return {
            "aligned_json": jpath,
            "srt": srt_path,
            "vtt": vtt_path,
            "clips_csv": clips_path,
            "wav": wav,
            "language": lang
        }

# Tubify

Local-first pipeline that pulls a YouTube video, grabs the right Whisper model, and ships aligned transcripts with zero cloud calls. Built for Apple Silicon but works anywhere ffmpeg and yt-dlp run.

## Features
- Downloads and caches Whisper models (via Hugging Face) automatically.
- Normalizes Shorts/watch URLs and fetches videos with yt-dlp.
- Transcribes with faster-whisper, then aligns every word with WhisperX.
- Emits `aligned.json`, `srt`, `vtt`, `clips.csv`, and a normalized 16 kHz WAV.

## Prerequisites
- Python 3.11+
- ffmpeg in your PATH (`brew install ffmpeg` on macOS)
- Optional but recommended: [uv](https://docs.astral.sh/uv/) for zero-setup execution

## Quick Start (uv)
```bash
brew install uv
chmod +x main.py
./main.py --video_url https://www.youtube.com/shorts/QMJAUg2snas --model tiny
```

`uv` will resolve `requirements.txt`, create a local environment under `.uv/`, and run the pipeline without touching your global Python install.

## Classic Virtualenv Setup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

## Run the Pipeline
```bash
./main.py --video_url <youtube_url> [--model large-v3] [--video-dir inputs] \
          [--models-dir models] [--transcribe-dir outputs]
```

Command-line options:
- `--video_url` *(required)*: Shorts or watch URL. Shorts are auto-normalized.
- `--model`: one of `tiny`, `base`, `small`, `medium`, `large-v3` (default `tiny`).
- `--video-dir`: destination for downloads (default `inputs/`).
- `--models-dir`: cache directory for Whisper models (default `models/`).
- `--transcribe-dir`: root folder for outputs (default `outputs/`).

Run output example:
```
outputs/
  Video_Name/
    tiny/
      Video_Name.aligned.json
      Video_Name.srt
      Video_Name.vtt
      Video_Name.clips.csv
      Video_Name_16k.wav
```

## Improving Transcript Quality
- Prefer `large-v3` for Hinglish and subject-heavy content; smaller models trade accuracy for speed.
- Feed clean audio when possible (reduce music beds, record from source files instead of screen captures).
- Force a language if auto-detect slips: edit the transcriber to pass `language="hi"` or `"en"` into faster-whisper.
- Tweak post-processing: the generated `.aligned.json` can be cleaned with your own script/LLM pass for punctuation or terminology fixes.

## Troubleshooting
- `yt-dlp` 403/PO Token warnings: ensure yt-dlp is up to date (`pip install -U yt-dlp`).
- `compute_type 'float16' unsupported`: CTranslate2 fell back to INT8, which is normal on CPU. Install Metal-enabled CTranslate2 and rerun if you need GPU speed (`pip install -U ctranslate2`).
- Alignment slow? WhisperX runs on CPU by default. You can skip alignment by short-circuiting `_align_whisperx` in `transcriber/whisper_transcriber.py` if you only need segment timestamps.

## Repo Layout
- `main.py` – orchestrates model download → video download → transcription.
- `model_downloader/` – Hugging Face fetch helpers.
- `video_downloader/` – yt-dlp wrapper with Shorts normalization.
- `transcriber/` – faster-whisper + WhisperX implementation.
- `inputs/`, `models/`, `outputs/` – default working directories (ignored by git).

## License
MIT

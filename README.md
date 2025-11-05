# ASR Local Agent (Apple Silicon friendly)

Fully local pipeline for transcripts + **word-level timestamps** on Apple Silicon (M-series).
- Transcribe with **faster-whisper**
- Align with **WhisperX** (word timestamps)
- Export **SRT/VTT**, `aligned.json`, and **clips.csv**
- Optional: auto-export subclips (H.264) from `clips.csv` via ffmpeg

## 0) Prereqs
- macOS (Apple Silicon recommended)
- Python 3.11+
- Homebrew `ffmpeg` installed: `brew install ffmpeg`
- Create and activate a venv


## Zero-setup with `uv` (recommended)
This project supports [uv](https://docs.astral.sh/uv/) so you don't need to manage virtualenvs.

### Install uv (macOS)
```bash
brew install uv
```

### Run directly (no venv, no pip)
```bash
chmod +x transcribe.py
./transcribe.py inputs/my_video.mp4 --outdir outputs
```

`uv` will detect `requirements.txt` in the project root, create/reuse a local environment under `.uv/`, and run the script with all deps.

You can also run the helper tools:
```bash
chmod +x cutter/build_clips.py cutter/export_subclips.py
./cutter/build_clips.py outputs/my_video.aligned.json --strategy pause --min-pause 0.40 --max-clip 18.0 --out outputs/my_video.clips.csv
./cutter/export_subclips.py inputs/my_video.mp4 outputs/my_video.clips.csv --outdir outputs/clips
```

> If you prefer classic virtualenvs, the original instructions below still work.


## 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

> Torch uses MPS on Apple Silicon automatically. If you hit a Torch wheel issue, try:
> `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu` (then rerun `pip install whisperx`).
> WhisperX alignment will still work on CPU/MPS; it's slower than CUDA but fine for solo projects.

## 2) Usage

Put your media in `inputs/` or anywhere. Then:

```bash
# Transcribe + align + export captions and CSV
python transcribe.py inputs/my_video.mp4 --outdir outputs
```

You’ll get:
- `outputs/my_video.aligned.json` – segments, each with **words[{start,end,word}]**
- `outputs/my_video.srt` and `outputs/my_video.vtt`
- `outputs/my_video.clips.csv` – simple sentence-level clips (start,end,text)

### Build custom clips from aligned words
```bash
# Strategy can be "pause" (split on silences) or "sentence" (split on punctuation)
python cutter/build_clips.py outputs/my_video.aligned.json   --strategy pause --min-pause 0.40 --max-clip 18.0 --out outputs/my_video.clips.csv
```

### Export actual subclips (mp4) with ffmpeg
```bash
python cutter/export_subclips.py inputs/my_video.mp4 outputs/my_video.clips.csv --outdir outputs/clips
```

## 3) Makefile helpers
```bash
make transcribe IN=inputs/my_video.mp4 OUT=outputs
make clips      JSON=outputs/my_video.aligned.json OUT=outputs/my_video.clips.csv
make subclips   IN=inputs/my_video.mp4 CSV=outputs/my_video.clips.csv OUTDIR=outputs/clips
```

## 4) Notes
- For Hinglish/code-switching, `large-v3` works very well. You can force a language with `--language hi` or `--language en` if auto-detect errs.
- If memory is tight, switch `--model medium` or `--compute-type int8`.
- For interviews, consider adding diarization (not included by default; pyannote is heavy on Mac).

## 5) Outputs format
- **aligned.json**:
```json
{
  "language": "en",
  "segments": [
    {
      "start": 1.12,
      "end": 3.80,
      "text": "Hello world",
      "words": [
        {"start":1.12,"end":1.45,"word":"Hello"},
        {"start":1.46,"end":1.70,"word":"world"}
      ]
    }
  ]
}
```
- **clips.csv**: CSV with header `start,end,text`.

## 6) Troubleshooting
- If Torch refuses MPS, it will still run on CPU; just slower. You can export a smaller model (`--model medium`).
- If WhisperX alignment fails to load an aligner for a rare language, try `--language en` (WhisperX will pick a suitable align model).

Enjoy!

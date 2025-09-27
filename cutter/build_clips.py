#!/usr/bin/env -S uv run python
import argparse, json, csv, re

def load_aligned(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sentence_split(text):
    parts = re.split(r'([.!?]+)\s+', text.strip())
    out = []
    for i in range(0, len(parts), 2):
        t = parts[i].strip()
        p = parts[i+1] if i+1 < len(parts) else ""
        if t:
            out.append((t + (p or "")).strip())
    return [s for s in out if s]

def clips_by_pause(aligned, min_pause=0.35, max_clip=20.0):
    clips = []
    cur_start = None; cur_end = None; cur_text = []; last_end = None
    for seg in aligned["segments"]:
        for w in seg.get("words", []) or [{"start":seg["start"],"end":seg["end"],"word":seg["text"]}]:
            s, e, word = w["start"], w["end"], w.get("word","").strip()
            if cur_start is None:
                cur_start, cur_end, cur_text, last_end = s, e, [word], e
                continue
            gap = s - last_end if last_end is not None else 0
            next_len = e - cur_start
            if gap >= min_pause or next_len >= max_clip:
                clips.append({"start": cur_start, "end": cur_end, "text": " ".join(cur_text)})
                cur_start, cur_end, cur_text = s, e, [word]
            else:
                cur_text.append(word); cur_end = e
            last_end = e
    if cur_start is not None:
        clips.append({"start": cur_start, "end": cur_end, "text": " ".join(cur_text)})
    return clips

def clips_by_sentence(aligned, max_clip=20.0):
    clips = []
    cur_start = None; cur_end = None; cur_text = []
    for seg in aligned["segments"]:
        words = seg.get("words", []) or [{"start":seg["start"],"end":seg["end"],"word":seg["text"]}]
        for w in words:
            s, e, word = w["start"], w["end"], w.get("word","")
            if cur_start is None:
                cur_start, cur_end, cur_text = s, e, [word]
            else:
                cur_text.append(word); cur_end = e
            end_sentence = bool(re.search(r'[.!?]["\']?$', word))
            long_enough = (cur_end - cur_start) >= max_clip
            if end_sentence or long_enough:
                clips.append({"start": cur_start, "end": cur_end, "text": " ".join(cur_text).strip()})
                cur_start, cur_end, cur_text = None, None, []
    if cur_start is not None:
        clips.append({"start": cur_start, "end": cur_end, "text": " ".join(cur_text).strip()})
    return clips

def main():
    ap = argparse.ArgumentParser(description="Build clips.csv from aligned.json")
    ap.add_argument("aligned_json", help="path to .aligned.json")
    ap.add_argument("--strategy", choices=["pause","sentence"], default="pause")
    ap.add_argument("--min-pause", type=float, default=0.40, help="min silence gap (s) for pause-split")
    ap.add_argument("--max-clip", type=float, default=18.0, help="max clip length (s) before split")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    aligned = load_aligned(args.aligned_json)
    if args.strategy == "pause":
        clips = clips_by_pause(aligned, min_pause=args.min_pause, max_clip=args.max_clip)
    else:
        clips = clips_by_sentence(aligned, max_clip=args.max_clip)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start","end","text"])
        for c in clips: w.writerow([c["start"], c["end"], c["text"]])
    print(f"✅ Wrote {len(clips)} clips → {args.out}")

if __name__ == "__main__":
    main()

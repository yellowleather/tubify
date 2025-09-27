IN?=inputs/sample.mp4
OUT?=outputs
JSON?=outputs/sample.aligned.json
CSV?=outputs/sample.clips.csv
OUTDIR?=outputs/clips

.PHONY: transcribe clips subclips

transcribe:
	python run.py $(IN) --outdir $(OUT)

clips:
	python cutter/build_clips.py $(JSON) --strategy pause --min-pause 0.40 --max-clip 18.0 --out $(CSV)

subclips:
	python cutter/export_subclips.py $(IN) $(CSV) --outdir $(OUTDIR)

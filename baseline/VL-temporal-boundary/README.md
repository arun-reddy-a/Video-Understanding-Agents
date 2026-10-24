# Multimodal temporal tree baseline (Ideas 3/4)

For the method, 24-video results, interpretation, and limitations, see
[`TREE_GENERATION_WRITEUP.md`](TREE_GENERATION_WRITEUP.md).

This baseline builds one reusable, query-agnostic JSON tree per Video-MME video.
It is separate from the exploratory Analysis 1 code, while reusing its tested
embedding, subtitle parsing, caching, and dataset-download utilities.

The three levels are the whole-video root, chapters, and leaf sections. The
search/reader stage should pull its 1 fps frames from leaf time ranges; the tree
does not duplicate frame files.

Boundaries come from 0-dimensional persistent homology over a path graph whose
vertex values are a fused visual/subtitle change signal. DINOv2 frame changes
and MiniLM subtitle changes are normalized separately, then max-fused so a
strong change in either modality is preserved instead of averaged away. Strong
persistent maxima form chapters and weaker maxima form sections, with maximum
duration safeguards for quiet videos. Each node's `subtitle_text` is an
extractive summary selected from subtitle windows near that node's embedding
centroid.

## JSON handoff

Trees are written to `trees/<mode>/<video_id>.json`:

```json
{"video_id":"616","tree_mode":"multimodal","nodes":[
  {"id":"n0","level":0,"start":0.0,"end":2645.5,
   "parent":null,"children":["n1"],"subtitle_text":"..."}
]}
```

## Correct Video-MME data path

Run `run_tree_colab.py` from a clone of the repository in a GPU Colab after
mounting Google Drive. It uses the committed stratified-24 manifest and the
pinned official `lmms-eval/Video-MME` revision. It downloads the official
subtitle archive and video chunks sequentially, extracts only the selected
MP4/SRT pairs, deletes each ZIP, and creates a stable pipeline manifest. Source
videos stay in temporary Colab storage; tree JSON and embedding caches persist
in Drive.

Set `MODE = "multimodal"` for the main baseline. Then rerun with
`MODE = "video_only"` for the ablation; both runs reuse cached embeddings.
The video-only JSON deliberately leaves `subtitle_text` empty so language does
not leak into tree construction, node summaries, or downstream search.

For an already downloaded dataset:

```bash
python baseline/VL-temporal-boundary/multimodal_tree.py \
  --dataset-dir /path/to/data \
  --metadata-csv /path/to/data/pipeline_manifest.csv \
  --output-dir /path/to/results --mode multimodal
```

Install dependencies from `analysis/VL-temporal-boundary/requirements-colab.txt`.

## Lightweight local tests

The unit tests use tiny NumPy arrays and monkeypatch every video decoder, model,
network, and dataset operation. They do not download Video-MME or model weights
and do not require a GPU:

```bash
python -m pytest -q baseline/VL-temporal-boundary/tests
```

Running `python -m pytest -q` from the repository root checks both the original
Analysis 1 utilities and this baseline.

If the committed Analysis 1 embedding caches are present, `run_cached_eda.py`
can build both ablations for all 24 videos without downloading MP4s or loading
models. It requires only the corresponding official SRT files:

```bash
python baseline/VL-temporal-boundary/run_cached_eda.py \
  --subtitle-dir /path/to/selected-srt-files
```

It writes 48 trees plus `tree_audit.csv`, `ablation_comparison.csv`, and an
aggregate `audit_summary.json` under `eda_24_results/`.

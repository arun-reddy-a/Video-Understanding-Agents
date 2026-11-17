# Idea 3: question-dependent search + reader (Sai)

This is the second half of Idea 3 (Reusable Multimodal Temporal Hierarchy). Sanay
builds one query-agnostic tree per video (see
[`baseline/VL-temporal-boundary`](../../baseline/VL-temporal-boundary/README.md)).
This folder takes that tree and, for a given question, searches it down to a
fixed budget of frames, then evaluates those frames with the team's
Qwen2.5-VL-7B reader.

Status: this is the **midterm** version of the pipeline. It is being extended
for the final assignment (mixing in uniform frames to fix the whole-video
coverage gap, and giving the reader the subtitles of the picked sections) —
not yet reflected here.

## Files

| File | What it does |
|---|---|
| `tree_search.py` | The search itself: chapters → sections → frames, with optional subtitle scoring. See the module docstring for the exact algorithm. |
| `build_notebook.py` | Generates `tree_baseline_reader_colab.ipynb` from `tree_search.py` plus a fixed set of markdown/code cells, so the notebook and the search code never drift apart. Run `python build_notebook.py` after editing `tree_search.py` and commit both. |
| `tree_baseline_reader_colab.ipynb` | The Colab notebook actually run for the midterm results: indexes the 24 dev videos (SigLIP + MiniLM), runs `search()` for every question and condition, exports frames, runs the reader, and produces the accuracy table and failure-analysis candidates. |
| `reader_karan/` | A copy of Karan's Qwen2.5-VL-7B reader (**not my code** — included here so this folder is self-contained and reproducible; see the note at the top of `reader_karan/README.md`). |

## How the search works (short version)

For each question: score all chapters (visual similarity via SigLIP, optionally
+ subtitle similarity via MiniLM, both z-scored), keep the top 3; score the
sections inside those chapters, keep the top 4; take the best 8 frames per
section. That's 32 frames total, handed to the reader frames-only. Settings
are fixed in advance and are not tuned on Video-MME — see the docstring in
`tree_search.py` for the full specification (fallback filling, de-duplication
gap, etc.).

## Midterm results (72 questions / 24 videos, 32 frames)

| Condition | Accuracy |
|---|---|
| Uniform (lmms-eval baseline) | 59.7% |
| Video-only tree | 54.2% |
| Multimodal tree, visual search only | 52.8% |
| Blind (no frames) | 47.2% |
| Multimodal tree, visual + subtitle search | 45.8% |

None of the tree variants beat uniform at this sample size; only the gap
between the subtitle-search variant and uniform is statistically clear. The
leading hypotheses — a coverage gap (the tree only sees ~16% of each video)
and a modality mismatch (subtitle search can pick evidence the frame-only
reader can't use) — are written up in the team's midterm report.

## Running it

Open `tree_baseline_reader_colab.ipynb` in a GPU Colab runtime (A100 or L4) and
run top to bottom. It clones this repo's `VL-temporal-boundries` branch for the
tree JSONs and dev-video manifest, and expects Karan's reader as
`MyDrive/qwen25_reader.zip` (or use the `reader_karan/` copy in this folder).

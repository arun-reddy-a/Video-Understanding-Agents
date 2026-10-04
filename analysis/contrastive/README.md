# Idea 1 feasibility: does cross-option variance point to answer evidence?

One analysis for Assignment 2, made of three sub-analyses. It tests the premise of
Contrastive Evidence Selection on a small Video-MME subset with a frozen
jina-clip-v2 dual encoder (8k-token text tower, so no truncation of "question + option").

| Sub-analysis | Question | Script | Output |
|---|---|---|---|
| 1. Language | Are the 4 options separable as text? | `a1_language.py` | `outputs/a1_language/` (Fig. 1) |
| 2. Signal | Is cross-option variance above a permutation null? | `a2_signal.py` | `outputs/a2_signal/` (Fig. 2) |
| 3. Vision + task | Do high-variance frames vote for the right answer? | `a3_task.py` | `outputs/a3_task/` (Figs. 3, 4) |

## Data subset
- Dev/test split: 100 dev videos per duration, fixed seed (`data.py`). Only dev videos are used.
- Sub-analysis 1: 25 dev questions per task type (280 questions, text only).
- Sub-analyses 2–3: 20 long dev videos (60 questions), round-robin over domains and
  preferring smaller files (`meta/subset_videos.json`). Videos are extracted one by one
  from the remote HF zip chunks, so only the subset is downloaded.

## Run
```bash
python data.py --n 20      # select + download subset videos -> data/videos
python a1_language.py      # text only
python embed.py            # 1 fps frames -> jina-clip-v2 embeddings -> data/emb
python a2_signal.py
python a3_task.py --options opt_tmpl   # or opt_only
```
`data/` (videos, embeddings) is git-ignored; figures and CSVs in `outputs/` are committed.

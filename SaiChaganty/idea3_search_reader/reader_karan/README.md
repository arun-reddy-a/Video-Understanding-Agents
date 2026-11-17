# Qwen2.5-VL-7B reader for Video-MME

> **Note:** this reader was written by Karan, not Sai. It's copied into
> `SaiChaganty/idea3_search_reader/` as-is so that folder is self-contained and
> reproducible — the search code in `../tree_search.py` feeds this reader
> exactly as described below. Treat this as his code; changes belong upstream
> with him.

This is the reader behind our Video-MME numbers. You give it the frames a selection method picked for each question, and it answers each question with Qwen2.5-VL-7B-Instruct, following lmms-eval's `videomme` protocol. Plug your frame selection in front of it, and your accuracy compares directly with ours.

## Files

| File | What it does |
|---|---|
| `read_videomme.py` | The reader: frames + question → answer, A–D probabilities, correct or not |
| `summarize_videomme.py` | Accuracy per condition, with 95% intervals and paired differences against a baseline (videos resampled) |
| `lmms_protocol/qwen_video_frames.py` | lmms-eval's uniform frame indices and qwen-vl-utils' frame resizing, reproduced exactly |
| `jsonl.py` | Resumable results file (a killed run picks up where it stopped) |
| `pyproject.toml` | Pinned versions (transformers 4.57.1, qwen-vl-utils 0.0.14), matching lmms-eval |
| `example/manifest.jsonl` | Two lines of a real input manifest |
| `example/qwen25_long_results.jsonl` | Our 6,300 result rows on the 900 Video-MME long questions, for comparison |

## Setup (Colab)

The model needs about 17 GB in bf16, so use an L4 or A100 runtime; a T4 won't fit it.

```
pip install transformers==4.57.1 qwen-vl-utils==0.0.14 accelerate decord pyarrow tyro
```

Colab's preinstalled torch works. Upload this folder and run from inside it.

## Input: a bundle directory

```
<bundle>/manifest.jsonl              one line per question
<bundle>/frames/<video_id>/<key>.jpg  the frames your method picked, already resized (see below)
```

Each manifest line holds the question fields from Video-MME (`qid`, `video`, `task`, `question`, `options`, `answer`) plus `conditions`, which maps a condition name to a list of frame keys:

```json
{"qid": "717-1", "video": "068rdc75mHM", "task": "Information Synopsis", "question": "...",
 "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "C",
 "conditions": {"uniform": ["n0", "n3000", ...], "tree": ["t12", "t95", ...], "blind": []}}
```

- **Key names are free.** Ours are `n<native frame>` and `g<second>`. Each key just has to match a jpg in `frames/<video_id>/`.
- **List the keys in time order.** The reader feeds frames in list order, and the model sees no timestamps, so out-of-order frames read as out-of-order video.
- **Use any condition name except ours** (`uniform`, `blind`, `clip_binned`, `evidential`, `evidential_topk`, `clip_logdet_b0.25`, `evidential_logdet_b0.25`), so your rows merge cleanly with our results.
- **An empty list means no frames:** the model gets the question alone (our `blind` condition).

### Making the frames

To see exactly what lmms-eval's Qwen2.5-VL sees, resize the frames the way qwen-vl-utils does:

```python
from lmms_protocol.qwen_video_frames import qwen_max_pixels, resize_like_qwen, lmms_uniform_indices

max_pixels = qwen_max_pixels(total_frames, native_fps)   # per-video cap (235,200 px for any long video)
resized = resize_like_qwen(frames_uint8_nhwc, max_pixels)  # (n, h, w, 3) uint8 -> resized uint8
# save each resized frame as frames/<video_id>/<key>.jpg (we used JPEG quality 95)
```

`lmms_uniform_indices(total_frames, native_fps)` gives the native frame numbers of lmms-eval's own 32 uniform frames, which are our `uniform` condition.

## Run

```
python read_videomme.py --bundle-dir <bundle> --conditions tree        # writes <bundle>/reader_results.jsonl
cat example/qwen25_long_results.jsonl <bundle>/reader_results.jsonl > combined.jsonl
python summarize_videomme.py --results-path combined.jsonl --baseline uniform
```

The summary covers only questions answered in every condition. So run all 900 long questions to compare against our numbers. For a smaller subset, read `uniform` on that same subset too and compare against it.

A read takes about 1 s per question on a 4090 with 32 frames.

## Protocol (fixed; don't change it if you want comparable numbers)

- System prompt "You are a helpful assistant."; the video comes before the text; the prompt is lmms-eval's Video-MME instruction, then the question, the options and "Answer with the option's letter from the given choices directly."
- Greedy decoding, at most 16 new tokens; the prediction is the first A–D in the reply, parsed as lmms-eval does.
- Temporal positions use the processor's default 2 fps, because lmms-eval passes no fps.
- Each row also stores `p`, the first-step probabilities of A–D (lmms-eval doesn't report them).

## Our numbers (Video-MME long, 900 questions, 32 frames)

| Condition | Accuracy |
|---|---|
| blind | 40.3 |
| uniform | 51.2 |
| clip_binned (CLIP top-1 in each of 32 bins) | 53.1 |
| evidential (our Evidential reimplementation) | 55.1 |
| clip_logdet_b0.25 (CLIP relevance + log-det diversity) | 55.2 |

Video-MME is our test set, so don't tune anything on it. Pick settings on another dataset (we use LVBench) and run Video-MME once.

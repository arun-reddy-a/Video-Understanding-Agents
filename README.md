# Video-language temporal boundary analysis

A modular, Colab-oriented pipeline for comparing visual and subtitle change
boundaries in long Video-MME videos. It uses frozen DINOv2 frame embeddings and
MiniLM subtitle-window embeddings, detects peaks, measures bidirectional temporal
alignment, and writes per-video and aggregate reports.

## Google Colab workflow

Select a GPU runtime, mount Drive, clone/upload this repository, and run:

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content/Video-Understanding-Agents
%pip install -q -r requirements-colab.txt
```

Open `video_boundary_analysis_colab.ipynb` and edit its final configuration cell.
The equivalent script workflow is to edit the six values at the top of
`run_colab.py`, then execute `%run run_colab.py`.

`VIDEO_LIST` can be `None` for recursive discovery or a list of video paths
relative to `DATASET_DIR`. Discovery pairs each video with a same-stem `.srt` or
`.vtt`. For other layouts, use `METADATA_CSV`, whose required columns are
`video_path` and `subtitle_path`; `video_id` is optional.

## Outputs

- `cache/`: source- and configuration-aware visual/text embedding caches
- `figures/`: overlaid per-video signals and peak markers
- `reports/`: per-video qualitative event tables
- `aggregate/all_events.csv`: one row per detected event
- `aggregate/video_statistics.csv`: speech/visual characterization
- `aggregate/*.png`: distance, event-count, and matched-strength plots
- `per_video_alignment.csv`: alignment statistics at each tolerance
- `run_config.json`: exact configuration used

Normalization, smoothing, prominence, minimum spacing, tolerances, model names,
batch sizes, and optional peak-frame export are configurable through
`PipelineConfig` in `video_boundary_pipeline.py`.

## Local checks

```bash
pip install -r requirements-colab.txt pytest
pytest -q
```

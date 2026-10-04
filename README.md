# Video-language temporal boundary analysis

A modular, Colab-oriented pipeline for comparing visual and subtitle change
boundaries in long Video-MME videos. It uses frozen DINOv2 frame embeddings and
MiniLM subtitle-window embeddings, detects peaks, measures bidirectional temporal
alignment, and writes per-video and aggregate reports.

## Google Colab workflow

Open `video_boundary_analysis_colab.ipynb` from GitHub in a GPU-enabled Colab
runtime and run it from the top. The first executable cell clones this private
repository into `/content/video-language-temporal-boundaries`, changes into that
directory, validates the required files, and installs dependencies using an
absolute requirements path. It works regardless of Colab's initial directory.

Before running, add a Colab secret named `GITHUB_TOKEN` with read access to this
private repository (key icon in Colab's left sidebar). If the secret is absent,
the setup cell prompts for a token without displaying it. The token is not saved
in the cloned repository's remote URL or Git configuration.

The equivalent script workflow is to clone the repository normally, edit the
six values at the top of `run_colab.py`, then execute `%run run_colab.py`.

`VIDEO_LIST` can be `None` for recursive discovery or a list of video paths
relative to `DATASET_DIR`. Discovery pairs each video with a same-stem `.srt` or
`.vtt`. For other layouts, use `METADATA_CSV`, whose required columns are
`video_path` and `subtitle_path`; `video_id` is optional.

The notebook validates the dataset directory and prints discovered pairs before
it downloads either embedding model. Each video must be beside a same-stem
subtitle, such as `B6fvT2LKEDI.mp4` plus `B6fvT2LKEDI.srt`.

For a first run, the notebook includes an optional two-video trial cell. It
downloads official media from `lmms-eval/Video-MME` into Colab's temporary
`/content` storage, selectively extracts two long MP4/SRT pairs, and deletes the
roughly 5.3 GB ZIP after successful extraction. Allow about 7 GiB of temporary
free space. Embedding caches and analysis outputs still go to Google Drive.

Set `STUDY_MODE = "stratified_24"` for the main exploratory sample. The sampler
uses only long videos, selects exactly 24, guarantees at least three videos from
each of the six domains, and covers distinct subcategories before taking a
second video from any subcategory. Remaining slots are assigned proportionally
to domain size. With seed 0, the current metadata produces domain counts of
5, 4, 4, 4, 4, and 3 and covers 22 domain/subcategory combinations (the
Multilingual domain has only one subcategory). The selection CSV records both
the three-digit `video_id` and source `videoID`. Official archives are downloaded
sequentially, selected assets are extracted, and each ZIP is deleted before
continuing.

## Outputs

- `cache/`: source- and configuration-aware visual/text embedding caches
- `figures/`: overlaid change signals plus one stacked similarity figure per
  video (visual similarity above, subtitle similarity below)
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

The committed notebook defaults to `stratified_24`; switch it to
`two_video_trial` only for a quick smoke test. Analysis outputs are written to
Google Drive by default and are not covered by this repository's `.gitignore`.
If results are copied into the repository, Git will show them as ordinary
untracked files so they can be reviewed and committed deliberately.

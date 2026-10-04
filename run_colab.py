"""Colab execution file: edit only the configuration block."""
from video_boundary_pipeline import PipelineConfig, run_pipeline

# --- EDIT THESE VALUES -----------------------------------------------------
DATASET_DIR = "/content/drive/MyDrive/Video-MME"
OUTPUT_DIR = "/content/drive/MyDrive/video_boundary_results"
VIDEO_LIST = None       # e.g. ["videos/example.mp4"], relative to DATASET_DIR
METADATA_CSV = None     # optional CSV: video_path, subtitle_path[, video_id]
FRAME_INTERVAL = 2.0
SUBTITLE_WINDOW = 15.0
# --------------------------------------------------------------------------

config = PipelineConfig(dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR,
    frame_interval=FRAME_INTERVAL, subtitle_window=SUBTITLE_WINDOW, max_videos=30)
results = run_pipeline(config, video_list=VIDEO_LIST, metadata_csv=METADATA_CSV)
print(results["alignment"])
print(results["stats"])


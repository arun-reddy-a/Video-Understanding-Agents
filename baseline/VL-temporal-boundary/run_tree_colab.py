"""Colab entry point for downloading the fixed sample and building tree JSON."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "analysis" / "VL-temporal-boundary"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from download_selected_subset import download_selected_subset
from multimodal_tree import TreeConfig, run_tree_pipeline
from sample_videos import build_pipeline_manifest

# Persist outputs/caches in Drive; keep the large source videos in temporary Colab storage.
SELECTION_CSV = ANALYSIS_DIR / "manifests" / "long_stratified_24.csv"
DATASET_DIR = Path("/content/Video-MME-stratified-24")
OUTPUT_DIR = Path("/content/drive/MyDrive/video_tree_results")
PIPELINE_MANIFEST = DATASET_DIR / "pipeline_manifest.csv"
MODE = "multimodal"  # rerun as "video_only" for the ablation; embeddings are cached

download_selected_subset(SELECTION_CSV, DATASET_DIR, delete_archives=True)
build_pipeline_manifest(SELECTION_CSV, DATASET_DIR, PIPELINE_MANIFEST)
paths = run_tree_pipeline(TreeConfig(str(DATASET_DIR), str(OUTPUT_DIR), mode=MODE),
                          metadata_csv=str(PIPELINE_MANIFEST))
print(f"Wrote {len(paths)} {MODE} trees to {OUTPUT_DIR / 'trees' / MODE}")

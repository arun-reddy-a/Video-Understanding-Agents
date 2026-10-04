import download_selected_subset
import download_trial_subset
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_selected_downloader_uses_trial_dataset_revision():
    assert download_selected_subset.REVISION == download_trial_subset.REVISION
    assert download_selected_subset.REPO_ID == download_trial_subset.REPO_ID


def test_fixed_manifest_uses_subtitle_backed_acrobatics_video():
    manifest = PROJECT_ROOT / "manifests" / "long_stratified_24.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = {row["videoID"] for row in rows}
    assert "B6fvT2LKEDI" in ids
    assert "7TydWUguPRU" not in ids

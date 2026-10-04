import csv

import pytest

from sample_videos import allocate, build_pipeline_manifest, sample_long_videos, write_selection


def _rows():
    rows = []
    number = 1
    for domain in ("A", "B"):
        for subcategory in ("one", "two"):
            for repeat in range(2):
                rows.append({"video_id": str(number), "videoID": f"asset{number}", "duration": "long",
                             "domain": domain, "sub_category": subcategory, "url": "https://example.test"})
                number += 1
    rows.append({"video_id": "999", "videoID": "short", "duration": "short",
                 "domain": "A", "sub_category": "one", "url": ""})
    return rows


def test_one_long_video_per_stratum():
    selected = sample_long_videos(_rows(), n=4, seed=7, min_per_domain=1)
    assert len(selected) == 4
    assert len({(row["domain"], row["sub_category"]) for row in selected}) == 4
    assert all(row["duration"] == "long" for row in selected)


def test_sample_is_reproducible():
    assert sample_long_videos(_rows(), 6, 3) == sample_long_videos(_rows(), 6, 3)


def test_rejects_too_small_sample():
    with pytest.raises(ValueError):
        sample_long_videos(_rows(), n=5)


def test_pipeline_manifest_uses_three_digit_id(tmp_path):
    selection = write_selection(sample_long_videos(_rows(), 4, 0, min_per_domain=1), tmp_path / "selection.csv")
    chosen = next(csv.DictReader(selection.open(encoding="utf-8")))
    (tmp_path / f"{chosen['videoID']}.mp4").write_bytes(b"video")
    (tmp_path / f"{chosen['videoID']}.srt").write_text("subtitle", encoding="utf-8")
    output = build_pipeline_manifest(selection, tmp_path, tmp_path / "pipeline.csv", require_all=False)
    row = next(csv.DictReader(output.open(encoding="utf-8")))
    assert row["video_id"] == chosen["video_id"]

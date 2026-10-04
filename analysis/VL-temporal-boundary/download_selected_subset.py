"""Download selected Video-MME assets archive-by-archive without retaining ZIPs."""
from __future__ import annotations

import csv
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import requests

from download_trial_subset import REPO_ID, REVISION

ARCHIVE_COUNT = 20


def _download(url: str, destination: Path) -> Path:
    """Stream a public file to disk, resuming a partial transfer when possible."""
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        if offset and response.status_code != 206:
            partial.unlink(missing_ok=True); offset = 0
            response.close()
            return _download(url, destination)
        response.raise_for_status()
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            for block in response.iter_content(8 * 1024**2):
                if block:
                    handle.write(block)
    partial.replace(destination)
    return destination


def _url(filename: str) -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{filename}?download=true"


def _extract_selected(archive_path: Path, wanted: set[str], output: Path,
                      suffix: str) -> set[str]:
    found = set()
    with zipfile.ZipFile(archive_path) as archive:
        entries = {PurePosixPath(item.filename).stem: item for item in archive.infolist()
                   if PurePosixPath(item.filename).suffix.lower() == suffix}
        for asset_id in sorted(wanted & entries.keys()):
            destination = output / f"{asset_id}{suffix}"
            if not destination.exists():
                with archive.open(entries[asset_id]) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024**2)
            found.add(asset_id)
    return found


def download_selected_subset(selection_csv: str | Path, output_dir: str | Path,
                             delete_archives: bool = True) -> Path:
    """Extract selected official videos and subtitles while cycling 20 archives."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    with Path(selection_csv).open(newline="", encoding="utf-8-sig") as handle:
        selected = {row["videoID"] for row in csv.DictReader(handle)}
    if not selected:
        raise ValueError("Selection manifest is empty")

    missing_subs = {asset for asset in selected if not (output / f"{asset}.srt").exists()}
    if missing_subs:
        subtitle_zip = _download(_url("subtitle.zip"), output / "subtitle.zip")
        _extract_selected(subtitle_zip, missing_subs, output, ".srt")
        if delete_archives: subtitle_zip.unlink(missing_ok=True)
        missing_subs = {asset for asset in selected if not (output / f"{asset}.srt").exists()}
        if missing_subs:
            raise RuntimeError(
                "The official subtitle archive does not contain: "
                f"{sorted(missing_subs)}. Update the selection before downloading videos."
            )

    missing_videos = {asset for asset in selected if not (output / f"{asset}.mp4").exists()}
    for number in range(1, ARCHIVE_COUNT + 1):
        if not missing_videos: break
        filename = f"videos_chunked_{number:02d}.zip"
        print(f"[{number}/{ARCHIVE_COUNT}] Downloading {filename}; {len(missing_videos)} videos remain")
        archive_path = _download(_url(filename), output / filename)
        found = _extract_selected(archive_path, missing_videos, output, ".mp4")
        missing_videos -= found
        if found: print("  extracted:", ", ".join(sorted(found)))
        if delete_archives: archive_path.unlink(missing_ok=True)
    missing_subs = {asset for asset in selected if not (output / f"{asset}.srt").exists()}
    if missing_videos or missing_subs:
        raise RuntimeError(f"Missing videos={sorted(missing_videos)}, subtitles={sorted(missing_subs)}")
    print(f"Selected dataset ready: {output} ({len(selected)} pairs)")
    return output

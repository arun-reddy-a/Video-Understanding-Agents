"""Download a two-video Video-MME trial subset from the official HF release."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

REPO_ID = "lmms-eval/Video-MME"
REVISION = "ead1408"
VIDEO_ARCHIVE = "videos_chunked_15.zip"
SUBTITLE_ARCHIVE = "subtitle.zip"
TRIAL_VIDEO_IDS = ("TGom0uiW130", "sDWOsWawxPc")
MIN_FREE_BYTES = 7 * 1024**3


def _extract_member(archive_path: Path, member: str, destination: Path) -> None:
    """Extract one known member without unpacking the rest of the archive."""
    with zipfile.ZipFile(archive_path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as exc:
            raise FileNotFoundError(f"{member} is missing from {archive_path.name}") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024**2)


def download_two_video_trial(
    output_dir: str | Path = "/content/Video-MME-two-video-trial",
    delete_archives: bool = True,
) -> Path:
    """Download, selectively extract, and validate two long video/SRT pairs.

    The archive is about 5.3 GB. Existing complete pairs are reused, which makes
    the cell safe to rerun during the same Colab session.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expected = [output / f"{video_id}{suffix}" for video_id in TRIAL_VIDEO_IDS for suffix in (".mp4", ".srt")]
    if all(path.is_file() and path.stat().st_size > 0 for path in expected):
        print(f"Reusing two existing video/subtitle pairs in {output}")
        return output

    from huggingface_hub import hf_hub_download

    free = shutil.disk_usage(output).free
    if free < MIN_FREE_BYTES:
        raise OSError(f"At least 7 GiB free is required; only {free / 1024**3:.1f} GiB is available")

    print("Downloading the official Video-MME archive (~5.3 GB)...")
    video_zip = Path(hf_hub_download(
        repo_id=REPO_ID, filename=VIDEO_ARCHIVE, repo_type="dataset", revision=REVISION, local_dir=output
    ))
    subtitle_zip = Path(hf_hub_download(
        repo_id=REPO_ID, filename=SUBTITLE_ARCHIVE, repo_type="dataset", revision=REVISION, local_dir=output
    ))

    for video_id in TRIAL_VIDEO_IDS:
        video_path, subtitle_path = output / f"{video_id}.mp4", output / f"{video_id}.srt"
        if not video_path.exists():
            print(f"Extracting {video_path.name}")
            _extract_member(video_zip, f"data/{video_id}.mp4", video_path)
        if not subtitle_path.exists():
            _extract_member(subtitle_zip, f"subtitle/{video_id}.srt", subtitle_path)

    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Trial extraction was incomplete: {missing}")

    if delete_archives:
        video_zip.unlink(missing_ok=True)
        subtitle_zip.unlink(missing_ok=True)
        print("Removed downloaded ZIP archives after successful extraction.")
    print(f"Trial dataset ready: {output}")
    return output


if __name__ == "__main__":
    download_two_video_trial()

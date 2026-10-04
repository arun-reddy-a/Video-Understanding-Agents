import zipfile

from download_trial_subset import _extract_member


def test_extracts_only_requested_zip_member(tmp_path):
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data/keep.mp4", b"video-bytes")
        handle.writestr("data/skip.mp4", b"other-bytes")

    destination = tmp_path / "keep.mp4"
    _extract_member(archive, "data/keep.mp4", destination)

    assert destination.read_bytes() == b"video-bytes"
    assert not (tmp_path / "skip.mp4").exists()

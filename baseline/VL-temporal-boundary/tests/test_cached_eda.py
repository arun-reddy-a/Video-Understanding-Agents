import json
import sys
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parents[1]
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from run_cached_eda import parse_srt_lightweight, validate_tree


def test_lightweight_srt_parser_needs_no_extra_package(tmp_path):
    path = tmp_path / "sample.srt"
    path.write_text(
        "1\n00:00:01,000 --> 00:00:03,500\n<b>Hello</b> world\n\n"
        "2\n00:01:00.000 --> 00:01:02.000\nLater\n",
        encoding="utf-8")
    cues = parse_srt_lightweight(path)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (1.0, 3.5, "Hello world"), (60.0, 62.0, "Later")]


def test_tree_validator_rejects_partition_gap():
    tree = {"nodes": [
        {"id": "n0", "level": 0, "start": 0, "end": 20, "parent": None,
         "children": ["n1"], "subtitle_text": ""},
        {"id": "n1", "level": 1, "start": 0, "end": 20, "parent": "n0",
         "children": ["n2", "n3"], "subtitle_text": ""},
        {"id": "n2", "level": 2, "start": 0, "end": 9, "parent": "n1",
         "children": [], "subtitle_text": ""},
        {"id": "n3", "level": 2, "start": 10, "end": 20, "parent": "n1",
         "children": [], "subtitle_text": ""},
    ]}
    try:
        validate_tree(tree)
        raise AssertionError("Expected a gap to be rejected")
    except ValueError as exc:
        assert "gap or overlap" in str(exc)


def test_colab_notebook_is_valid_and_runs_baseline_entrypoint():
    notebook = json.loads((BASELINE_DIR / "video_tree_baseline_colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert "requirements-colab.txt" in source
    assert "%run baseline/VL-temporal-boundary/run_tree_colab.py" in source

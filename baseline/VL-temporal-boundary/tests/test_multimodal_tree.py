import sys
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BASELINE_DIR = Path(__file__).resolve().parents[1]
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import multimodal_tree as tree_module
from multimodal_tree import (
    TreeConfig, _node_summary, _select_cuts, build_tree, fused_change_signal,
    run_tree_pipeline, superlevel_persistence,
)


def test_superlevel_persistence_ranks_prominent_peaks():
    pairs = superlevel_persistence(np.array([0., 3., 0., 1., 0.]))
    by_index = {p["index"]: p["persistence"] for p in pairs}
    assert by_index[1] > by_index[3] > 0


def test_superlevel_persistence_handles_empty_constant_and_invalid_inputs():
    assert superlevel_persistence(np.array([])) == []
    assert superlevel_persistence(np.ones(5)) == []
    with pytest.raises(ValueError, match="one-dimensional"):
        superlevel_persistence(np.ones((2, 2)))


def test_fusion_interpolates_modalities_and_video_only_ignores_text():
    visual = SimpleNamespace(
        timestamps=np.array([0., 4.]), normalized=np.array([0., 4.]))
    text = SimpleNamespace(
        timestamps=np.array([0., 4.]), normalized=np.array([4., 0.]))
    multimodal = TreeConfig("data", "out", grid_seconds=2,
                            visual_weight=0.75, text_weight=0.25)
    grid, fused = fused_change_signal(visual, text, 4, multimodal)
    assert np.allclose(grid, [0, 2, 4])
    assert np.allclose(fused, [4 / 3, 2, 4])

    video_only = TreeConfig("data", "out", mode="video_only", grid_seconds=2)
    _, visual_only = fused_change_signal(visual, text, 4, video_only)
    assert np.allclose(visual_only, [0, 2, 4])


def test_cut_selection_bounds_quiet_intervals_without_model_outputs():
    grid = np.arange(0, 101, 10, dtype=float)
    assert _select_cuts(grid, [], 100, .5, 10, 30) == [30, 60, 90]
    # A tiny remainder is merged instead of becoming a useless final leaf.
    assert _select_cuts(grid, [], 90.04, .5, 10, 30) == [30, 60]


def test_summary_is_ordered_bounded_and_empty_without_subtitles():
    times = np.array([5., 15., 25.])
    texts = ["first", "representative", "third"]
    embeddings = np.array([[1., 0.], [1., 1.], [0., 1.]], dtype=np.float32)
    summary = _node_summary(0, 30, times, texts, embeddings, count=2, max_chars=20)
    assert len(summary) <= 20
    assert summary in {"first representative", "representative third"}
    assert _node_summary(40, 50, times, texts, embeddings, 2, 20) == ""


def test_tree_is_three_level_partition():
    config = TreeConfig("data", "out", section_min_seconds=10,
                        section_max_seconds=30, chapter_min_seconds=30,
                        chapter_max_seconds=60, summary_windows=2)
    grid = np.arange(0, 121, 10, dtype=float)
    signal = np.array([0, 1, 0, 4, 0, 1, 0, 3, 0, 1, 0, 2, 0], dtype=float)
    times = np.arange(5, 120, 10, dtype=float)
    texts = [f"window {i}" for i in range(len(times))]
    embeddings = np.eye(len(times), dtype=np.float32)
    tree = build_tree("007", 120, grid, signal, [], times, texts, embeddings, config)
    nodes = {node["id"]: node for node in tree["nodes"]}
    assert nodes["n0"]["level"] == 0
    assert {node["level"] for node in nodes.values()} == {0, 1, 2}
    leaves = [node for node in nodes.values() if node["level"] == 2]
    assert leaves[0]["start"] == 0 and leaves[-1]["end"] == 120
    assert all(nodes[leaf["parent"]]["level"] == 1 for leaf in leaves)
    assert all(leaf["end"] - leaf["start"] <= 30 for leaf in leaves)
    assert [node["level"] for node in tree["nodes"]] == sorted(
        node["level"] for node in tree["nodes"])
    assert all(nodes[child]["parent"] == "n0" for child in nodes["n0"]["children"])
    assert all(nodes[child]["start"] >= nodes[parent]["start"] and
               nodes[child]["end"] <= nodes[parent]["end"]
               for parent in nodes for child in nodes[parent]["children"])
    json.dumps(tree)  # The handoff must remain directly JSON serializable.


def test_video_only_config_is_explicit():
    config = TreeConfig("data", "out", mode="video_only")
    config.validate()
    assert config.mode == "video_only"


def test_video_only_tree_does_not_leak_subtitles():
    config = TreeConfig("data", "out", mode="video_only",
                        section_min_seconds=10, section_max_seconds=30,
                        chapter_min_seconds=30, chapter_max_seconds=60)
    tree = build_tree("007", 60, np.arange(0, 61, 10),
                      np.array([0, 1, 0, 2, 0, 1, 0], dtype=float), [],
                      np.array([10., 30., 50.]), ["secret", "subtitle", "text"],
                      np.eye(3, dtype=np.float32), config)
    assert tree["node_summary_source"] == "none"
    assert all(node["subtitle_text"] == "" for node in tree["nodes"])


@pytest.mark.parametrize("changes", [
    {"mode": "unknown"},
    {"fusion_method": "unknown"},
    {"grid_seconds": 0},
    {"visual_weight": -1},
    {"visual_weight": 0, "text_weight": 0},
    {"section_persistence_quantile": 1.1},
    {"section_min_seconds": 20, "section_max_seconds": 10},
])
def test_invalid_configs_fail_before_loading_models(changes):
    values = {"dataset_dir": "data", "output_dir": "out", **changes}
    with pytest.raises(ValueError):
        TreeConfig(**values).validate()


def test_mocked_pipeline_writes_valid_json_without_video_models_or_downloads(tmp_path, monkeypatch):
    """Local end-to-end smoke test; every expensive/external operation is fake."""
    record = SimpleNamespace(video_id="007", video_path=tmp_path / "fake.mp4",
                             subtitle_path=tmp_path / "fake.srt")
    frame_times = np.arange(0, 61, 10, dtype=float)
    visual_embeddings = np.array(
        [[1, 0], [1, 0], [0, 1], [0, 1], [-1, 0], [-1, 0], [0, -1]],
        dtype=np.float32)
    text_times = np.array([7.5, 22.5, 37.5, 52.5])
    text_embeddings = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    texts = ["opening", "demonstration", "explanation", "conclusion"]

    monkeypatch.setattr(tree_module, "discover_videos", lambda *a, **k: [record])
    monkeypatch.setattr(tree_module, "get_device", lambda: "cpu")
    monkeypatch.setattr(tree_module, "load_visual_encoder", lambda *a: ("processor", "visual"))
    monkeypatch.setattr(tree_module, "load_text_encoder", lambda *a: "text")
    monkeypatch.setattr(tree_module, "_video_metadata", lambda *a: (30., 60., 1800))
    monkeypatch.setattr(tree_module, "encode_video_frames",
                        lambda *a, **k: (frame_times, visual_embeddings))
    monkeypatch.setattr(tree_module, "encode_subtitle_windows",
                        lambda *a, **k: (text_times, text_embeddings, texts, []))

    output = tmp_path / "output"
    paths = run_tree_pipeline(TreeConfig(
        str(tmp_path), str(output), frame_interval=10, grid_seconds=5,
        section_min_seconds=10, section_max_seconds=30,
        chapter_min_seconds=30, chapter_max_seconds=60, max_videos=1))

    assert paths == [output / "trees" / "multimodal" / "007.json"]
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["video_id"] == "007"
    assert payload["tree_mode"] == "multimodal"
    assert payload["nodes"][0]["parent"] is None
    assert payload["nodes"][0]["start"] == 0
    assert payload["nodes"][0]["end"] == 60
    assert (output / "tree_config_multimodal.json").is_file()
    assert (output / "cache").is_dir()

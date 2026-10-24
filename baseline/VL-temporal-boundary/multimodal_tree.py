"""Build reusable root -> chapter -> section trees for long Video-MME videos.

Cut candidates are maxima whose connected components persist in a superlevel
filtration of a temporal change signal.  This is 0-dimensional persistent
homology on a path graph; a peak's persistence is the height at which its
component is born minus the height at which it merges into an older component.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Reuse the tested Analysis 1 embedding and signal primitives without modifying
# that exploratory analysis package.
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "analysis" / "VL-temporal-boundary"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from video_boundary_pipeline import (
    PipelineConfig, _make_signal, _video_metadata, build_subtitle_windows,
    discover_videos, encode_subtitle_windows, encode_video_frames, get_device,
    load_text_encoder, load_visual_encoder, parse_subtitles,
)

LOG = logging.getLogger(__name__)


@dataclass
class TreeConfig:
    dataset_dir: str
    output_dir: str
    mode: str = "multimodal"  # multimodal or video_only
    frame_interval: float = 2.0
    subtitle_window: float = 15.0
    grid_seconds: float = 2.0
    fusion_method: str = "max"
    visual_weight: float = 0.5
    text_weight: float = 0.5
    section_min_seconds: float = 45.0
    section_max_seconds: float = 180.0
    chapter_min_seconds: float = 240.0
    chapter_max_seconds: float = 600.0
    section_persistence_quantile: float = 0.55
    chapter_persistence_quantile: float = 0.85
    summary_windows: int = 4
    summary_max_chars: int = 1200
    visual_model: str = "facebook/dinov2-small"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    frame_batch_size: int = 16
    text_batch_size: int = 64
    max_videos: int | None = 24

    def validate(self) -> None:
        if self.mode not in {"multimodal", "video_only"}:
            raise ValueError("mode must be multimodal or video_only")
        if self.fusion_method not in {"max", "mean"}:
            raise ValueError("fusion_method must be max or mean")
        if min(self.frame_interval, self.subtitle_window, self.grid_seconds) <= 0:
            raise ValueError("Sampling intervals must be positive")
        if not 0 <= self.visual_weight or not 0 <= self.text_weight:
            raise ValueError("Modality weights must be non-negative")
        if self.mode == "multimodal" and self.visual_weight + self.text_weight <= 0:
            raise ValueError("At least one modality weight must be positive")
        if not (0 <= self.section_persistence_quantile <= 1 and
                0 <= self.chapter_persistence_quantile <= 1):
            raise ValueError("Persistence quantiles must be in [0, 1]")
        if not (0 < self.section_min_seconds <= self.section_max_seconds):
            raise ValueError("Invalid section duration bounds")
        if not (0 < self.chapter_min_seconds <= self.chapter_max_seconds):
            raise ValueError("Invalid chapter duration bounds")


def superlevel_persistence(values: np.ndarray) -> list[dict]:
    """Return finite H0 peak-persistence pairs for a signal on a path graph."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not len(x):
        return []
    order = sorted(range(len(x)), key=lambda i: (-x[i], i))
    parent = np.arange(len(x)); birth_index = np.arange(len(x))
    active = np.zeros(len(x), dtype=bool); pairs: list[dict] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = int(parent[i])
        return i

    for i in order:
        active[i] = True
        roots = {find(j) for j in (i - 1, i + 1) if 0 <= j < len(x) and active[j]}
        for root in sorted(roots):
            current = find(i); other = find(root)
            if current == other:
                continue
            a, b = birth_index[current], birth_index[other]
            older, younger = (current, other) if (x[a], -a) >= (x[b], -b) else (other, current)
            y = birth_index[younger]
            pairs.append({"index": int(y), "birth": float(x[y]),
                          "death": float(x[i]), "persistence": float(x[y] - x[i])})
            parent[younger] = older
            parent[i] = older
            birth_index[older] = birth_index[older]
    survivor = birth_index[find(int(np.argmax(x)))]
    pairs.append({"index": int(survivor), "birth": float(x[survivor]),
                  "death": float(np.min(x)), "persistence": float(x[survivor] - np.min(x)),
                  "essential": True})
    best: dict[int, dict] = {}
    for pair in pairs:
        if pair["persistence"] > 0 and pair["persistence"] > best.get(pair["index"], {"persistence": -1})["persistence"]:
            best[pair["index"]] = pair
    return sorted(best.values(), key=lambda p: (-p["persistence"], p["index"]))


def _interp(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if not len(times):
        return np.zeros(len(grid), dtype=float)
    return np.interp(grid, times, values, left=values[0], right=values[-1])


def fused_change_signal(visual, text, duration: float, config: TreeConfig):
    grid = np.arange(0.0, duration + config.grid_seconds * 0.5, config.grid_seconds)
    visual_values = _interp(visual.timestamps, visual.normalized, grid)
    if config.mode == "video_only":
        return grid, visual_values
    text_values = _interp(text.timestamps, text.normalized, grid)
    total = config.visual_weight + config.text_weight
    if config.fusion_method == "mean":
        return grid, (config.visual_weight * visual_values + config.text_weight * text_values) / total
    scale = max(config.visual_weight, config.text_weight)
    return grid, np.maximum(config.visual_weight * visual_values,
                            config.text_weight * text_values) / scale


def _select_cuts(grid: np.ndarray, persistence: list[dict], duration: float,
                 quantile: float, min_gap: float, max_gap: float) -> list[float]:
    candidates = [p for p in persistence if 0 < grid[p["index"]] < duration]
    if candidates:
        threshold = float(np.quantile([p["persistence"] for p in candidates], quantile))
        candidates = [p for p in candidates if p["persistence"] >= threshold]
    selected: list[tuple[float, float]] = []
    for pair in candidates:
        t = float(grid[pair["index"]])
        if all(abs(t - old) >= min_gap for old, _ in selected):
            selected.append((t, pair["persistence"]))
    cuts = sorted(t for t, _ in selected)
    # Bound leaf sizes even when the signal has no suitable topological maximum.
    bounded, previous = [], 0.0
    for target in cuts + [duration]:
        while target - previous > max_gap:
            inserted = previous + max_gap
            bounded.append(inserted); previous = inserted
        if target < duration and target - previous >= min_gap:
            bounded.append(target); previous = target
    # A duration just beyond a forced cut can create a useless sub-second tail.
    # Merge that tail into the preceding interval. The merged interval is at
    # most max_gap + min_gap, while every leaf remains useful to the reader.
    if bounded and duration - bounded[-1] < min_gap:
        bounded.pop()
    return bounded


def _node_summary(start: float, end: float, window_times: np.ndarray, texts: list[str],
                  embeddings: np.ndarray, count: int, max_chars: int) -> str:
    indices = np.flatnonzero((window_times >= start) & (window_times < end))
    if not len(indices):
        return ""
    if len(indices) <= count or embeddings.size == 0:
        chosen = indices
    else:
        block = embeddings[indices]
        centroid = block.mean(axis=0)
        scores = block @ centroid / max(np.linalg.norm(centroid), 1e-12)
        chosen = np.sort(indices[np.argsort(scores)[-count:]])
    summary = " ".join(texts[int(i)] for i in chosen)
    return summary[:max_chars].rstrip()


def build_tree(video_id: str, duration: float, grid: np.ndarray, fused: np.ndarray,
               cues, text_times: np.ndarray, texts: list[str], text_embeddings: np.ndarray,
               config: TreeConfig) -> dict:
    persistence = superlevel_persistence(fused)
    section_cuts = _select_cuts(grid, persistence, duration,
                                config.section_persistence_quantile,
                                config.section_min_seconds, config.section_max_seconds)
    chapter_candidates = _select_cuts(grid, persistence, duration,
                                      config.chapter_persistence_quantile,
                                      config.chapter_min_seconds, config.chapter_max_seconds)
    # Chapters must end on section boundaries so the hierarchy is a true partition.
    section_edges = [0.0] + section_cuts + [duration]
    chapter_cuts = []
    for cut in chapter_candidates:
        nearest = min(section_edges[1:-1], key=lambda x: abs(x - cut), default=None)
        if nearest is not None and all(abs(nearest - old) >= config.chapter_min_seconds for old in chapter_cuts):
            chapter_cuts.append(nearest)
    chapter_edges = [0.0] + sorted(set(chapter_cuts)) + [duration]

    def summary(start: float, end: float) -> str:
        if config.mode == "video_only":
            return ""
        return _node_summary(start, end, text_times, texts, text_embeddings,
                             config.summary_windows, config.summary_max_chars)

    chapter_nodes, leaf_nodes = [], []
    root_id = "n0"
    chapters = []
    next_id = 1
    for c_start, c_end in zip(chapter_edges[:-1], chapter_edges[1:]):
        chapter_id = f"n{next_id}"; next_id += 1
        leaves = []
        inside = [x for x in section_edges if c_start < x < c_end]
        for s_start, s_end in zip([c_start] + inside, inside + [c_end]):
            leaf_id = f"n{next_id}"; next_id += 1; leaves.append(leaf_id)
            leaf_nodes.append({"id": leaf_id, "level": 2, "start": round(s_start, 3),
                               "end": round(s_end, 3), "parent": chapter_id, "children": [],
                               "subtitle_text": summary(s_start, s_end)})
        chapters.append(chapter_id)
        chapter_nodes.append({"id": chapter_id, "level": 1, "start": round(c_start, 3),
                              "end": round(c_end, 3), "parent": root_id, "children": leaves,
                              "subtitle_text": summary(c_start, c_end)})
    root = {"id": root_id, "level": 0, "start": 0.0, "end": round(duration, 3),
            "parent": None, "children": chapters,
            "subtitle_text": summary(0, duration)}
    return {"video_id": video_id, "duration_seconds": round(duration, 3),
            "tree_mode": config.mode,
            "node_summary_source": "subtitles" if config.mode == "multimodal" else "none",
            "nodes": [root] + chapter_nodes + leaf_nodes,
            "boundary_diagnostics": {
                "persistence_method": "superlevel_h0_path_graph",
                "section_cut_seconds": [round(x, 3) for x in section_cuts],
                "chapter_cut_seconds": [round(x, 3) for x in chapter_cuts],
                "persistent_peak_count": len(persistence),
            }}


def run_tree_pipeline(config: TreeConfig, metadata_csv: str | None = None) -> list[Path]:
    config.validate(); output = Path(config.output_dir)
    trees, cache = output / "trees" / config.mode, output / "cache"
    trees.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    (output / f"tree_config_{config.mode}.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8")
    records = discover_videos(config.dataset_dir, metadata_csv=metadata_csv,
                              max_videos=config.max_videos)
    if not records:
        raise RuntimeError("No video/subtitle pairs found")
    device = get_device()
    processor, visual_model = load_visual_encoder(config.visual_model, device)
    text_model = load_text_encoder(config.text_model, device)
    signal_config = PipelineConfig(config.dataset_dir, config.output_dir,
        frame_interval=config.frame_interval, subtitle_window=config.subtitle_window,
        smoothing="gaussian", smoothing_seconds=6.0, normalization="zscore")
    written = []
    for number, record in enumerate(records, 1):
        print(f"[{number}/{len(records)}] building {config.mode} tree for {record.video_id}")
        try:
            _, duration, _ = _video_metadata(record.video_path)
            vt, ve = encode_video_frames(record.video_path, config.frame_interval,
                config.frame_batch_size, processor, visual_model, device, cache, config.visual_model)
            tt, te, texts, cues = encode_subtitle_windows(record.subtitle_path,
                config.subtitle_window, duration, text_model, config.text_batch_size,
                cache, config.text_model)
            visual, text = _make_signal(ve, vt, signal_config), _make_signal(te, tt, signal_config)
            grid, fused = fused_change_signal(visual, text, duration, config)
            tree = build_tree(record.video_id, duration, grid, fused, cues, tt, texts, te, config)
            path = trees / f"{record.video_id}.json"
            path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(path)
        except Exception as exc:
            LOG.exception("Skipping %s: %s", record.video_id, exc)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-csv")
    parser.add_argument("--mode", choices=("multimodal", "video_only"), default="multimodal")
    parser.add_argument("--max-videos", type=int, default=24)
    args = parser.parse_args()
    paths = run_tree_pipeline(TreeConfig(args.dataset_dir, args.output_dir,
                                        mode=args.mode, max_videos=args.max_videos),
                              metadata_csv=args.metadata_csv)
    print(f"Wrote {len(paths)} trees")


if __name__ == "__main__":
    main()

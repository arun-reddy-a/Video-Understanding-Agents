"""Build and audit 24-video trees from the committed Analysis 1 embedding cache.

This avoids video downloads and model inference. Only the selected SRT files are
required for human-readable node summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

BASELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_DIR.parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis" / "VL-temporal-boundary"
for folder in (BASELINE_DIR, ANALYSIS_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from multimodal_tree import TreeConfig, build_tree, fused_change_signal
from video_boundary_pipeline import PipelineConfig, SubtitleCue, _make_signal, build_subtitle_windows


def parse_srt_lightweight(path: Path) -> list[SubtitleCue]:
    """Parse ordinary SRT cues without adding a local dependency on pysubs2."""
    text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    cues = []
    stamp = re.compile(
        r"(?P<sh>\d{1,2}):(?P<sm>\d{2}):(?P<ss>\d{2})[,.](?P<sms>\d{3})\s*-->\s*"
        r"(?P<eh>\d{1,2}):(?P<em>\d{2}):(?P<es>\d{2})[,.](?P<ems>\d{3})")
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        match_index = next((i for i, line in enumerate(lines) if stamp.search(line)), None)
        if match_index is None:
            continue
        match = stamp.search(lines[match_index])
        values = {key: int(value) for key, value in match.groupdict().items()}
        start = values["sh"] * 3600 + values["sm"] * 60 + values["ss"] + values["sms"] / 1000
        end = values["eh"] * 3600 + values["em"] * 60 + values["es"] + values["ems"] / 1000
        cue_text = " ".join(lines[match_index + 1:])
        cue_text = re.sub(r"<[^>]+>|\{[^}]+\}", "", cue_text)
        cue_text = " ".join(cue_text.split())
        if cue_text and end > start:
            cues.append(SubtitleCue(start, end, cue_text))
    return sorted(cues, key=lambda cue: cue.start)


def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["timestamps"].copy(), data["embeddings"].copy()


def match_caches(stats: pd.DataFrame, cache_files: list[Path], kind: str) -> dict[str, Path]:
    """Assign caches using duration plus the recorded Analysis 1 peak count."""
    endings, peak_counts = [], []
    signal_config = PipelineConfig("unused", "unused", smoothing="gaussian",
                                   smoothing_seconds=6.0, normalization="zscore")
    for path in cache_files:
        times, embeddings = _load_cache(path)
        endings.append(float(times[-1]) if len(times) else 0.0)
        peak_counts.append(len(_make_signal(embeddings, times, signal_config).peak_indices))
    durations = stats["duration_seconds"].to_numpy(dtype=float)
    expected_peaks = stats[f"{kind}_peaks"].to_numpy(dtype=int)
    duration_cost = np.abs(durations[:, None] - np.asarray(endings)[None, :])
    peak_cost = np.abs(expected_peaks[:, None] - np.asarray(peak_counts)[None, :])
    cost = duration_cost + 1000.0 * peak_cost
    rows, cols = linear_sum_assignment(cost)
    result = {str(stats.iloc[row]["video_id"]): cache_files[col] for row, col in zip(rows, cols)}
    errors = [float(duration_cost[row, col]) for row, col in zip(rows, cols)]
    mismatches = [int(peak_cost[row, col]) for row, col in zip(rows, cols)]
    limit = 3.0 if kind == "visual" else 30.0
    if len(result) != len(stats) or max(errors, default=0) > limit or any(mismatches):
        raise RuntimeError(f"Ambiguous {kind} cache mapping; maximum duration error="
                           f"{max(errors):.2f}s, peak-count mismatches={sum(bool(x) for x in mismatches)}")
    return result


def validate_tree(tree: dict) -> dict:
    nodes = {node["id"]: node for node in tree["nodes"]}
    root = nodes["n0"]
    if root["parent"] is not None or root["level"] != 0:
        raise ValueError("Invalid root")
    for node in nodes.values():
        if node["end"] <= node["start"]:
            raise ValueError(f"Non-positive interval: {node['id']}")
        for child_id in node["children"]:
            child = nodes[child_id]
            if child["parent"] != node["id"] or child["level"] != node["level"] + 1:
                raise ValueError(f"Invalid edge {node['id']} -> {child_id}")
            if child["start"] < node["start"] or child["end"] > node["end"]:
                raise ValueError(f"Child outside parent: {child_id}")
    leaves = sorted((n for n in nodes.values() if n["level"] == 2), key=lambda n: n["start"])
    if not leaves or leaves[0]["start"] != 0 or leaves[-1]["end"] != root["end"]:
        raise ValueError("Leaves do not cover the video")
    for left, right in zip(leaves, leaves[1:]):
        if left["end"] != right["start"]:
            raise ValueError("Leaf partition has a gap or overlap")
    return {
        "chapters": sum(n["level"] == 1 for n in nodes.values()),
        "sections": len(leaves),
        "mean_section_seconds": float(np.mean([n["end"] - n["start"] for n in leaves])),
        "min_section_seconds": float(min(n["end"] - n["start"] for n in leaves)),
        "max_section_seconds": float(max(n["end"] - n["start"] for n in leaves)),
        "empty_summary_fraction": float(np.mean([not n["subtitle_text"] for n in nodes.values()])),
    }


def _matched_fraction(a: list[float], b: list[float], tolerance: float = 10.0) -> float:
    if not a:
        return 1.0 if not b else 0.0
    return float(np.mean([any(abs(x - y) <= tolerance for y in b) for x in a]))


def run_cached_eda(cache_root: Path, stats_csv: Path, selection_csv: Path,
                   subtitle_dir: Path, output_dir: Path) -> pd.DataFrame:
    stats = pd.read_csv(stats_csv, dtype={"video_id": str})
    stats["video_id"] = stats["video_id"].str.zfill(3)
    visual_map = match_caches(stats, sorted(cache_root.glob("visual_*.npz")), "visual")
    text_map = match_caches(stats, sorted(cache_root.glob("text_*.npz")), "text")
    with selection_csv.open(encoding="utf-8-sig", newline="") as handle:
        assets = {row["video_id"].zfill(3): row["videoID"] for row in csv.DictReader(handle)}

    output_dir.mkdir(parents=True, exist_ok=True)
    configs = {
        mode: TreeConfig(str(subtitle_dir), str(output_dir), mode=mode)
        for mode in ("multimodal", "video_only")
    }
    signal_config = PipelineConfig(str(subtitle_dir), str(output_dir), smoothing="gaussian",
                                   smoothing_seconds=6.0, normalization="zscore")
    rows, cuts = [], {}
    for _, stat in stats.iterrows():
        video_id, duration = str(stat.video_id), float(stat.duration_seconds)
        vt, ve = _load_cache(visual_map[video_id]); tt, te = _load_cache(text_map[video_id])
        cues = parse_srt_lightweight(subtitle_dir / f"{assets[video_id]}.srt")
        rebuilt_times, texts = build_subtitle_windows(cues, 15.0, duration)
        if len(rebuilt_times) != len(tt) or len(texts) != len(tt):
            raise RuntimeError(f"Subtitle/cache window mismatch for {video_id}: {len(texts)} != {len(tt)}")
        visual, text = _make_signal(ve, vt, signal_config), _make_signal(te, tt, signal_config)
        for mode, config in configs.items():
            grid, fused = fused_change_signal(visual, text, duration, config)
            tree = build_tree(video_id, duration, grid, fused, cues, tt, texts, te, config)
            target = output_dir / "trees" / mode / f"{video_id}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
            audit = validate_tree(tree)
            rows.append({"video_id": video_id, "mode": mode, **audit})
            cuts[(video_id, mode)] = tree["boundary_diagnostics"]["section_cut_seconds"]
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "tree_audit.csv", index=False)
    comparisons = []
    for video_id in stats.video_id:
        mm, vo = cuts[(video_id, "multimodal")], cuts[(video_id, "video_only")]
        comparisons.append({"video_id": video_id, "multimodal_sections": len(mm) + 1,
                            "video_only_sections": len(vo) + 1,
                            "multimodal_matched_to_video_only_at_10s": _matched_fraction(mm, vo),
                            "video_only_matched_to_multimodal_at_10s": _matched_fraction(vo, mm)})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output_dir / "ablation_comparison.csv", index=False)
    audit_by_mode = {}
    for mode, part in audit.groupby("mode"):
        audit_by_mode[mode] = {
            "chapters_mean": float(part.chapters.mean()),
            "chapters_min": int(part.chapters.min()),
            "chapters_max": int(part.chapters.max()),
            "sections_mean": float(part.sections.mean()),
            "sections_min": int(part.sections.min()),
            "sections_max": int(part.sections.max()),
            "mean_section_seconds": float(part.mean_section_seconds.mean()),
            "shortest_section_seconds": float(part.min_section_seconds.min()),
            "longest_section_seconds": float(part.max_section_seconds.max()),
            "empty_summary_fraction": float(part.empty_summary_fraction.mean()),
        }
    serialized_configs = {}
    for mode, config in configs.items():
        values = asdict(config)
        values.pop("dataset_dir"); values.pop("output_dir")
        serialized_configs[mode] = values
    summary = {
        "videos": len(stats), "modes": list(configs),
        "config": serialized_configs,
        "audit_by_mode": audit_by_mode,
        "mean_multimodal_cut_match_to_video_only_at_10s":
            float(comparison.multimodal_matched_to_video_only_at_10s.mean()),
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    default_results = ANALYSIS_DIR / "video_boundary_results" / "video_boundary_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=default_results / "cache")
    parser.add_argument("--stats-csv", type=Path, default=default_results / "aggregate" / "video_statistics.csv")
    parser.add_argument("--selection-csv", type=Path, default=ANALYSIS_DIR / "manifests" / "long_stratified_24.csv")
    parser.add_argument("--subtitle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=BASELINE_DIR / "eda_24_results")
    args = parser.parse_args()
    audit = run_cached_eda(args.cache_root, args.stats_csv, args.selection_csv,
                           args.subtitle_dir, args.output_dir)
    print(audit.groupby("mode")[["chapters", "sections", "mean_section_seconds",
                                  "empty_summary_fraction"]].mean())


if __name__ == "__main__":
    main()

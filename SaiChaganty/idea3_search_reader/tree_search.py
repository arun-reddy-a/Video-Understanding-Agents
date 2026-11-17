"""Coarse-to-fine search over the temporal trees (combined Ideas 3+4, midterm baseline).

For one question and one tree, pick a fixed budget of frames (default 32):
  1. score every chapter, keep the top K_CHAPTERS
  2. score every section inside those chapters, keep the top K_SECTIONS
  3. inside each kept section, keep the FRAMES_PER_SECTION best-matching 1-fps frames
     (at least MIN_GAP_S seconds apart, so they are not near-duplicates)
  4. if the kept sections cannot supply the whole budget, fill from the next-best sections
  5. return the chosen seconds in time order (the reader sees no timestamps)

Scoring a node (chapter or section):
  visual score   = mean of the node's TOP_FRAMES highest SigLIP similarities between
                   the question text and its 1-fps frames
  subtitle score = cosine similarity between the question (+ options) and the node's
                   subtitle summary, both embedded with MiniLM
  Each score is z-scored across the nodes being compared, because raw scores from
  different encoders are not on the same scale.
  use_subtitles=False -> z(visual)            (video-only search)
  use_subtitles=True  -> z(visual) + z(subtitle) (multimodal search)

Settings are fixed in advance and are not tuned on Video-MME.
"""
from __future__ import annotations

import numpy as np

K_CHAPTERS = 3
K_SECTIONS = 4
FRAMES_PER_SECTION = 8
BUDGET = K_SECTIONS * FRAMES_PER_SECTION  # 32
MIN_GAP_S = 2
TOP_FRAMES = 8


def zscore(values):
    values = np.asarray(values, dtype=float)
    spread = values.std()
    if len(values) < 2 or spread < 1e-8:
        return np.zeros_like(values)
    return (values - values.mean()) / spread


def node_seconds(node, num_seconds):
    """1-fps frame indices that fall inside a node's [start, end) span"""
    start = max(0, int(np.floor(node["start"])))
    end = min(num_seconds, int(np.ceil(node["end"])))
    return np.arange(start, max(start + 1, end)) if start < num_seconds else np.array([], dtype=int)


def visual_score(node, frame_sims):
    seconds = node_seconds(node, len(frame_sims))
    if len(seconds) == 0:
        return -1.0
    sims = np.sort(frame_sims[seconds])[::-1]
    return float(sims[:TOP_FRAMES].mean())


def rank_nodes(nodes, frame_sims, subtitle_sims, use_subtitles):
    """Return (nodes sorted best-first, combined scores in the same order)"""
    visual = zscore([visual_score(node, frame_sims) for node in nodes])
    combined = visual.copy()
    if use_subtitles:
        has_text = np.array([bool(node.get("subtitle_text", "").strip()) for node in nodes])
        raw = np.array([subtitle_sims.get(node["id"], 0.0) for node in nodes])
        text = np.zeros(len(nodes))
        if has_text.sum() >= 2:
            text[has_text] = zscore(raw[has_text])
        combined = visual + text
    order = np.argsort(-combined, kind="stable")
    return [nodes[i] for i in order], combined[order]


def pick_frames_in_section(section, frame_sims, k, taken):
    """Greedy: best-matching seconds first, skipping any within MIN_GAP_S of one already taken"""
    seconds = node_seconds(section, len(frame_sims))
    chosen = []
    for second in seconds[np.argsort(-frame_sims[seconds], kind="stable")]:
        if len(chosen) == k:
            break
        if all(abs(int(second) - other) >= MIN_GAP_S for other in taken + chosen):
            chosen.append(int(second))
    return chosen


def search(tree, frame_sims, subtitle_sims=None, use_subtitles=False, budget=BUDGET):
    """tree: one tree JSON (dict). frame_sims: SigLIP similarity of the question to each
    1-fps frame (index = second). subtitle_sims: {node id: MiniLM similarity}.
    Returns the chosen seconds plus a trace for the failure analysis."""
    frame_sims = np.asarray(frame_sims, dtype=float)
    subtitle_sims = subtitle_sims or {}
    by_id = {node["id"]: node for node in tree["nodes"]}
    chapters = [node for node in tree["nodes"] if node["level"] == 1]

    ranked_chapters, chapter_scores = rank_nodes(chapters, frame_sims, subtitle_sims, use_subtitles)
    kept_chapters = ranked_chapters[:K_CHAPTERS]

    candidate_sections = [by_id[child] for chapter in kept_chapters for child in chapter["children"]]
    ranked_sections, section_scores = rank_nodes(candidate_sections, frame_sims, subtitle_sims, use_subtitles)

    # Sections from the chapters that were not kept, as a fallback to fill the budget
    other_sections = [node for node in tree["nodes"]
                      if node["level"] == 2 and node["id"] not in {s["id"] for s in candidate_sections}]
    if other_sections:
        ranked_other, _ = rank_nodes(other_sections, frame_sims, subtitle_sims, use_subtitles)
    else:
        ranked_other = []

    taken, used_sections = [], []
    for section in ranked_sections[:K_SECTIONS]:
        picked = pick_frames_in_section(section, frame_sims, FRAMES_PER_SECTION, taken)
        taken += picked
        used_sections.append(section["id"])

    for section in ranked_sections[K_SECTIONS:] + ranked_other:
        if len(taken) >= budget:
            break
        picked = pick_frames_in_section(section, frame_sims, budget - len(taken), taken)
        if picked:
            taken += picked
            used_sections.append(section["id"])

    seconds = sorted(taken[:budget])
    covered = sum(by_id[s]["end"] - by_id[s]["start"] for s in used_sections)
    return {
        "seconds": seconds,
        "chapters": [chapter["id"] for chapter in kept_chapters],
        "sections": used_sections,
        "section_spans": [(by_id[s]["start"], by_id[s]["end"]) for s in used_sections],
        "covered_seconds": float(covered),
        "top_chapter_score_gap": float(chapter_scores[0] - chapter_scores[1]) if len(chapter_scores) > 1 else 0.0,
    }

# Multimodal Temporal Tree: Baseline and Initial Analysis

## Goal

This baseline implements the shared direction formed by Ideas 3 and 4. It
builds a reusable hierarchy once per long video, before a question is known.
A separate search component can then traverse that hierarchy and spend a fixed
reader budget only on promising leaf sections. This separates the expensive
video-indexing cost from the per-question search cost.

Analysis 1 showed that visual and linguistic representations do not change at
the same times. Across the stratified 24-video sample, only 5.6% of visual
candidate boundaries and 12.4% of language candidates matched within five
seconds. Visual changes were also about 2.2 times as frequent. The tree therefore
preserves boundaries detected by either modality instead of assuming that one
modality can substitute for the other.

## Data and representations

The experiment uses the same fixed sample of 24 long Video-MME videos used by
Analysis 1. It covers all six Video-MME domains and 22 domain/subcategory
combinations. The selection is recorded in
`analysis/VL-temporal-boundary/manifests/long_stratified_24.csv`.

The baseline uses the representations already computed for Analysis 1:

- RGB frames sampled every two seconds and embedded with frozen DINOv2-small.
- Subtitles grouped into 15-second windows and embedded with frozen
  MiniLM-L6-v2.

For the local validation run, committed embedding caches were reused, so no
videos, model weights, or GPU inference were required. The official subtitle
archive was downloaded at its pinned Video-MME revision and only the selected
24 SRT files were extracted. Cache files were associated with videos using a
one-to-one assignment constrained by both video duration and the exact boundary
count recorded by Analysis 1. This avoids accidental cache assignments when a
subtitle track ends before its video.

## Boundary construction

For normalized consecutive embeddings, the modality-specific change signals
are

\[
d^v_t = 1 - \cos(v_t,v_{t-1}), \qquad
d^s_t = 1 - \cos(s_t,s_{t-1}).
\]

Each signal is smoothed with the same six-second Gaussian kernel and standardized
independently. They are interpolated onto a common two-second grid. The main
condition uses

\[
d_t = \max(d^v_t,d^s_t),
\]

which treats the modalities as a union of boundary evidence. Max fusion was
chosen instead of averaging because averaging can suppress a strong change in
one modality when the other is quiet, contrary to the complementarity observed
in Analysis 1. The ablation uses only \(d^v_t\).

Cut candidates are selected with zero-dimensional persistent homology on the
path graph formed by this temporal signal. In a superlevel filtration, every
local maximum creates a connected component. Its persistence is the difference
between its birth value and the level at which it merges into an older
component. High-persistence maxima are stable over a wider range of thresholds.
The 85th percentile supplies chapter candidates and the 55th percentile supplies
section candidates. Minimum temporal spacing prevents duplicate nearby cuts.
Maximum spans of 600 seconds for chapters and 180 seconds for sections ensure
coverage in low-change intervals. A final interval shorter than the minimum is
merged into its predecessor.

This is specifically H0 persistence over a one-dimensional temporal signal. It
is a lightweight topological baseline, not yet persistent homology over the
full high-dimensional embedding trajectory.

## Tree and handoff

Every JSON file contains three levels:

1. Level 0: one root spanning the full video.
2. Level 1: chapters produced by the strongest persistent boundaries.
3. Level 2: leaf sections produced by the broader persistent-boundary set.

Nodes contain stable IDs, start and end times, parent and child IDs, and a
`subtitle_text` field. In the multimodal condition, the summary is extractive:
subtitle-window embeddings nearest the node centroid are selected and restored
to chronological order. This avoids adding an LLM summarization dependency to
the indexing baseline. The search side can retrieve 1-fps frames directly from
the time range of a selected leaf.

The `video_only` condition uses only visual changes and deliberately leaves all
`subtitle_text` fields empty. Thus subtitles cannot leak into boundary creation,
node summaries, or downstream search. Each JSON records its `tree_mode` and
`node_summary_source` explicitly.

## Validation

The local suite contains unit tests for persistence pairing, signal fusion,
interpolation, cut spacing, tail merging, extractive summaries, invalid
configuration, JSON serialization, hierarchy construction, subtitle leakage,
the dependency-free SRT parser, tree auditing, and the Colab notebook. A mocked
end-to-end test replaces all model, video, dataset, and network operations.

All 34 repository tests pass on a CPU-only local machine. The 24-video cached
run generated 48 trees and validated the following invariants:

- Every node has a positive interval contained within its parent.
- Parent and child pointers agree and levels increase by exactly one.
- Leaf sections form a gap-free, non-overlapping partition of the video.
- All multimodal nodes have summaries; all video-only summaries are empty.
- No leaf is shorter than 45 seconds or longer than 180 seconds in this sample.

## Initial tree-generation results

| Condition | Chapters/video | Sections/video | Mean section length | Observed range |
|---|---:|---:|---:|---:|
| Video + subtitles | 6.42 | 28.29 | 94.38 s | 46–180 s |
| Video only | 6.38 | 29.58 | 90.35 s | 46–180 s |

The main hierarchy is compact: a search agent sees about six root children and
about 28 leaves across a long video, rather than thousands of frames. A roughly
94-second leaf contains about 94 candidate frames at 1 fps; the downstream
search can rank leaves and allocate its total 32-frame reader budget within the
selected ranges.

At a ten-second tolerance, 67.4% of multimodal section cuts have a nearby
video-only cut. Therefore the two trees share most major visual structure, while
about one-third of multimodal cuts move or appear because of subtitle changes.
This is the desired qualitative behavior following Analysis 1: language adds
structure without discarding the visual organization of the video. The ablation
does not yet show that those changed boundaries improve answering accuracy.

Detailed outputs are in `eda_24_results/`: `tree_audit.csv` contains per-video
structure statistics, `ablation_comparison.csv` compares cuts, and
`audit_summary.json` records the configuration and aggregate values. The 48
handoff trees are under `eda_24_results/trees/`.

## Interpretation and next evaluation

The construction is aligned with the revised problem and solution:

- It uses complementary visual and linguistic evidence at indexing time.
- It builds structure once per video and keeps query dependence in traversal.
- It gives the search side a small, explicit coarse-to-fine action space.
- It exposes a clean video-only versus multimodal ablation.
- It supports amortized-cost measurement because indexing and search are
  separate programs.

Tree generation alone cannot establish that the hierarchy selects answer
evidence or improves Video-MME accuracy. The next experiment must run the same
coarse-to-fine search policy and frozen Qwen2.5-VL reader on both tree modes,
with the same 32-frame budget. It should report answer accuracy, correct-answer
probability, frames and tokens consumed, indexing time, query time, and results
against uniform 32-frame sampling. If possible, an oracle-leaf condition should
separate failures of segmentation, search, and reading.

Other limitations are that persistence quantiles and duration constraints are
currently fixed rather than tuned on development data, subtitle summaries are
extractive rather than abstractive, and Video-MME lacks ground-truth evidence
intervals. These are appropriate limitations for a first baseline and should be
reported rather than interpreted as settled design choices.

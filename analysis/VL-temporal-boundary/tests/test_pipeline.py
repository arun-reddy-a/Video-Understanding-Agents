import numpy as np
from video_boundary_pipeline import (SubtitleCue, build_subtitle_windows,
    compute_change_scores, compute_peak_alignment, normalize_signal, subtitle_coverage)

def test_subtitle_windows_overlap_and_skip_empty():
    cues = [SubtitleCue(2, 4, "hello"), SubtitleCue(14, 17, "boundary"), SubtitleCue(31, 32, "later")]
    times, texts = build_subtitle_windows(cues, 15, 45)
    assert np.allclose(times, [7.5, 22.5, 37.5])
    assert texts == ["hello boundary", "boundary", "later"]

def test_change_scores():
    times, scores = compute_change_scores(np.array([[1, 0], [1, 0], [0, 1.]]), np.array([0, 2, 4]))
    assert np.allclose(times, [2, 4]) and np.allclose(scores, [0, 1])

def test_constant_normalization():
    assert np.allclose(normalize_signal(np.ones(5), "zscore"), 0)
    assert np.allclose(normalize_signal(np.ones(5), "robust_percentile"), 0)

def test_bidirectional_alignment():
    table, details = compute_peak_alignment(np.array([10, 50]), np.array([12, 80]), [5, 20])
    assert table.loc[0, "visual_only"] == 1 and table.loc[0, "text_only"] == 1
    assert np.allclose(details["visual_distance"], [2, 30])

def test_coverage_unions_overlaps():
    cues = [SubtitleCue(0, 5, "a"), SubtitleCue(3, 10, "b"), SubtitleCue(20, 25, "c")]
    assert subtitle_coverage(cues, 25) == 0.6

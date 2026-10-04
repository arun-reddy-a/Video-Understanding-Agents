import numpy as np

from video_boundary_pipeline import Signal, plot_modality_similarities


def test_stacked_similarity_plot(tmp_path):
    visual = Signal(np.array([2., 4., 6.]), np.array([.1, .4, .2]), np.array([0., 1., .2]), np.array([1]))
    text = Signal(np.array([15., 30., 45.]), np.array([.2, .1, .5]), np.array([0., .1, 1.]), np.array([2]))
    output = tmp_path / "similarity.png"
    plot_modality_similarities("001", visual, text, output)
    assert output.is_file() and output.stat().st_size > 0

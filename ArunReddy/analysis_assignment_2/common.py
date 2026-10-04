"""Shared encoder wrapper, scoring functions and plot style for the Idea 1 analysis."""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENCODER = "jinaai/jina-clip-v2"
TEMPLATE = "Question: {q} Answer: {o}"

# Task types grouped as in the analysis plan.
TASK_GROUP = {
    "Object Recognition": "Perception", "Attribute Perception": "Perception",
    "Action Recognition": "Perception", "OCR Problems": "Perception",
    "Spatial Perception": "Perception",
    "Object Reasoning": "Reasoning", "Action Reasoning": "Reasoning",
    "Spatial Reasoning": "Reasoning", "Information Synopsis": "Reasoning",
    "Counting Problem": "Counting/Temporal", "Temporal Reasoning": "Counting/Temporal",
    "Temporal Perception": "Counting/Temporal",
}
GROUPS = ["Perception", "Reasoning", "Counting/Temporal"]

# Reference categorical palette (fixed order) and ink colors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


class Encoder:
    """Frozen jina-clip-v2 dual encoder; returns L2-normalised float32 numpy arrays."""

    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained(
            ENCODER, trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()

    def _norm(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def text(self, texts, batch_size=64):
        with self.torch.no_grad():
            return self._norm(self.model.encode_text(list(texts), batch_size=batch_size))

    def images(self, pil_images, batch_size=32):
        with self.torch.no_grad():
            return self._norm(self.model.encode_image(list(pil_images), batch_size=batch_size))


def option_texts(question, options):
    return [TEMPLATE.format(q=question, o=o) for o in options]


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def frame_scores(F, q_emb, opt_emb):
    """F: (T, d) frames, q_emb: (d,), opt_emb: (4, d).

    Returns relevance r (T,), option scores S (T, 4), centred variance v (T,), blend b (T,).
    Each option's mean over frames is subtracted before taking the variance so a
    single option that matches every frame does not dominate.
    """
    r = F @ q_emb
    S = F @ opt_emb.T
    Sc = S - S.mean(axis=0, keepdims=True)
    v = Sc.var(axis=1)
    b = 0.5 * zscore(r) + 0.5 * zscore(v)
    return r, S, v, b


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 1.6, "pdf.fonttype": 42,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def savefig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

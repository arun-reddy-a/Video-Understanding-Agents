"""Sub-analysis 2: is the cross-option variance a real, video-specific signal?

Video-swap null: a question's own option embeddings are scored against the frames of
every *other* subset video. Text (and hence option spread) is held fixed, so only the
visual content changes. The signal z-score compares the real top-K mean variance with
that null. Run for the proposal's "Question + Answer" template and for options alone.
(An option-swap null was tried first but is confounded with option spread: z vs.
spread Spearman ~0.8.)
"""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from a1_language import spread
from common import GROUPS, INK, INK2, SERIES, frame_scores, savefig, style
from data import OUT_DIR
from subset import load_subset

TOP_K, Z_CRIT = 16, 1.645
OUT = os.path.join(OUT_DIR, "a2_signal")
VARIANTS = {"template": "Question + option (proposal)", "option": "Option alone"}


def topk_mean(v, k=TOP_K):
    return np.sort(v)[-k:].mean()


def main():
    os.makedirs(OUT, exist_ok=True)
    questions, frames = load_subset()
    print(f"{len(questions)} questions, {len(frames)} videos, "
          f"{sum(len(f[1]) for f in frames.values())} frames")
    rows = []
    for q in tqdm(questions, desc="a2 video-swap null"):
        F, _ = frames[q["videoID"]]
        others = [frames[v][0] for v in frames if v != q["videoID"]]
        row = dict(question_id=q["question_id"], videoID=q["videoID"], task_type=q["task_type"],
                   group=q["group"], spread_template=spread(q["opt_tmpl"]), spread_option=spread(q["opt_only"]))
        for name, opts in [("template", q["opt_tmpl"]), ("option", q["opt_only"])]:
            s_real = topk_mean(frame_scores(F, q["q_emb"], opts)[2])
            s_null = np.array([topk_mean(frame_scores(G, q["q_emb"], opts)[2]) for G in others])
            row[f"z_{name}"] = (s_real - s_null.mean()) / (s_null.std() + 1e-12)
            row[f"pct_{name}"] = (s_null < s_real).mean()
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "signal_z.csv"), index=False)

    summ = []
    for g in GROUPS + ["All"]:
        d = res if g == "All" else res[res.group == g]
        for v in VARIANTS:
            summ.append(dict(group=g, variant=v, n=len(d), median_z=d[f"z_{v}"].median(),
                             frac_above=(d[f"z_{v}"] > Z_CRIT).mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(summ.round(3).to_string(index=False))
    print("Spearman(spread, z):", res[["spread_template", "z_template", "spread_option", "z_option"]]
          .corr(method="spearman").round(3).to_string())
    plot(res)


def plot(res):
    style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    rng = np.random.default_rng(0)
    groups = GROUPS + ["All"]
    for k, v in enumerate(VARIANTS):
        for gi, g in enumerate(groups):
            z = (res if g == "All" else res[res.group == g])[f"z_{v}"].values
            x = gi + (-0.18 if k == 0 else 0.18)
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(z)), z, s=11, color=SERIES[k], alpha=0.75,
                       linewidths=0.4, edgecolors="white", label=VARIANTS[v] if gi == 0 else None)
            ax.hlines(np.median(z), x - 0.13, x + 0.13, color=INK, linewidth=1.4)
            ax.annotate(f"{(z > Z_CRIT).mean():.0%}", (x, 1.0), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.axhline(Z_CRIT, color=INK2, linewidth=0.9, linestyle="--")
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xticks(range(len(groups)), [f"{g}\n(n={len(res) if g == 'All' else (res.group == g).sum()})"
                                       for g in groups])
    ax.set_ylabel("Signal z-score (own video vs. other videos)")
    ax.set_title("Do the options separate more on their own video?   (% above z = 1.64)", loc="left",
                 color=INK, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)

    ax2.scatter(res.spread_template, res.z_template, s=12, color=SERIES[0], linewidths=0.4,
                edgecolors="white", label=VARIANTS["template"])
    ax2.scatter(res.spread_option, res.z_option, s=12, color=SERIES[1], linewidths=0.4,
                edgecolors="white", label=VARIANTS["option"])
    ax2.axhline(Z_CRIT, color=INK2, linewidth=0.9, linestyle="--")
    ax2.set_xlabel("Option spread (sub-analysis 1)")
    ax2.set_ylabel("Signal z-score")
    ax2.set_title("Signal vs. text separability", loc="left", color=INK, pad=12)
    savefig(fig, OUT, "fig2_signal_vs_null")


if __name__ == "__main__":
    main()

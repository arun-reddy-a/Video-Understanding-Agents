"""Sub-analysis 3 (vision + task): do high-variance frames point to the correct answer?

Four selectors pick k frames per question: uniform, relevance r(t), variance v(t), and
the blend b(t). Without any reader, each selected frame votes for its argmax option
(option scores centred per option across the video), and the majority vote is scored
against the ground truth (chance = 25%). We also report the correct-option margin and
how different/spread-out the selected frames are. Figure 4 is a qualitative example.
"""
import argparse
import os

import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from sklearn.manifold import TSNE
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import gridspec
from common import (INK, INK2, SERIES, bootstrap_ci, frame_scores, savefig, style, zscore)
from data import OUT_DIR, video_path
from subset import load_subset

KS = [4, 8, 16, 32]
SELECTORS = ["Uniform", "Relevance", "Variance", "Blend"]
OUT = os.path.join(OUT_DIR, "a3_task")


def select(T, r, v, b, k):
    k = min(k, T)
    top = lambda s: np.sort(np.argsort(-s)[:k])
    return {"Uniform": np.linspace(0, T - 1, k).round().astype(int),
            "Relevance": top(r), "Variance": top(v), "Blend": top(b)}


def evaluate(S, idx, ans):
    Sc = S - S.mean(axis=0, keepdims=True)
    sel = Sc[idx]
    votes = np.bincount(sel.argmax(axis=1), minlength=4)
    # Ties broken by summed centred score, so the vote is deterministic.
    best = np.flatnonzero(votes == votes.max())
    pred = best[np.argmax(sel[:, best].sum(axis=0))]
    wrong = np.delete(sel, ans, axis=1)
    return float(pred == ans), float((sel[:, ans] - wrong.max(axis=1)).mean())


def main(opt_key):
    os.makedirs(OUT, exist_ok=True)
    questions, frames = load_subset()
    rows, sets = [], []
    for q in tqdm(questions, desc="a3 selectors"):
        F, t = frames[q["videoID"]]
        r, S, v, b = frame_scores(F, q["q_emb"], q[opt_key])
        for k in KS:
            picks = select(len(F), r, v, b, k)
            for name, idx in picks.items():
                acc, margin = evaluate(S, idx, q["answer_idx"])
                rows.append(dict(question_id=q["question_id"], group=q["group"], task_type=q["task_type"],
                                 k=k, selector=name, vote_acc=acc, margin=margin,
                                 span_min=(t[idx].max() - t[idx].min()) / 60,
                                 n_segments=len(np.unique(t[idx] // 60))))
            if k == 16:
                R, V = set(picks["Relevance"]), set(picks["Variance"])
                sets.append(dict(question_id=q["question_id"], jaccard_rel_var=len(R & V) / len(R | V),
                                 spearman_r_v=pd.Series(r).corr(pd.Series(v), method="spearman")))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, f"selectors_{opt_key}.csv"), index=False)
    pd.DataFrame(sets).to_csv(os.path.join(OUT, f"overlap_{opt_key}.csv"), index=False)

    table = (res.groupby(["k", "selector"])
                .agg(vote_acc=("vote_acc", "mean"), margin=("margin", "mean"),
                     span_min=("span_min", "mean"), n_segments=("n_segments", "mean")).unstack("selector"))
    table.to_csv(os.path.join(OUT, f"summary_{opt_key}.csv"))
    print(table.round(3).to_string())
    print(res[res.k == 16].groupby(["group", "selector"]).vote_acc.mean().unstack().round(3).to_string())
    print(pd.DataFrame(sets).describe().round(3).to_string())
    ci = [(s, m, *bootstrap_ci(res[(res.k == 16) & (res.selector == s)][m]))
          for s in SELECTORS for m in ("vote_acc", "margin")]
    ci = pd.DataFrame(ci, columns=["selector", "metric", "mean", "ci_lo", "ci_hi"])
    ci.to_csv(os.path.join(OUT, f"ci_k16_{opt_key}.csv"), index=False)
    print(ci.round(4).to_string(index=False))
    plot_curves(res, opt_key)
    return questions, frames


def plot_curves(res, opt_key):
    style()
    panels = [("All questions", res), ("Perception", res[res.group == "Perception"]),
              ("Reasoning + Counting/Temporal", res[res.group != "Perception"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, (title, d) in zip(axes, panels):
        n = d.question_id.nunique()
        for i, s in enumerate(SELECTORS):
            m = d[d.selector == s].groupby("k").vote_acc.mean().reindex(KS)
            ax.plot(range(len(KS)), m.values, marker="o", ms=4, color=SERIES[i], label=s,
                    markeredgecolor="white", markeredgewidth=0.6)
        ax.axhline(0.25, color=INK2, linestyle="--", linewidth=0.8)
        ax.set_xticks(range(len(KS)), KS)
        ax.set_xlabel("Frames selected (k)")
        ax.set_title(f"{title} (n={n})", loc="left", color=INK)
    axes[0].set_ylabel("Frame-vote accuracy")
    axes[0].text(0, 0.255, "chance", fontsize=6.5, color=INK2)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    savefig(fig, OUT, f"fig3_frame_vote_{opt_key}")


def thumbs(vid, times, size=160):
    vr = VideoReader(video_path(vid), ctx=cpu(0))
    idx = [min(int(s * vr.get_avg_fps()), len(vr) - 1) for s in times]
    out = []
    for f in vr.get_batch(idx).asnumpy():
        h, w = f.shape[:2]
        step = max(1, h // size)
        out.append(f[::step, ::step])
    return out


def plot_example(questions, frames, opt_key, qid=None):
    """Timeline of scores + t-SNE of frames for one question, with selected frames marked."""
    res = pd.read_csv(os.path.join(OUT, f"selectors_{opt_key}.csv"))
    if qid is None:  # a question where variance is right and relevance is not, at k=16
        d = res[res.k == 16].pivot(index="question_id", columns="selector", values="vote_acc")
        cand = d[(d.Variance == 1) & (d.Relevance == 0)].index.tolist() or d.index.tolist()
        qid = cand[0]
    q = next(x for x in questions if x["question_id"] == qid)
    F, t = frames[q["videoID"]]
    r, S, v, b = frame_scores(F, q["q_emb"], q[opt_key])
    picks = select(len(F), r, v, b, 16)
    Sc = S - S.mean(axis=0, keepdims=True)
    tm = t / 60

    style()
    fig = plt.figure(figsize=(7.0, 4.4))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 0.9], width_ratios=[1, 1, 1, 1.25],
                           hspace=0.55, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :3])
    for i in range(4):
        lab = f"{'ABCD'[i]}. {q['options'][i][:38]}" + (" (correct)" if i == q["answer_idx"] else "")
        ax1.plot(tm, Sc[:, i], color=SERIES[i], linewidth=1.6 if i == q["answer_idx"] else 0.8, label=lab)
    ax1.set_ylabel("Option score\n(centred)")
    ax1.set_title(f"{q['task_type']}: {q['question'][:95]}", loc="left", color=INK, fontsize=7.5)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.0, 1.15), frameon=False, fontsize=6)
    ax2 = fig.add_subplot(gs[1, :3], sharex=ax1)
    ax2.plot(tm, zscore(r), color=SERIES[6], linewidth=0.8, label="Relevance r(t)")
    ax2.plot(tm, zscore(v), color=SERIES[7], linewidth=0.8, label="Variance v(t)")
    ymax = max(zscore(r).max(), zscore(v).max())
    for name, c, off in [("Relevance", SERIES[6], 0.6), ("Variance", SERIES[7], 1.2)]:
        ax2.scatter(tm[picks[name]], np.full(16, ymax + off), marker="|", s=40, color=c)
    ax2.set_ylabel("z-score")
    ax2.set_xlabel("Time (min)   (ticks above: the 16 frames each selector picks)")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6)

    ax3 = fig.add_subplot(gs[2, 3])
    Y = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(F)
    ax3.scatter(Y[:, 0], Y[:, 1], c=tm, cmap="Blues", s=2, linewidths=0)
    ax3.scatter(*Y[picks["Relevance"]].T, s=18, marker="o", facecolors="none", edgecolors=SERIES[6],
                linewidths=1.0, label="Relevance")
    ax3.scatter(*Y[picks["Variance"]].T, s=18, marker="^", facecolors="none", edgecolors=SERIES[7],
                linewidths=1.0, label="Variance")
    ax3.set_xticks([]), ax3.set_yticks([])
    ax3.set_title("t-SNE of frames\n(light→dark = time)", loc="left", color=INK, fontsize=7)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6)

    top3 = picks["Variance"][np.argsort(-v[picks["Variance"]])[:3]]
    imgs = thumbs(q["videoID"], t[top3])
    for j, (img, i) in enumerate(zip(imgs, top3)):
        a = fig.add_subplot(gs[2, j])
        a.imshow(img)
        a.set_xticks([]), a.set_yticks([])
        a.grid(False)
        a.set_title(f"top-variance frame, {tm[i]:.1f} min\nvotes {'ABCD'[Sc[i].argmax()]}", fontsize=6.5,
                    color=INK2)
    savefig(fig, OUT, f"fig4_example_{opt_key}")
    print("example question:", qid, q["question"], q["options"], "answer", "ABCD"[q["answer_idx"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--options", choices=["opt_tmpl", "opt_only"], default="opt_tmpl")
    ap.add_argument("--example", default=None, help="question_id for Figure 4")
    args = ap.parse_args()
    qs, fr = main(args.options)
    plot_example(qs, fr, args.options, args.example)

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
    return float(pred == ans), float((sel[:, ans] - wrong.max(axis=1)).mean()), int(pred)


def odd_one_out(E):
    """Index of the option farthest (summed cosine distance) from the other three in text space."""
    return int((1 - E @ E.T).sum(axis=1).argmax())


def main(opt_key):
    os.makedirs(OUT, exist_ok=True)
    questions, frames = load_subset()
    rows, sets = [], []
    for q in tqdm(questions, desc="a3 selectors"):
        F, t = frames[q["videoID"]]
        r, S, v, b = frame_scores(F, q["q_emb"], q[opt_key])
        odd = odd_one_out(q[opt_key])
        whole_acc, _, _ = evaluate(S, np.arange(len(F)), q["answer_idx"])
        for k in KS:
            picks = select(len(F), r, v, b, k)
            for name, idx in picks.items():
                acc, margin, pred = evaluate(S, idx, q["answer_idx"])
                rows.append(dict(question_id=q["question_id"], group=q["group"], task_type=q["task_type"],
                                 k=k, selector=name, vote_acc=acc, margin=margin, votes_odd=float(pred == odd),
                                 odd_is_answer=float(odd == q["answer_idx"]), whole_video_acc=whole_acc,
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
                .agg(vote_acc=("vote_acc", "mean"), margin=("margin", "mean"), votes_odd=("votes_odd", "mean"),
                     span_min=("span_min", "mean"), n_segments=("n_segments", "mean")).unstack("selector"))
    table.to_csv(os.path.join(OUT, f"summary_{opt_key}.csv"))
    print(table.round(3).to_string())
    print(res[res.k == 16].groupby(["group", "selector"]).vote_acc.mean().unstack().round(3).to_string())
    print(pd.DataFrame(sets).describe().round(3).to_string())
    ci = [(s, m, *bootstrap_ci(res[(res.k == 16) & (res.selector == s)][m]))
          for s in SELECTORS for m in ("vote_acc", "margin", "votes_odd")]
    ci = pd.DataFrame(ci, columns=["selector", "metric", "mean", "ci_lo", "ci_hi"])
    ci.to_csv(os.path.join(OUT, f"ci_k16_{opt_key}.csv"), index=False)
    print(ci.round(4).to_string(index=False))
    one = res[(res.k == 16) & (res.selector == "Uniform")]
    print(f"whole-video vote acc {one.whole_video_acc.mean():.3f}; odd-one-out is answer {one.odd_is_answer.mean():.3f}")
    plot_curves(res, opt_key)
    return questions, frames


def plot_curves(res, opt_key):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    n = res.question_id.nunique()
    ax = axes[0]
    for i, s in enumerate(SELECTORS):
        m = res[res.selector == s].groupby("k").vote_acc.mean().reindex(KS)
        ax.plot(range(len(KS)), m.values, marker="o", ms=4, color=SERIES[i], label=s,
                markeredgecolor="white", markeredgewidth=0.6)
    whole = res.drop_duplicates("question_id").whole_video_acc.mean()
    ax.axhline(whole, color=INK, linestyle=":", linewidth=1.0)
    ax.text(len(KS) - 1, whole + 0.01, "all frames", ha="right", va="bottom", fontsize=6.5, color=INK)
    ax.axhline(0.25, color=INK2, linestyle="--", linewidth=0.8)
    ax.text(len(KS) - 1, 0.255, "chance", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(KS)), KS)
    ax.set_xlabel("Frames selected (k)")
    ax.set_ylabel("Frame-vote accuracy")
    ax.set_ylim(0, 0.55)
    ax.set_title(f"(a) Vote accuracy (n={n})", loc="left", color=INK)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=6.3)

    d16 = res[res.k == 16]
    for ax, col, title, ref, ref_lab in [
            (axes[1], "votes_odd", "(b) Votes for text odd-one-out", 0.25, "chance"),
            (axes[2], "n_segments", "(c) Distinct minutes hit", None, None)]:
        stats = [bootstrap_ci(d16[d16.selector == s][col]) for s in SELECTORS]
        m = np.array([x[0] for x in stats])
        err = [m - np.array([x[1] for x in stats]), np.array([x[2] for x in stats]) - m]
        ax.bar(range(4), m, color=SERIES[:4], width=0.62, edgecolor="white", linewidth=1.0)
        ax.errorbar(range(4), m, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.9)
        if ref is not None:
            ax.axhline(ref, color=INK2, linestyle="--", linewidth=0.8)
            ax.text(3.4, ref + 0.01, ref_lab, ha="right", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(range(4), ["Unif.", "Rel.", "Var.", "Blend"])
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("k = 16")
    axes[1].set_ylabel("Share of questions")
    axes[2].set_ylabel("Minutes containing a pick")
    fig.tight_layout(w_pad=1.2)
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
    if qid is None:  # the perception question with the strongest signal in sub-analysis 2
        z = pd.read_csv(os.path.join(OUT_DIR, "a2_signal", "signal_z.csv"))
        qid = z[z.group == "Perception"].sort_values("z_template").question_id.iloc[-1]
    q = next(x for x in questions if x["question_id"] == qid)
    F, t = frames[q["videoID"]]
    r, S, v, b = frame_scores(F, q["q_emb"], q[opt_key])
    picks = select(len(F), r, v, b, 16)
    Sc = S - S.mean(axis=0, keepdims=True)
    tm = t / 60

    smooth = lambda x: pd.Series(x).rolling(15, center=True, min_periods=1).mean().values

    style()
    fig = plt.figure(figsize=(7.0, 4.8))
    gs = gridspec.GridSpec(3, 5, height_ratios=[1, 1, 1.05], width_ratios=[1, 1, 1, 1, 1.25],
                           hspace=0.85, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, :4])
    for i in range(4):
        lab = f"{'ABCD'[i]}. {q['options'][i][:38]}" + (" (correct)" if i == q["answer_idx"] else "")
        ax1.plot(tm, smooth(Sc[:, i]), color=SERIES[i], linewidth=1.8 if i == q["answer_idx"] else 0.9,
                 label=lab, zorder=3 if i == q["answer_idx"] else 2)
    ax1.set_ylabel("Option score\n(centred, 15 s mean)")
    ax1.set_title(f"(a) {q['task_type']}: {q['question'][:95]}", loc="left", color=INK, fontsize=7.5)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.0, 1.1), frameon=False, fontsize=6)
    ax2 = fig.add_subplot(gs[1, :4], sharex=ax1)
    ax2.plot(tm, zscore(r), color=SERIES[6], linewidth=0.7, label="Relevance r(t)")
    ax2.plot(tm, zscore(v), color=SERIES[7], linewidth=0.7, label="Variance v(t)")
    ymax = max(zscore(r).max(), zscore(v).max())
    for name, c, off in [("Relevance", SERIES[6], 0.8), ("Variance", SERIES[7], 1.8)]:
        ax2.scatter(tm[picks[name]], np.full(16, ymax + off), marker="|", s=40, color=c)
    ax2.set_ylabel("z-score")
    ax2.set_xlabel("Time (min); ticks on top = the 16 frames each selector picks")
    ax2.set_title("(b) Relevance vs. cross-option variance", loc="left", color=INK, fontsize=7.5)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6)

    ax3 = fig.add_subplot(gs[2, 4])
    Y = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(F)
    ax3.scatter(Y[:, 0], Y[:, 1], c=tm, cmap="Blues", s=2, linewidths=0)
    ax3.scatter(*Y[picks["Relevance"]].T, s=18, marker="o", facecolors="none", edgecolors=SERIES[6],
                linewidths=1.0, label="Relevance")
    ax3.scatter(*Y[picks["Variance"]].T, s=18, marker="^", facecolors="none", edgecolors=SERIES[7],
                linewidths=1.0, label="Variance")
    ax3.set_xticks([]), ax3.set_yticks([])
    ax3.grid(False)
    ax3.set_title("(d) t-SNE of frames\n(light→dark = time)", loc="left", color=INK, fontsize=7)
    ax3.legend(loc="lower left", bbox_to_anchor=(0.0, -0.42), ncol=2, frameon=False, fontsize=6,
               handletextpad=0.2, columnspacing=0.6)

    def distinct_top(idx, score, n=2, gap=60):
        out = []
        for i in idx[np.argsort(-score[idx])]:
            if all(abs(t[i] - t[j]) >= gap for j in out):
                out.append(i)
            if len(out) == n:
                break
        return out

    shown = [(i, "variance", SERIES[7]) for i in distinct_top(picks["Variance"], v)] + \
            [(i, "relevance", SERIES[6]) for i in distinct_top(picks["Relevance"], r)]
    imgs = thumbs(q["videoID"], t[[i for i, _, _ in shown]])
    for j, (img, (i, kind, c)) in enumerate(zip(imgs, shown)):
        a = fig.add_subplot(gs[2, j])
        a.imshow(img)
        a.set_xticks([]), a.set_yticks([])
        a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(True), sp.set_color(c), sp.set_linewidth(1.6)
        a.set_title(f"{'(c) ' if j == 0 else ''}top {kind}\n{tm[i]:.1f} min, votes {'ABCD'[Sc[i].argmax()]}",
                    fontsize=6.3, color=INK2)
    savefig(fig, OUT, f"fig4_example_{opt_key}")
    print("example question:", qid, q["question"], q["options"], "answer", "ABCD"[q["answer_idx"]])


if __name__ == "__main__":
    import sys
    if "--replot" in sys.argv:
        k = "opt_only" if "opt_only" in sys.argv else "opt_tmpl"
        plot_curves(pd.read_csv(os.path.join(OUT, f"selectors_{k}.csv")), k)
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("--options", choices=["opt_tmpl", "opt_only"], default="opt_tmpl")
    ap.add_argument("--example", default=None, help="question_id for Figure 4")
    args = ap.parse_args()
    qs, fr = main(args.options)
    plot_example(qs, fr, args.options, args.example)

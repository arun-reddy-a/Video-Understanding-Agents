"""Sub-analysis 1 (language modality): are the four answer options separable as text?

For a task-stratified sample of dev-split questions we embed every option with the
jina-clip-v2 text tower, both inside the proposal's "Question + Answer" template and
alone, and measure the option spread (mean pairwise cosine distance of the 4 options).
Options are also tagged as numeric / temporal-order / concrete-visual / abstract using
regex rules plus Empath categories (the interpretable representation).
"""
import itertools
import os
import re

import numpy as np
import pandas as pd
from empath import Empath
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from common import (GRID, INK2, SERIES, TASK_GROUP, Encoder, bootstrap_ci, option_texts,
                    savefig, style)
from data import OUT_DIR, SEED, load_annotations

import matplotlib.pyplot as plt

PER_TASK = 25
OUT = os.path.join(OUT_DIR, "a1_language")

NUM_WORDS = r"(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|hundred)"
NUMERIC = re.compile(rf"^\W*(\d[\d,\.]*|{NUM_WORDS})\b", re.I)
TEMPORAL = re.compile(
    r"\b(before|after|first|then|finally|later|earlier|begin\w*|end\w*|start\w*|order|"
    r"sequence|last|next|followed|while|during|until|since|minutes?|seconds?|hours?)\b|->|→|\(\d\)",
    re.I)
CIRCLED = re.compile(r"[\u2460-\u2473]")
COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black",
          "gray", "grey", "brown", "golden", "silver", "beige"}
CONCRETE_CATS = [
    "animal", "body", "car", "clothing", "cooking", "dance", "eating", "fabric", "fashion",
    "fire", "furniture", "home", "liquid", "ocean", "plant", "shape_and_size", "ship",
    "sports", "swimming", "tool", "toy", "vehicle", "water", "weather", "weapon", "beach",
    "hiking", "driving", "exercise", "appearance", "sailing", "air_travel", "music",
]
TYPES = ["Numeric", "Temporal/order", "Concrete visual", "Abstract"]


def _physical_noun(word):
    """True if the word's most frequent noun sense is a physical object (WordNet)."""
    syns = wn.synsets(word, pos=wn.NOUN)
    if not syns:
        return False
    names = {h.name() for path in syns[0].hypernym_paths() for h in path}
    return "physical_entity.n.01" in names and "abstraction.n.06" not in names \
        and "person.n.01" not in names


def option_type(text, lex, is_permutation=False):
    if NUMERIC.match(text) and len(text.split()) <= 5:
        return "Numeric"
    if is_permutation or CIRCLED.search(text) or TEMPORAL.search(text):
        return "Temporal/order"
    words = re.findall(r"[a-z]+", text.lower())
    cats = lex.analyze(text, categories=CONCRETE_CATS) or {}
    if sum(cats.values()) > 0 or COLORS & set(words) or any(_physical_noun(w) for w in words if len(w) > 2):
        return "Concrete visual"
    return "Abstract"


def is_permutation(options):
    """All options list the same items in a different order (ordering questions)."""
    bags = [sorted(re.findall(r"\w+", o.lower())) for o in options]
    return len(bags[0]) > 1 and all(b == bags[0] for b in bags)


def spread(E):
    """Mean pairwise cosine distance among the rows of E (unit-norm)."""
    return float(np.mean([1 - E[i] @ E[j] for i, j in itertools.combinations(range(len(E)), 2)]))


def sample_questions(df):
    dev = df[df.split == "dev"]
    return (dev.sample(frac=1, random_state=SEED).groupby("task_type").head(PER_TASK)
               .reset_index(drop=True))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = sample_questions(load_annotations())
    print(f"{len(df)} questions over {df.task_type.nunique()} task types")
    enc, lex = Encoder(), Empath()

    rows = []
    for _, q in tqdm(df.iterrows(), total=len(df), desc="a1 embed options"):
        E_tmpl = enc.text(option_texts(q.question, q.options))
        E_opt = enc.text(q.options)
        perm = is_permutation(q.options)
        types = [option_type(o, lex, perm) for o in q.options]
        rows.append(dict(
            question_id=q.question_id, task_type=q.task_type, group=TASK_GROUP[q.task_type],
            duration=q.duration, spread_template=spread(E_tmpl), spread_option=spread(E_opt),
            option_type=max(set(types), key=types.count), n_words=np.mean([len(o.split()) for o in q.options]),
        ))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "option_spread.csv"), index=False)

    summary = (res.groupby("task_type")
                  .agg(n=("question_id", "size"), spread_template=("spread_template", "mean"),
                       spread_option=("spread_option", "mean"))
                  .sort_values("spread_template"))
    summary["compression"] = summary.spread_template / summary.spread_option
    summary.to_csv(os.path.join(OUT, "summary_by_task.csv"))
    print(summary.round(4).to_string())
    print("\nquestion-type mix:\n", pd.crosstab(res.task_type, res.option_type, normalize="index").round(2))
    plot(res, summary.index.tolist())


def plot(res, order):
    style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.6, 1]},
                                  sharey=True)
    y = np.arange(len(order))
    for k, (col, label) in enumerate([("spread_option", "Option alone"),
                                      ("spread_template", "Question + option (proposal template)")]):
        stats = [bootstrap_ci(res.loc[res.task_type == t, col]) for t in order]
        m = np.array([s[0] for s in stats])
        lo, hi = np.array([s[1] for s in stats]), np.array([s[2] for s in stats])
        ax.errorbar(m, y + (0.17 if k == 0 else -0.17), xerr=[m - lo, hi - m], fmt="o", ms=4.5,
                    color=SERIES[k], ecolor=SERIES[k], elinewidth=1.2, capsize=0, label=label,
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.set_yticks(y, order)
    ax.set_xlabel("Option spread (mean pairwise cosine distance)")
    ax.set_title("How different are the 4 options as text?", loc="left", color="#0b0b0b")
    ax.legend(loc="lower right", frameon=False)

    mix = pd.crosstab(res.task_type, res.option_type, normalize="index").reindex(order).reindex(columns=TYPES).fillna(0)
    left = np.zeros(len(order))
    for k, t in enumerate(TYPES):
        ax2.barh(y, mix[t], left=left, color=SERIES[k + 2],
                 height=0.7, edgecolor="white", linewidth=1.0, label=t)
        left += mix[t].values
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Share of questions (dominant option type)")
    ax2.set_title("Option type (regex + Empath + WordNet)", loc="left", color="#0b0b0b")
    ax2.grid(axis="y", visible=False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    savefig(fig, OUT, "fig1_option_spread")


if __name__ == "__main__":
    main()

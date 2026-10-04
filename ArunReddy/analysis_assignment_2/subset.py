"""Load the video subset: frame embeddings plus question/option text embeddings (cached)."""
import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from common import TASK_GROUP, Encoder, option_texts
from data import DATA_DIR, META_DIR, load_annotations

EMB_DIR = os.path.join(DATA_DIR, "emb")
TEXT_CACHE = os.path.join(DATA_DIR, "text_emb.pkl")


def load_subset(enc=None):
    """Returns (questions, frames).

    questions: list of dicts with ids, task info, answer_idx and text embeddings
      q_emb (d,), opt_tmpl (4, d) "Question+Answer" template, opt_only (4, d) option alone.
    frames: {videoID: (emb (T, d) float32, t (T,) seconds)}
    """
    vids = json.load(open(os.path.join(META_DIR, "subset_videos.json")))
    vids = [v for v in vids if os.path.exists(os.path.join(EMB_DIR, f"{v}.npz"))]
    frames = {}
    for v in tqdm(vids, desc="load frame emb"):
        z = np.load(os.path.join(EMB_DIR, f"{v}.npz"))
        frames[v] = (z["emb"].astype(np.float32), z["t"])

    cache = pickle.load(open(TEXT_CACHE, "rb")) if os.path.exists(TEXT_CACHE) else {}
    df = load_annotations()
    df = df[df.videoID.isin(vids)].reset_index(drop=True)
    questions = []
    for _, q in tqdm(df.iterrows(), total=len(df), desc="text emb"):
        if q.question_id not in cache:
            enc = enc or Encoder()
            cache[q.question_id] = dict(
                q_emb=enc.text([q.question])[0],
                opt_tmpl=enc.text(option_texts(q.question, q.options)),
                opt_only=enc.text(q.options),
            )
        questions.append(dict(
            question_id=q.question_id, videoID=q.videoID, task_type=q.task_type,
            group=TASK_GROUP[q.task_type], question=q.question, options=list(q.options),
            answer_idx=int(q.answer_idx), **cache[q.question_id]))
    pickle.dump(cache, open(TEXT_CACHE, "wb"))
    return questions, frames


def embed_foreign_options(enc, question, options):
    """Template embeddings of another question's options under this question."""
    return enc.text(option_texts(question, options))

"""Decode the subset videos at 1 fps and cache jina-clip-v2 frame embeddings.

Writes data/emb/<videoID>.npz with `emb` (T, 1024, float16, unit norm) and `t` (seconds).
"""
import json
import os
import queue
import threading

import numpy as np
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

from common import Encoder
from data import DATA_DIR, META_DIR, video_path

EMB_DIR = os.path.join(DATA_DIR, "emb")
FPS = 1.0
CHUNK = 64


def frame_chunks(path):
    """Yield (timestamps, PIL frames) chunks sampled at FPS; decoding runs in a thread."""
    vr = VideoReader(path, ctx=cpu(0), num_threads=4)
    native = vr.get_avg_fps()
    idx = np.arange(0, len(vr), native / FPS).astype(int)
    q = queue.Queue(maxsize=4)

    def work():
        for s in range(0, len(idx), CHUNK):
            sel = idx[s:s + CHUNK]
            frames = vr.get_batch(sel).asnumpy()
            q.put((sel / native, [Image.fromarray(f) for f in frames]))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    return len(idx), iter(q.get, None)


def main():
    os.makedirs(EMB_DIR, exist_ok=True)
    vids = json.load(open(os.path.join(META_DIR, "subset_videos.json")))
    enc = Encoder()
    for vid in tqdm(vids, desc="videos"):
        out = os.path.join(EMB_DIR, f"{vid}.npz")
        if os.path.exists(out):
            continue
        n, chunks = frame_chunks(video_path(vid))
        embs, ts = [], []
        with tqdm(total=n, desc=f"  {vid} frames", leave=False) as bar:
            for t, frames in chunks:
                embs.append(enc.images(frames, batch_size=CHUNK))
                ts.append(t)
                bar.update(len(frames))
        np.savez(out, emb=np.concatenate(embs).astype(np.float16), t=np.concatenate(ts))


if __name__ == "__main__":
    main()

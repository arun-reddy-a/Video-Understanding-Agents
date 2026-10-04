"""Video-MME annotations, dev/test split, subset selection and partial video download.

Videos are pulled one file at a time out of the remote HF zip chunks (HTTP range
reads), so only the selected subset is ever downloaded.
"""
import argparse
import json
import os
import re
import shutil
import zipfile

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem, hf_hub_download
from tqdm import tqdm

REPO = "lmms-lab/Video-MME"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
OUT_DIR = os.path.join(HERE, "outputs")
META_DIR = os.path.join(HERE, "meta")
SEED = 777

LETTERS = "ABCD"
_PREFIX = re.compile(r"^\s*[A-D][\.\)]\s*")


def strip_option(o):
    return _PREFIX.sub("", o).strip()


def load_annotations():
    path = hf_hub_download(REPO, "videomme/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df["options"] = df["options"].apply(lambda os_: [strip_option(o) for o in os_])
    df["answer_idx"] = df["answer"].map(LETTERS.index)
    df = df.merge(dev_test_split(df), on="videoID")
    return df


def dev_test_split(df, n_dev=100):
    """Per duration split: 100 dev videos, the rest test. Split by video, fixed seed."""
    rng = np.random.default_rng(SEED)
    rows = []
    for dur, g in df.groupby("duration"):
        vids = sorted(g["videoID"].unique())
        rng.shuffle(vids)
        rows += [(v, "dev" if i < n_dev else "test") for i, v in enumerate(vids)]
    return pd.DataFrame(rows, columns=["videoID", "split"])


def select_videos(df, n=20, duration="long", max_mb=700):
    """Round-robin over domains among dev videos, preferring smaller files."""
    index = json.load(open(os.path.join(META_DIR, "zip_index.json")))
    sizes = {k.split("/")[-1].rsplit(".", 1)[0]: v[1] for k, v in index.items()}
    g = df[(df.duration == duration) & (df.split == "dev")].drop_duplicates("videoID").copy()
    g["mb"] = g.videoID.map(sizes) / 1e6
    g = g[g.mb < max_mb]
    pools = {d: list(x.sort_values("mb").videoID) for d, x in g.groupby("domain")}
    chosen = []
    while len(chosen) < n and any(pools.values()):
        for d in sorted(pools):
            if pools[d] and len(chosen) < n:
                chosen.append(pools[d].pop(0))
    return chosen


def download_videos(video_ids):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    index = json.load(open(os.path.join(META_DIR, "zip_index.json")))
    by_id = {k.split("/")[-1].rsplit(".", 1)[0]: (k, v[0]) for k, v in index.items()}
    fs = HfFileSystem()
    zips = {}
    for vid in tqdm(video_ids, desc="download videos"):
        name, chunk = by_id[vid]
        dst = os.path.join(VIDEO_DIR, os.path.basename(name))
        if os.path.exists(dst):
            continue
        if chunk not in zips:
            f = fs.open(f"datasets/{REPO}/videos_chunked_{chunk:02d}.zip", "rb", block_size=8 << 20)
            zips[chunk] = zipfile.ZipFile(f)
        with zips[chunk].open(name) as src, open(dst + ".part", "wb") as out:
            shutil.copyfileobj(src, out, length=8 << 20)
        os.rename(dst + ".part", dst)


def video_path(vid):
    return os.path.join(VIDEO_DIR, f"{vid}.mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    df = load_annotations()
    vids = select_videos(df, n=args.n)
    os.makedirs(META_DIR, exist_ok=True)
    json.dump(vids, open(os.path.join(META_DIR, "subset_videos.json"), "w"), indent=1)
    print(df[df.videoID.isin(vids)].drop_duplicates("videoID")[["videoID", "domain"]].domain.value_counts())
    download_videos(vids)

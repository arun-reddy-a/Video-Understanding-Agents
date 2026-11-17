"""Builds tree_baseline_reader_colab.ipynb (run locally; the notebook is what Sai opens in Colab)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
TREE_SEARCH = (HERE / "tree_search.py").read_text()

cells = []
def md(text): cells.append(("markdown", text.strip("\n")))
def code(text): cells.append(("code", text.strip("\n")))

md("""
# Ideas 3+4 midterm baseline: tree search + Qwen2.5-VL reader

Runs the question-dependent half of the combined idea on the 24 dev videos (72 questions):

| Condition | Tree | Search scores |
|---|---|---|
| `tree_mm` | multimodal (video + subtitles) | SigLIP visual + MiniLM subtitle |
| `tree_mm_vis` | multimodal | SigLIP visual only |
| `tree_vo` | video only | SigLIP visual only |
| `uniform_check` | none | lmms-eval's 32 uniform frames (should reproduce Karan's `uniform`) |

`blind` and `uniform` come from Karan's results file (same reader, same protocol), so we don't rerun them.
Every condition gives the reader exactly 32 frames, and the reader sees frames only (no subtitles), as in lmms-eval's Video-MME protocol.
So any difference between conditions comes from **which** frames were chosen.

Search settings (3 chapters → 4 sections → 8 frames each) are fixed in advance. Don't tune them on these results.

**Before you start:** Runtime → Change runtime type → **A100** (or L4). Upload Karan's `qwen25_reader.zip` to the top of your Google Drive (`MyDrive/qwen25_reader.zip`).
Then run the cells top to bottom. Everything slow is cached in Drive, so if Colab disconnects, just rerun from the top.
""")

md("## 1. Install and set up")
code("""
!pip -q install transformers==4.57.1 qwen-vl-utils==0.0.14 accelerate decord pyarrow tyro sentencepiece protobuf
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")
code("""
from google.colab import drive
drive.mount('/content/drive')

import os, sys, json, csv, time, shutil, subprocess, zipfile
DRIVE = '/content/drive/MyDrive/mmml_tree_baseline'     # cache + final results (survives disconnects)
REPO_DIR = '/content/vua'
READER_DIR = '/content/qwen25_reader'
VIDEO_DIR = '/content/videos'
BUNDLE = '/content/bundle'                              # frames + manifest for the reader (local disk is faster)
for d in [DRIVE, f'{DRIVE}/cache', VIDEO_DIR, BUNDLE]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(REPO_DIR):
    !git clone -q -b VL-temporal-boundries https://github.com/arun-reddy-a/Video-Understanding-Agents.git {REPO_DIR}
if not os.path.exists(READER_DIR):
    !unzip -q -o /content/drive/MyDrive/qwen25_reader.zip -d /content/
print(os.listdir(READER_DIR))
""")

md("## 2. Search code (writes `tree_search.py`)")
code("%%writefile /content/tree_search.py\n" + TREE_SEARCH)
code("""
sys.path.insert(0, '/content')
sys.path.insert(0, READER_DIR)
from tree_search import search
from lmms_protocol.qwen_video_frames import qwen_max_pixels, resize_like_qwen, lmms_uniform_indices
""")

md("## 3. Questions for the 24 dev videos")
code("""
import pandas as pd, numpy as np
from huggingface_hub import hf_hub_download

dev = list(csv.DictReader(open(f'{REPO_DIR}/analysis/VL-temporal-boundary/manifests/long_stratified_24.csv')))
tree_id = {row['videoID']: row['video_id'] for row in dev}       # YouTube id -> tree file name
videos = sorted(tree_id)

try:
    meta_path = hf_hub_download('lmms-lab/Video-MME', 'videomme/test-00000-of-00001.parquet', repo_type='dataset')
except Exception:
    meta_path = hf_hub_download('lmms-eval/Video-MME', 'videomme/test-00000-of-00001.parquet', repo_type='dataset', revision='ead1408')
meta = pd.read_parquet(meta_path)
qs = meta[meta.videoID.isin(videos)].reset_index(drop=True)
print(len(qs), 'questions on', qs.videoID.nunique(), 'videos')   # expect 72 on 24

karan = [json.loads(line) for line in open(f'{READER_DIR}/example/qwen25_long_results.jsonl')]
karan_qids = {r['qid'] for r in karan if r['video'] in tree_id}
assert karan_qids == set(qs.question_id), 'question ids do not match Karan\\'s results'
print('question ids match Karan\\'s results file')
""")

md("## 4. Download the 24 videos (~7.4 GB, pulled one file at a time from the HF zips)")
code("""
from huggingface_hub import HfFileSystem
index = json.load(open(f'{REPO_DIR}/ArunReddy/analysis_assignment_2/meta/zip_index.json'))
where = {k.split('/')[-1].rsplit('.', 1)[0]: (k, v[0]) for k, v in index.items()}

fs, zips = HfFileSystem(), {}
for vid in videos:
    dst = f'{VIDEO_DIR}/{vid}.mp4'
    if os.path.exists(dst):
        continue
    name, chunk = where[vid]
    if chunk not in zips:
        zips[chunk] = zipfile.ZipFile(fs.open(f'datasets/lmms-lab/Video-MME/videos_chunked_{chunk:02d}.zip', 'rb', block_size=8 << 20))
    t0 = time.time()
    with zips[chunk].open(name) as src, open(dst + '.part', 'wb') as out:
        shutil.copyfileobj(src, out, length=8 << 20)
    os.rename(dst + '.part', dst)
    print(f'{vid}  {os.path.getsize(dst)/1e6:.0f} MB  {time.time()-t0:.0f}s')
print(len(os.listdir(VIDEO_DIR)), 'videos ready')
""")

md("""
## 5. Index each video once: SigLIP embeddings of 1-fps frames

This is the query-agnostic part (together with Sanay's tree). Times are saved for the cost table.
""")
code("""
import torch
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModel, AutoProcessor, AutoTokenizer

SIGLIP = 'google/siglip-so400m-patch14-384'
sig_model = AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).cuda().eval()
sig_proc = AutoProcessor.from_pretrained(SIGLIP)
SIZE = 384
MEAN = torch.tensor(sig_proc.image_processor.image_mean, device='cuda').view(1, 3, 1, 1)
STD = torch.tensor(sig_proc.image_processor.image_std, device='cuda').view(1, 3, 1, 1)

def decode_1fps(vid):
    t0 = time.time()
    cmd = ['ffmpeg', '-v', 'error', '-threads', '0', '-i', f'{VIDEO_DIR}/{vid}.mp4',
           '-vf', f'fps=1,scale={SIZE}:{SIZE}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return vid, np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE, 3), time.time() - t0

@torch.no_grad()
def embed_frames(frames, batch=128):
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i:i+batch]).cuda().permute(0, 3, 1, 2).float() / 255
        x = ((x - MEAN) / STD).half()
        out.append(torch.nn.functional.normalize(sig_model.get_image_features(pixel_values=x).float(), dim=-1).cpu())
    return torch.cat(out).numpy().astype(np.float16)

todo = [v for v in videos if not os.path.exists(f'{DRIVE}/cache/{v}.npz')]
with ThreadPoolExecutor(max_workers=3) as pool:          # decode 3 videos on CPU while the GPU embeds
    for vid, frames, decode_s in pool.map(decode_1fps, todo):
        t0 = time.time()
        emb = embed_frames(frames)
        np.savez(f'{DRIVE}/cache/{vid}.npz', emb=emb, decode_s=decode_s, embed_s=time.time() - t0)
        print(f'{vid}: {len(emb)} frames  decode {decode_s:.0f}s  embed {time.time()-t0:.0f}s')
print('all videos indexed')
""")

md("## 6. Question and node text embeddings")
code("""
MINILM = 'sentence-transformers/all-MiniLM-L6-v2'          # same text encoder Sanay used for the tree
mini_tok = AutoTokenizer.from_pretrained(MINILM)
mini = AutoModel.from_pretrained(MINILM).cuda().eval()

@torch.no_grad()
def minilm(texts, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        b = mini_tok(texts[i:i+batch], padding=True, truncation=True, max_length=256, return_tensors='pt').to('cuda')
        h = mini(**b).last_hidden_state
        m = b['attention_mask'].unsqueeze(-1).float()
        out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def siglip_text(texts):
    b = sig_proc.tokenizer(texts, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to('cuda')
    return torch.nn.functional.normalize(sig_model.get_text_features(**b).float(), dim=-1).cpu().numpy()

TREES = f'{REPO_DIR}/baseline/VL-temporal-boundary/eda_24_results/trees'
trees = {mode: {vid: json.load(open(f'{TREES}/{mode}/{tree_id[vid]}.json')) for vid in videos}
         for mode in ['multimodal', 'video_only']}

# Node summaries are part of the index (query-agnostic): embed them once per video
node_text_s, node_emb = {}, {}
for vid in videos:
    t0 = time.time()
    nodes = [n for n in trees['multimodal'][vid]['nodes'] if n.get('subtitle_text', '').strip()]
    embs = minilm([n['subtitle_text'] for n in nodes])
    node_emb[vid] = {n['id']: e for n, e in zip(nodes, embs)}
    node_text_s[vid] = time.time() - t0

qs['options_list'] = qs.options.apply(list)
q_sig = siglip_text(list(qs.question))                                     # visual search: question text
q_mini = minilm([q + ' ' + ' '.join(o) for q, o in zip(qs.question, qs.options_list)])  # subtitle search: question + options
print(q_sig.shape, q_mini.shape)
""")

md("## 7. Run the search for every question and condition")
code("""
from decord import VideoReader, cpu

CONDITIONS = {'tree_mm': ('multimodal', True), 'tree_mm_vis': ('multimodal', False), 'tree_vo': ('video_only', False)}
emb = {vid: np.load(f'{DRIVE}/cache/{vid}.npz')['emb'].astype(np.float32) for vid in videos}
video_info = {}
for vid in videos:
    vr = VideoReader(f'{VIDEO_DIR}/{vid}.mp4', ctx=cpu(0))
    video_info[vid] = (len(vr), vr.get_avg_fps())

traces, manifest = [], []
for i, q in qs.iterrows():
    vid = q.videoID
    frame_sims = emb[vid] @ q_sig[i]
    num_frames, fps = video_info[vid]
    conditions = {'uniform_check': [f'n{k}' for k in lmms_uniform_indices(num_frames, fps)]}
    for name, (mode, use_subs) in CONDITIONS.items():
        sub_sims = {nid: float(e @ q_mini[i]) for nid, e in node_emb[vid].items()} if use_subs else {}
        t0 = time.time()
        result = search(trees[mode][vid], frame_sims, sub_sims, use_subtitles=use_subs)
        result.update(qid=q.question_id, video=vid, condition=name, search_s=time.time() - t0)
        traces.append(result)
        conditions[name] = [f'g{s}' for s in result['seconds']]
    manifest.append({'qid': q.question_id, 'video': vid, 'task': q.task_type, 'question': q.question,
                     'options': q.options_list, 'answer': q.answer,
                     'native_fps': fps, 'num_frames': num_frames, 'conditions': conditions})

assert all(len(m['conditions'][c]) == 32 for m in manifest for c in CONDITIONS)
json.dump(traces, open(f'{DRIVE}/search_traces.json', 'w'), indent=1)
print(len(manifest), 'questions,', len(traces), 'searches')
""")

md("## 8. Export the chosen frames exactly the way the reader expects")
code("""
from PIL import Image
for vid in videos:
    num_frames, fps = video_info[vid]
    keys = sorted({k for m in manifest if m['video'] == vid for ks in m['conditions'].values() for k in ks})
    out_dir = f'{BUNDLE}/frames/{vid}'
    os.makedirs(out_dir, exist_ok=True)
    keys = [k for k in keys if not os.path.exists(f'{out_dir}/{k}.jpg')]
    if not keys:
        continue
    native = [int(k[1:]) if k[0] == 'n' else min(round(int(k[1:]) * fps), num_frames - 1) for k in keys]
    vr = VideoReader(f'{VIDEO_DIR}/{vid}.mp4', ctx=cpu(0))
    max_pixels = qwen_max_pixels(num_frames, fps)
    for start in range(0, len(keys), 32):
        frames = vr.get_batch(native[start:start+32]).asnumpy()
        for key, frame in zip(keys[start:start+32], resize_like_qwen(frames, max_pixels)):
            Image.fromarray(frame).save(f'{out_dir}/{key}.jpg', quality=95)
    print(vid, len(keys), 'frames')

with open(f'{BUNDLE}/manifest.jsonl', 'w') as f:
    for m in manifest:
        f.write(json.dumps(m) + '\\n')
shutil.copy(f'{BUNDLE}/manifest.jsonl', f'{DRIVE}/manifest.jsonl')
""")

md("## 9. Run the reader (288 reads; free GPU memory first)")
code("""
del sig_model, mini
import gc; gc.collect(); torch.cuda.empty_cache()

# Resume: copy any earlier results back from Drive so finished reads are skipped
if os.path.exists(f'{DRIVE}/reader_results.jsonl') and not os.path.exists(f'{BUNDLE}/reader_results.jsonl'):
    shutil.copy(f'{DRIVE}/reader_results.jsonl', f'{BUNDLE}/reader_results.jsonl')
""")
code("""
%cd {READER_DIR}
!python read_videomme.py --bundle-dir {BUNDLE} --conditions uniform_check tree_mm tree_mm_vis tree_vo
shutil.copy(f'{BUNDLE}/reader_results.jsonl', f'{DRIVE}/reader_results.jsonl')
""")

md("## 10. Accuracy table (with Karan's `blind` and `uniform` on the same 72 questions)")
code("""
%cd {READER_DIR}
ours = [json.loads(line) for line in open(f'{BUNDLE}/reader_results.jsonl')]
dev_q = set(qs.question_id)
karan_dev = [r for r in karan if r['qid'] in dev_q and r['condition'] in ('blind', 'uniform')]
with open(f'{DRIVE}/combined_results.jsonl', 'w') as f:
    for r in karan_dev + ours:
        f.write(json.dumps(r) + '\\n')

# Sanity check: our uniform frames should give (almost) the same answers as Karan's uniform
k_uni = {r['qid']: r['pred'] for r in karan_dev if r['condition'] == 'uniform'}
o_uni = {r['qid']: r['pred'] for r in ours if r['condition'] == 'uniform_check'}
print(f"uniform_check agrees with Karan's uniform on {np.mean([k_uni[q] == o_uni[q] for q in o_uni]):.0%} of questions\\n")

!python summarize_videomme.py --results-path {DRIVE}/combined_results.jsonl --baseline uniform
""")

md("## 11. Cost, per-task accuracy, and failure-analysis candidates")
code("""
res = pd.DataFrame(karan_dev + ours)
tr = pd.DataFrame(traces)

# Accuracy by task type
by_task = res.pivot_table(index='task', columns='condition', values='correct', aggfunc='mean').round(2)
by_task['n'] = res[res.condition == 'uniform'].groupby('task').size()
display(by_task)

# Confidence in the correct answer and tokens
display(res.groupby('condition')[['correct', 'p_correct', 'n_tokens', 'seconds']].mean().round(3))

# Coverage: how much of the video the chosen sections span
display(tr.groupby('condition')['covered_seconds'].describe().round(0))

# Cost: index once per video, then search + read per question
idx = pd.DataFrame([{**{k: float(v) for k, v in np.load(f'{DRIVE}/cache/{v_}.npz').items() if k != 'emb'},
                     'video': v_, 'node_text_s': node_text_s[v_]} for v_ in videos])
idx['index_s'] = idx.decode_s + idx.embed_s + idx.node_text_s
print(f"index per video: {idx.index_s.mean():.1f}s (decode {idx.decode_s.mean():.1f} + SigLIP {idx.embed_s.mean():.1f} + node text {idx.node_text_s.mean():.2f}); add Sanay's tree-build time")
print(f"search per question: {tr.search_s.mean()*1000:.2f} ms")
print(f"index cost per question with 3 questions/video: {idx.index_s.mean()/3:.1f}s")

by_task.to_csv(f'{DRIVE}/accuracy_by_task.csv'); idx.to_csv(f'{DRIVE}/index_cost.csv'); res.to_csv(f'{DRIVE}/all_results.csv', index=False)
""")
code("""
# Paired outcomes: multimodal tree vs uniform, and multimodal tree vs video-only tree
wide = res.pivot_table(index='qid', columns='condition', values='correct', aggfunc='first')
for a, b in [('tree_mm', 'uniform'), ('tree_mm', 'tree_vo'), ('tree_mm', 'tree_mm_vis')]:
    print(f'{a} vs {b}:'); print(pd.crosstab(wide[a], wide[b], rownames=[a], colnames=[b])); print()

# Questions to inspect by hand: uniform right but tree_mm wrong (losses) and the reverse (wins)
q_by_id = qs.set_index('question_id')
tmm = tr[tr.condition == 'tree_mm'].set_index('qid')
for label, mask in [('LOSS (uniform right, tree_mm wrong)', (wide.uniform == 1) & (wide.tree_mm == 0)),
                    ('WIN (tree_mm right, uniform wrong)', (wide.uniform == 0) & (wide.tree_mm == 1))]:
    print('=' * 20, label, int(mask.sum()))
    for qid in wide[mask].index:
        q, t = q_by_id.loc[qid], tmm.loc[qid]
        spans = ', '.join(f'{a/60:.1f}-{b/60:.1f} min' for a, b in t.section_spans[:4])
        print(f'{qid} [{q.task_type}] {q.question}\\n   answer {q.answer}: {[o for o in q.options if o.startswith(q.answer)][0]}\\n   chose sections at {spans}\\n')
""")

nb = {
    "cells": [
        {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True),
         **({"execution_count": None, "outputs": []} if kind == "code" else {})}
        for kind, src in cells
    ],
    "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 0,
}
(HERE / "tree_baseline_reader_colab.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote", len(cells), "cells")

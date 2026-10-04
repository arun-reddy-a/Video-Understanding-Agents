"""Modular visual/language temporal-boundary analysis for long videos."""
from __future__ import annotations

import hashlib, json, logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks

LOG = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}
EVENT_COLUMNS = ["video_id", "modality", "timestamp_seconds", "visual_score", "text_score",
                 "nearest_other_peak_seconds", "event_type", "subtitle_text"]


@dataclass
class PipelineConfig:
    dataset_dir: str
    output_dir: str
    frame_interval: float = 2.0
    subtitle_window: float = 15.0
    frame_batch_size: int = 16
    text_batch_size: int = 64
    visual_model: str = "facebook/dinov2-small"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalization: str = "zscore"
    robust_percentiles: tuple[float, float] = (5.0, 95.0)
    smoothing: str = "gaussian"
    smoothing_seconds: float = 6.0
    peak_prominence: float = 0.8
    peak_min_spacing: float = 20.0
    alignment_tolerances: tuple[float, ...] = (5.0, 10.0, 20.0)
    max_videos: int | None = 30
    qualitative_context: float = 10.0
    save_peak_frames: bool = False

    def validate(self):
        if self.frame_interval <= 0 or self.subtitle_window <= 0:
            raise ValueError("Sampling intervals must be positive")
        if self.normalization not in {"zscore", "robust_percentile"}:
            raise ValueError("Unknown normalization")
        if self.smoothing not in {"none", "moving_average", "gaussian"}:
            raise ValueError("Unknown smoothing")


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    video_path: Path
    subtitle_path: Path


@dataclass
class Signal:
    timestamps: np.ndarray
    raw: np.ndarray
    normalized: np.ndarray
    peak_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def peak_times(self): return self.timestamps[self.peak_indices]
    @property
    def peak_scores(self): return self.normalized[self.peak_indices]


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({p.total_memory / 2**30:.1f} GiB)")
    return device


def discover_videos(dataset_dir, video_list=None, metadata_csv=None, max_videos=30):
    """Pair videos/subtitles or read video_path, subtitle_path[, video_id] CSV."""
    root, candidates = Path(dataset_dir).expanduser(), []
    if metadata_csv:
        table = pd.read_csv(metadata_csv)
        if not {"video_path", "subtitle_path"}.issubset(table.columns):
            raise ValueError("Metadata needs video_path and subtitle_path columns")
        for _, row in table.iterrows():
            v, s = Path(str(row.video_path)), Path(str(row.subtitle_path))
            v, s = (v if v.is_absolute() else root / v), (s if s.is_absolute() else root / s)
            candidates.append((str(row.video_id) if "video_id" in table else v.stem, v, s))
    else:
        videos = ([Path(v) if Path(v).is_absolute() else root / v for v in video_list]
                  if video_list else sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS))
        for v in videos:
            s = next((v.with_suffix(ext) for ext in (".srt", ".vtt") if v.with_suffix(ext).exists()), None)
            candidates.append((v.stem, v, s))
    result, ids = [], {}
    for raw_id, video, subtitle in candidates:
        if not video.exists() or subtitle is None or not subtitle.exists():
            LOG.warning("Skipping missing pair: %s / %s", video, subtitle); continue
        count = ids.get(raw_id, 0); ids[raw_id] = count + 1
        vid = raw_id if not count else f"{raw_id}_{count + 1}"
        result.append(VideoRecord(vid, video, subtitle))
        if max_videos is not None and len(result) >= max_videos: break
    return result


def _video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise OSError(f"Cannot open video: {path}")
    fps, count = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0: raise OSError(f"Invalid FPS: {path}")
    return fps, count / fps, count


def sample_video_frames(video_path, interval_seconds=2.0, batch_size=16) -> Iterator[tuple[np.ndarray, list[Image.Image]]]:
    """Seek and yield bounded timestamped RGB batches; tolerate bad frames."""
    _, duration, _ = _video_metadata(video_path)
    cap, times, frames = cv2.VideoCapture(str(video_path)), [], []
    try:
        for t in np.arange(0, duration, interval_seconds):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                LOG.warning("Unreadable frame at %.2fs in %s", t, video_path); continue
            times.append(float(t)); frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if len(frames) == batch_size:
                yield np.asarray(times), frames; times, frames = [], []
        if frames: yield np.asarray(times), frames
    finally: cap.release()


def load_visual_encoder(name, device):
    from transformers import AutoImageProcessor, AutoModel
    return AutoImageProcessor.from_pretrained(name), AutoModel.from_pretrained(name).to(device).eval()


@torch.inference_mode()
def encode_frames(frames, processor, model, device):
    inputs = {k: v.to(device) for k, v in processor(images=list(frames), return_tensors="pt").items()}
    out = model(**inputs)
    x = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
    return F.normalize(x.float(), dim=1).cpu().numpy()


def _fingerprint(path, settings):
    p, stat = Path(path).resolve(), Path(path).stat()
    data = {"path": str(p), "size": stat.st_size, "mtime": stat.st_mtime_ns, **settings}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def encode_video_frames(video_path, interval_seconds, batch_size, processor, model, device, cache_dir, model_name):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"visual_{_fingerprint(video_path, {'interval': interval_seconds, 'model': model_name})}.npz"
    if cache.exists():
        with np.load(cache) as d: return d["timestamps"], d["embeddings"]
    times, embeddings = [], []
    for ts, frames in sample_video_frames(video_path, interval_seconds, batch_size):
        times.append(ts); embeddings.append(encode_frames(frames, processor, model, device))
    if not embeddings: raise RuntimeError(f"No readable frames: {video_path}")
    ts, emb = np.concatenate(times), np.concatenate(embeddings)
    np.savez_compressed(cache, timestamps=ts, embeddings=emb)
    return ts, emb


def parse_subtitles(path):
    import pysubs2
    cues = []
    for item in pysubs2.load(str(path), encoding="utf-8"):
        text = " ".join(item.plaintext.split())
        if text and item.end > item.start: cues.append(SubtitleCue(item.start / 1000, item.end / 1000, text))
    return sorted(cues, key=lambda c: c.start)


def build_subtitle_windows(cues, window_seconds=15.0, duration=None):
    if not cues: return np.array([], dtype=float), []
    duration = duration if duration is not None else max(c.end for c in cues)
    times, texts = [], []
    for start in np.arange(0, duration, window_seconds):
        end = min(start + window_seconds, duration)
        overlap = [c.text for c in cues if c.start < end and c.end > start]
        if overlap: times.append((start + end) / 2); texts.append(" ".join(overlap))
    return np.asarray(times), texts


def load_text_encoder(name, device):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=str(device))


def encode_text(texts, model, batch_size=64):
    if not texts: return np.empty((0, 0), dtype=np.float32)
    return np.asarray(model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False))


def encode_subtitle_windows(path, window, duration, model, batch_size, cache_dir, model_name):
    cues = parse_subtitles(path); times, texts = build_subtitle_windows(cues, window, duration)
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"text_{_fingerprint(path, {'window': window, 'duration': round(duration, 3), 'model': model_name})}.npz"
    if cache.exists():
        with np.load(cache) as d: return d["timestamps"], d["embeddings"], texts, cues
    emb = encode_text(texts, model, batch_size)
    np.savez_compressed(cache, timestamps=times, embeddings=emb)
    return times, emb, texts, cues


def compute_change_scores(embeddings, timestamps):
    if len(embeddings) < 2: return np.array([], dtype=float), np.array([], dtype=float)
    x = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    return np.asarray(timestamps[1:], dtype=float), (1 - np.sum(x[1:] * x[:-1], axis=1)).astype(float)


def smooth_signal(values, timestamps, method="gaussian", seconds=6.0):
    if method == "none" or len(values) < 3 or seconds <= 0: return values.copy()
    step = np.median(np.diff(timestamps)); samples = max(seconds / max(step, 1e-9), 1)
    if method == "gaussian": return gaussian_filter1d(values, samples, mode="nearest")
    if method == "moving_average": return uniform_filter1d(values, max(1, round(samples)), mode="nearest")
    raise ValueError(method)


def normalize_signal(values, method="zscore", percentiles=(5.0, 95.0)):
    values = np.asarray(values, dtype=float)
    if not len(values): return values.copy()
    if method == "zscore":
        std = values.std(); return (values - values.mean()) / std if std > 1e-12 else np.zeros_like(values)
    if method == "robust_percentile":
        low, high = np.percentile(values, percentiles)
        return np.clip((values - low) / (high - low), 0, 1) if high - low > 1e-12 else np.zeros_like(values)
    raise ValueError(method)


def detect_peaks(values, timestamps, prominence=.8, min_spacing_seconds=20.0):
    if len(values) < 3: return np.array([], dtype=int)
    candidates, props = find_peaks(values, prominence=prominence)
    order, kept = candidates[np.argsort(props["prominences"])[::-1]], []
    for idx in order:
        if all(abs(timestamps[idx] - timestamps[j]) >= min_spacing_seconds for j in kept): kept.append(int(idx))
    return np.asarray(sorted(kept), dtype=int)


def _nearest(source, target):
    if not len(source): return np.array([], dtype=int), np.array([], dtype=float)
    if not len(target): return np.full(len(source), -1), np.full(len(source), np.inf)
    d = np.abs(source[:, None] - target[None, :]); idx = d.argmin(axis=1)
    return idx, d[np.arange(len(source)), idx]


def compute_peak_alignment(visual_times, text_times, tolerances=(5., 10., 20.)):
    vi, vd = _nearest(visual_times, text_times); ti, td = _nearest(text_times, visual_times)
    rows = []
    for tol in tolerances:
        va, ta = vd <= tol, td <= tol
        rows.append({"tolerance_seconds": tol,
                     "visual_alignment_fraction": va.mean() if len(va) else np.nan,
                     "text_alignment_fraction": ta.mean() if len(ta) else np.nan,
                     "visual_only": int((~va).sum()), "text_only": int((~ta).sum()),
                     "aligned_pairs": len({(i, int(vi[i])) for i in np.flatnonzero(va)})})
    return pd.DataFrame(rows), {"visual_nearest_index": vi, "visual_distance": vd,
                                "text_nearest_index": ti, "text_distance": td}


def subtitle_text_near(cues, timestamp, context=10.0):
    return " ".join(c.text for c in cues if c.start <= timestamp + context and c.end >= timestamp - context)


def _score_near(signal, timestamp):
    return float(signal.normalized[np.argmin(abs(signal.timestamps - timestamp))]) if len(signal.timestamps) else np.nan


def build_event_table(video_id, visual, text, cues, tolerance=10.0, context=10.0):
    rows = []; vi, vd = _nearest(visual.peak_times, text.peak_times); ti, td = _nearest(text.peak_times, visual.peak_times)
    for i, (t, score) in enumerate(zip(visual.peak_times, visual.peak_scores)):
        aligned = vd[i] <= tolerance
        rows.append({"video_id": video_id, "modality": "visual", "timestamp_seconds": t,
                     "visual_score": score, "text_score": float(text.peak_scores[vi[i]]) if aligned else _score_near(text, t),
                     "nearest_other_peak_seconds": vd[i], "event_type": "aligned" if aligned else "visual-only",
                     "subtitle_text": subtitle_text_near(cues, t, context)})
    for i, (t, score) in enumerate(zip(text.peak_times, text.peak_scores)):
        aligned = td[i] <= tolerance
        rows.append({"video_id": video_id, "modality": "text", "timestamp_seconds": t,
                     "visual_score": float(visual.peak_scores[ti[i]]) if aligned else _score_near(visual, t),
                     "text_score": score, "nearest_other_peak_seconds": td[i],
                     "event_type": "aligned" if aligned else "text-only", "subtitle_text": subtitle_text_near(cues, t, context)})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def plot_video_signals(video_id, visual, text, output_path):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(visual.timestamps / 60, visual.normalized, lw=1, label="Visual change")
    ax.plot(text.timestamps / 60, text.normalized, lw=1.2, label="Language change")
    ax.scatter(visual.peak_times / 60, visual.peak_scores, s=28, label="Visual peaks")
    ax.scatter(text.peak_times / 60, text.peak_scores, s=35, marker="x", label="Text peaks")
    ax.set(title=f"Temporal boundary signals — {video_id}", xlabel="Video time (minutes)", ylabel="Normalized change")
    ax.grid(alpha=.2); ax.legend(ncol=4); fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(output_path, dpi=160); plt.close(fig)


def extract_frame_at(video_path, timestamp, output_path):
    cap = cv2.VideoCapture(str(video_path)); cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = cap.read(); cap.release()
    if not ok: return False
    Path(output_path).parent.mkdir(parents=True, exist_ok=True); return bool(cv2.imwrite(str(output_path), frame))


def subtitle_coverage(cues, duration):
    if not cues or duration <= 0: return 0.
    spans = sorted((max(0., c.start), min(duration, c.end)) for c in cues
                   if c.end > 0 and c.start < duration)
    if not spans: return 0.
    total, start, end = 0., spans[0][0], spans[0][1]
    for a, b in spans[1:]:
        if a <= end: end = max(end, b)
        else: total += end - start; start, end = a, b
    return min(1., (total + end - start) / duration)


def characterize_videos(stats):
    stats = stats.copy()
    if stats.empty: return stats
    speech, visual = stats.subtitle_coverage.median(), stats.average_visual_change.median()
    speech_labels = np.where(stats.subtitle_coverage >= speech, "high speech", "low speech")
    visual_labels = np.where(stats.average_visual_change >= visual, "high visual change", "low visual change")
    stats["group"] = [f"{a} / {b}" for a, b in zip(speech_labels, visual_labels)]
    return stats


def summarize_dataset(events, stats, output_dir, tolerances=(5., 10., 20.)):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "all_events.csv", index=False); stats = characterize_videos(stats)
    stats.to_csv(output / "video_statistics.csv", index=False)
    visual = events[events.modality == "visual"] if not events.empty else events
    text = events[events.modality == "text"] if not events.empty else events
    alignment = pd.DataFrame([{"tolerance_seconds": tol,
        "visual_alignment_fraction": (visual.nearest_other_peak_seconds <= tol).mean() if len(visual) else np.nan,
        "text_alignment_fraction": (text.nearest_other_peak_seconds <= tol).mean() if len(text) else np.nan} for tol in tolerances])
    alignment.to_csv(output / "aggregate_alignment.csv", index=False)
    ax = alignment.set_index("tolerance_seconds")[["visual_alignment_fraction", "text_alignment_fraction"]].plot(
        kind="bar", figsize=(7, 4), ylim=(0, 1), rot=0)
    ax.set(xlabel="Tolerance (seconds)", ylabel="Aligned fraction", title="Cross-modal peak alignment")
    ax.figure.tight_layout(); ax.figure.savefig(output / "aggregate_alignment.png", dpi=160); plt.close(ax.figure)
    if not events.empty:
        finite = events.loc[np.isfinite(events.nearest_other_peak_seconds), "nearest_other_peak_seconds"]
        fig, ax = plt.subplots(figsize=(8, 4)); ax.hist(finite, bins=30)
        ax.set(xlabel="Nearest cross-modal peak distance (seconds)", ylabel="Events"); fig.tight_layout()
        fig.savefig(output / "peak_distance_histogram.png", dpi=160); plt.close(fig)
        count_rows = []
        for video_id, part in events.groupby("video_id"):
            count_rows.append({"video_id": video_id,
                "visual-only": int(((part.modality == "visual") & (part.event_type == "visual-only")).sum()),
                "text-only": int(((part.modality == "text") & (part.event_type == "text-only")).sum()),
                "shared": int(((part.modality == "visual") & (part.event_type == "aligned")).sum())})
        pd.DataFrame(count_rows).set_index("video_id").plot(kind="bar", stacked=True, figsize=(10, 5))
        plt.ylabel("Detected events"); plt.tight_layout(); plt.savefig(output / "boundary_counts.png", dpi=160); plt.close()
        matched = visual[(visual.event_type == "aligned") & visual.text_score.notna()]
        fig, ax = plt.subplots(figsize=(5, 5)); ax.scatter(matched.visual_score, matched.text_score, alpha=.65)
        ax.set(xlabel="Visual peak strength", ylabel="Nearest text peak strength"); fig.tight_layout()
        fig.savefig(output / "matched_peak_strengths.png", dpi=160); plt.close(fig)
    groups = pd.DataFrame()
    if not stats.empty and not visual.empty:
        merged = visual.merge(stats[["video_id", "group"]], on="video_id")
        groups = merged.groupby("group").agg(videos=("video_id", "nunique"),
            mean_nearest_distance=("nearest_other_peak_seconds", lambda x: np.mean(x[np.isfinite(x)]) if np.isfinite(x).any() else np.nan),
            aligned_at_10s=("nearest_other_peak_seconds", lambda x: np.mean(x <= 10))).reset_index()
        groups.to_csv(output / "group_alignment.csv", index=False)
    return {"alignment": alignment, "groups": groups, "stats": stats}


def _make_signal(emb, times, config):
    times, raw = compute_change_scores(emb, times)
    values = smooth_signal(raw, times, config.smoothing, config.smoothing_seconds)
    norm = normalize_signal(values, config.normalization, config.robust_percentiles)
    return Signal(times, raw, norm, detect_peaks(norm, times, config.peak_prominence, config.peak_min_spacing))


def run_pipeline(config, video_list=None, metadata_csv=None):
    config.validate(); output = Path(config.output_dir)
    for name in ("cache", "figures", "reports", "peak_frames", "aggregate"): (output / name).mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    records = discover_videos(config.dataset_dir, video_list, metadata_csv, config.max_videos)
    if not records: raise RuntimeError("No video/subtitle pairs found")
    print(f"Found {len(records)} video/subtitle pairs"); device = get_device()
    processor, visual_model = load_visual_encoder(config.visual_model, device)
    text_model = load_text_encoder(config.text_model, device)
    events_all, stats, alignments = [], [], []
    for n, record in enumerate(records, 1):
        print(f"[{n}/{len(records)}] {record.video_id}")
        try:
            _, duration, _ = _video_metadata(record.video_path)
            vt, ve = encode_video_frames(record.video_path, config.frame_interval, config.frame_batch_size,
                processor, visual_model, device, output / "cache", config.visual_model)
            tt, te, _, cues = encode_subtitle_windows(record.subtitle_path, config.subtitle_window, duration,
                text_model, config.text_batch_size, output / "cache", config.text_model)
            visual, text = _make_signal(ve, vt, config), _make_signal(te, tt, config)
            plot_video_signals(record.video_id, visual, text, output / "figures" / f"{record.video_id}.png")
            a, _ = compute_peak_alignment(visual.peak_times, text.peak_times, config.alignment_tolerances)
            a.insert(0, "video_id", record.video_id); alignments.append(a)
            events = build_event_table(record.video_id, visual, text, cues, 10., config.qualitative_context)
            events.to_csv(output / "reports" / f"{record.video_id}_events.csv", index=False); events_all.append(events)
            if config.save_peak_frames:
                for t in visual.peak_times: extract_frame_at(record.video_path, t, output / "peak_frames" / record.video_id / f"{t:.1f}.jpg")
            stats.append({"video_id": record.video_id, "duration_seconds": duration,
                "subtitle_coverage": subtitle_coverage(cues, duration), "average_visual_change": np.mean(visual.raw),
                "visual_peaks": len(visual.peak_indices), "text_peaks": len(text.peak_indices)})
        except Exception as exc: LOG.exception("Skipping %s: %s", record.video_id, exc)
    events = pd.concat(events_all, ignore_index=True) if events_all else pd.DataFrame(columns=EVENT_COLUMNS)
    stats = pd.DataFrame(stats); per_video = pd.concat(alignments, ignore_index=True) if alignments else pd.DataFrame()
    per_video.to_csv(output / "per_video_alignment.csv", index=False)
    result = summarize_dataset(events, stats, output / "aggregate", config.alignment_tolerances)
    result.update(events=events, per_video_alignment=per_video); return result

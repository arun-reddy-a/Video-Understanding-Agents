"""Select a reproducible, subcategory-balanced sample of long Video-MME videos."""
from __future__ import annotations

import argparse
import collections
import csv
import random
from pathlib import Path
from typing import Hashable, Iterable

HF_REPO = "lmms-eval/Video-MME"
HF_REVISION = "ead1408"
HF_METADATA_FILE = "videomme/test-00000-of-00001.parquet"
VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")


def allocate(sizes: dict[Hashable, int], n: int, minimum: int = 1) -> dict[Hashable, int]:
    """Allocate proportionally after giving every group a required minimum."""
    if any(size < minimum for size in sizes.values()):
        raise ValueError("A group has fewer items than the requested minimum")
    if n < minimum * len(sizes):
        raise ValueError(f"n={n} cannot give {minimum} items to all {len(sizes)} groups")
    if n > sum(sizes.values()):
        raise ValueError(f"n={n} exceeds the {sum(sizes.values())} available unique videos")
    quota = {key: minimum for key in sizes}
    left = n - minimum * len(sizes)
    capacities = {key: size - minimum for key, size in sizes.items()}
    while left:
        eligible = {key: cap for key, cap in capacities.items() if quota[key] < sizes[key]}
        total = sum(eligible.values())
        if not eligible or total <= 0:
            break
        shares = {key: left * cap / total for key, cap in eligible.items()}
        awarded = 0
        for key, share in shares.items():
            seats = min(int(share), sizes[key] - quota[key])
            quota[key] += seats
            awarded += seats
        left -= awarded
        if left:
            order = sorted(eligible, key=lambda key: (-(shares[key] % 1), -eligible[key], key))
            for key in order:
                if left == 0:
                    break
                if quota[key] < sizes[key]:
                    quota[key] += 1
                    left -= 1
    return quota


def sample_long_videos(rows: Iterable[dict], n: int = 24, seed: int = 0,
                       min_per_domain: int = 3) -> list[dict]:
    """Sample long videos with domain minimums and broad subcategory coverage."""
    unique: dict[str, dict] = {}
    for row in rows:
        if str(row.get("duration", "")).lower() != "long":
            continue
        asset_id = str(row["videoID"])
        unique.setdefault(asset_id, {
            "video_id": str(row["video_id"]).zfill(3),
            "videoID": asset_id,
            "duration": "long",
            "domain": str(row["domain"]),
            "sub_category": str(row["sub_category"]),
            "url": str(row.get("url", "")),
        })
    strata: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for record in unique.values():
        strata[(record["domain"], record["sub_category"])].append(record)
    if not strata:
        raise ValueError("No long videos were found in the metadata")
    by_domain: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for (domain, subcategory), values in strata.items():
        by_domain[domain][subcategory].extend(values)
    domain_sizes = {domain: sum(map(len, groups.values())) for domain, groups in by_domain.items()}
    domain_quotas = allocate(domain_sizes, n, minimum=min_per_domain)
    rng, selected = random.Random(seed), []
    for domain in sorted(by_domain):
        groups, quota = by_domain[domain], domain_quotas[domain]
        order = sorted(groups); rng.shuffle(order)
        subquotas = {subcategory: 0 for subcategory in groups}
        for subcategory in order[:min(quota, len(order))]:
            subquotas[subcategory] = 1
        remaining = quota - sum(subquotas.values())
        if remaining:
            capacities = {subcategory: len(values) - subquotas[subcategory]
                          for subcategory, values in groups.items()}
            extras = allocate({key: cap for key, cap in capacities.items() if cap > 0},
                              remaining, minimum=0)
            for subcategory, extra in extras.items(): subquotas[subcategory] += extra
        for subcategory in sorted(groups):
            candidates = sorted(groups[subcategory], key=lambda row: (row["video_id"], row["videoID"]))
            selected.extend(rng.sample(candidates, subquotas[subcategory]))
    return sorted(selected, key=lambda row: row["video_id"])


def official_metadata_path(cache_dir: str | Path = ".cache/video_mme") -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=HF_REPO, repo_type="dataset", revision=HF_REVISION,
                                filename=HF_METADATA_FILE, local_dir=cache_dir))


def write_selection(records: list[dict], output_csv: str | Path) -> Path:
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["video_id", "videoID", "duration", "domain", "sub_category", "url"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    output.with_suffix(".txt").write_text("\n".join(row["videoID"] for row in records) + "\n", encoding="utf-8")
    return output


def build_pipeline_manifest(selection_csv: str | Path, dataset_dir: str | Path,
                            output_csv: str | Path, require_all: bool = True) -> Path:
    """Resolve selected asset IDs to same-stem local video/SRT pairs."""
    root, rows, missing = Path(dataset_dir), [], []
    with Path(selection_csv).open(newline="", encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            asset_id = record["videoID"]
            video = next((root / f"{asset_id}{suffix}" for suffix in VIDEO_SUFFIXES
                          if (root / f"{asset_id}{suffix}").is_file()), None)
            subtitle = next((root / f"{asset_id}{suffix}" for suffix in (".srt", ".vtt")
                             if (root / f"{asset_id}{suffix}").is_file()), None)
            if video is None or subtitle is None:
                missing.append(asset_id); continue
            rows.append({"video_id": record["video_id"], "video_path": str(video),
                         "subtitle_path": str(subtitle), "asset_id": asset_id,
                         "domain": record["domain"], "sub_category": record["sub_category"]})
    if require_all and missing:
        preview = ", ".join(missing[:8])
        raise FileNotFoundError(f"Missing {len(missing)} selected video/subtitle pairs in {root}: {preview}")
    output = Path(output_csv); output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["video_id", "video_path", "subtitle_path", "asset_id", "domain", "sub_category"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return output


def main() -> None:
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", help="Parquet metadata; official metadata is downloaded when omitted")
    parser.add_argument("-n", type=int, default=24)
    parser.add_argument("--min-per-domain", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", default="long_stratified_24.csv")
    args = parser.parse_args()
    metadata = Path(args.metadata) if args.metadata else official_metadata_path()
    records = sample_long_videos(pq.read_table(metadata).to_pylist(), args.n, args.seed,
                                 args.min_per_domain)
    output = write_selection(records, args.o)
    for row in records:
        print(f"{row['video_id']}  {row['videoID']:14}  {row['domain']} / {row['sub_category']}")
    print(f"\nWrote {output} ({len(records)} long videos, seed={args.seed})")


if __name__ == "__main__":
    main()

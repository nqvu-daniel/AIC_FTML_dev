#!/usr/bin/env python3
"""
Dataset downloader & organizer for AIC 2025.

Reads a CSV of filenames + URLs, downloads each archive, extracts,
and sorts the contents into the structure expected by this repo:

dataset_root/
  videos/                   # *.mp4
  keyframes/<VID>/*.png     # competition keyframes
  meta/
    <VID>.map_keyframe.csv
    <VID>.media_info.json
    objects/<VID>/*.json
  features/<VID>.npy        # optional precomputed features

Usage:
  python scripts/dataset_downloader.py \
    --dataset_root /data/aic2025 \
    --csv AIC_2025_dataset_download_link.csv \
    --only all

Options for --only: all, videos, keyframes, features, meta, objects
"""

import argparse
import csv
import os
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple


def read_links(csv_path: Path) -> List[Tuple[str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # detect column indices (some files may have an empty first column)
        name_idx = None
        url_idx = None
        if header:
            for i, col in enumerate(header):
                c = (col or "").strip().lower()
                if c.startswith("filename"):
                    name_idx = i
                if c.startswith("download") or c.endswith("link"):
                    url_idx = i
        for row in reader:
            if not row:
                continue
            if name_idx is None or url_idx is None:
                # heuristic: last two columns
                name = row[-2].strip()
                url = row[-1].strip()
            else:
                name = row[name_idx].strip()
                url = row[url_idx].strip()
            if not name or not url:
                continue
            rows.append((name, url))
    return rows


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download(url: str, outfile: Path) -> None:
    from urllib.request import urlopen, Request

    ensure_dir(outfile.parent)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"[INFO] Downloading: {url}")
    with urlopen(req) as r:  # nosec - URL supplied by user
        total = int(r.headers.get("Content-Length", 0) or 0)
        read = 0
        chunk = 1 << 20
        tmp = outfile.with_suffix(outfile.suffix + ".part")
        with open(tmp, "wb") as f:
            while True:
                b = r.read(chunk)
                if not b:
                    break
                f.write(b)
                read += len(b)
                if total:
                    pct = 100.0 * read / total
                    print(f"\r  → {read/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)", end="")
        if total:
            print()
        tmp.replace(outfile)
    print(f"[OK] Saved → {outfile}")


def is_video_file(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".mkv", ".mov"}


def looks_like_vid_folder(name: str) -> bool:
    # e.g., L21_V001 or L26_V012
    return bool(re.match(r"^L\d{2}[_-]V\d{3}$", name, re.IGNORECASE))


def sort_extracted_to_layout(extracted_root: Path, dataset_root: Path) -> None:
    vids_dir = dataset_root / "videos"
    kf_dir = dataset_root / "keyframes"
    meta_dir = dataset_root / "meta"
    obj_dir = meta_dir / "objects"
    feat_dir = dataset_root / "features"
    for d in (vids_dir, kf_dir, meta_dir, obj_dir, feat_dir):
        ensure_dir(d)

    for root, _, files in os.walk(extracted_root):
        root_path = Path(root)
        for fn in files:
            src = root_path / fn
            name = src.name
            lower = name.lower()

            # Features
            if src.suffix.lower() == ".npy":
                dst = feat_dir / name
                ensure_dir(dst.parent)
                _move_if_needed(src, dst)
                continue

            # Meta: map_keyframe and media_info
            if lower.endswith(".map_keyframe.csv"):
                dst = meta_dir / name
                _move_if_needed(src, dst)
                continue
            if lower.endswith(".media_info.json"):
                dst = meta_dir / name
                _move_if_needed(src, dst)
                continue

            # Objects JSON: detect by folder name 'objects' or filename pattern 001.json etc
            if src.suffix.lower() == ".json" and not lower.endswith(".media_info.json"):
                # try to infer video id from parent folders
                parents = list(src.parents)
                vid = None
                for p in parents:
                    if looks_like_vid_folder(p.name):
                        vid = p.name
                        break
                if vid is None:
                    # fallback: strip prefix 'objects-'
                    m = re.search(r"(L\d{2}[_-]V\d{3})", name, re.IGNORECASE)
                    if m:
                        vid = m.group(1).replace("-", "_")
                if vid:
                    dst = obj_dir / vid / name
                    ensure_dir(dst.parent)
                    _move_if_needed(src, dst)
                    continue

            # Videos
            if is_video_file(src):
                dst = vids_dir / name
                _move_if_needed(src, dst)
                continue

            # Keyframes PNG
            if src.suffix.lower() == ".png":
                # find VID folder in parents; else keep in a flat folder
                vid = None
                for p in src.parents:
                    if looks_like_vid_folder(p.name):
                        vid = p.name
                        break
                if vid is None:
                    # heuristic: paths like .../Keyframes_L21/L21_V001/001.png
                    # above loop should have found; if not, leave under keyframes/_misc
                    dst = kf_dir / "_misc" / name
                else:
                    dst = kf_dir / vid / name
                ensure_dir(dst.parent)
                _move_if_needed(src, dst)
                continue

            # Other CSVs might belong to meta (keep them for debugging)
            if src.suffix.lower() == ".csv" and not lower.endswith(".map_keyframe.csv"):
                dst = meta_dir / name
                _move_if_needed(src, dst)
                continue

    # Cleanup empty dirs in extracted_root
    _remove_empty_dirs(extracted_root)


def _remove_empty_dirs(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        p = Path(dirpath)
        if not dirnames and not filenames:
            try:
                p.rmdir()
            except OSError:
                pass


def _move_if_needed(src: Path, dst: Path) -> None:
    if dst.exists():
        # Skip if same size (best-effort)
        try:
            if src.stat().st_size == dst.stat().st_size:
                return
        except Exception:
            pass
    shutil.move(str(src), str(dst))


def extract_archive(archive: Path, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    dest = out_dir / archive.stem
    ensure_dir(dest)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")
    return dest


def filter_rows(rows: List[Tuple[str, str]], only: str) -> List[Tuple[str, str]]:
    if only == "all":
        return rows
    def ok(name: str) -> bool:
        s = name.lower()
        if only == "videos":
            return "video" in s
        if only == "keyframes":
            return "keyframe" in s
        if only == "features":
            return "feature" in s or s.endswith(".npy")
        if only == "meta":
            return "media-info" in s or "map-keyframe" in s
        if only == "objects":
            return "object" in s
        return True
    return [(n, u) for n, u in rows if ok(n)]


def main():
    ap = argparse.ArgumentParser(description="Download and arrange AIC 2025 dataset")
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=Path("AIC_2025_dataset_download_link.csv"))
    ap.add_argument("--only", choices=["all", "videos", "keyframes", "features", "meta", "objects"], default="all")
    ap.add_argument("--skip-existing", action="store_true", help="Skip downloads if file already exists in downloads/")
    ap.add_argument("--keep-downloads", action="store_true", help="Keep archives after extraction (default removes them)")
    args = ap.parse_args()

    root = args.dataset_root
    ensure_dir(root)
    downloads = root / "downloads"
    extracted_tmp = root / "_extracted_tmp"
    ensure_dir(downloads)
    ensure_dir(extracted_tmp)

    rows = read_links(args.csv)
    rows = filter_rows(rows, args.only)
    if not rows:
        print(f"[WARN] No matching files found for --only={args.only}")
        return 0

    for name, url in rows:
        out = downloads / name
        if out.exists() and args.skip_existing:
            print(f"[SKIP] {name} (exists)")
        else:
            download(url, out)

        try:
            extracted = extract_archive(out, extracted_tmp)
        except Exception as e:
            print(f"[WARN] Failed to extract {name}: {e}")
            continue
        sort_extracted_to_layout(extracted, root)
        if not args.keep_downloads:
            try:
                out.unlink()
            except Exception:
                pass

    print("[OK] Dataset prepared at:", root)
    print("Layout:")
    print(" - videos/ (mp4)")
    print(" - keyframes/<VID>/*.png")
    print(" - meta/*.map_keyframe.csv, *.media_info.json, objects/<VID>/*.json")
    print(" - features/*.npy (optional)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


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


def filter_csv_for_videos(csv_path: Path, video_list: List[str], output_path: Path) -> bool:
    """Filter the CSV file to only include entries for specified videos + essential metadata"""
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return False

    filtered_rows = []
    total_rows = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            filtered_rows.append(header)

        for row in reader:
            total_rows += 1
            if not row or len(row) < 2:
                continue
            
            # Get filename from row (usually last or second-to-last column)
            filename = row[-2].strip() if len(row) >= 2 else ""
            filename_upper = filename.upper()

            # Always include essential metadata files (needed for all videos)
            essential_files = [
                'MAP-KEYFRAMES-AIC25-B1.ZIP',
                'MEDIA-INFO-AIC25-B1.ZIP', 
                'OBJECTS-AIC25-B1.ZIP',
                'CLIP-FEATURES-32-AIC25-B1.ZIP',
                'CLIP-FEATURES-AIC25-B1.ZIP'
            ]

            is_essential = any(essential in filename_upper for essential in essential_files)
            is_target_video = any(vid.upper() in filename_upper for vid in video_list)

            if is_essential or is_target_video:
                filtered_rows.append(row)

    # Write filtered CSV
    ensure_dir(output_path.parent)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(filtered_rows)

    print(f"📊 Filtered CSV: {len(filtered_rows)-1}/{total_rows} entries for videos {video_list} + essential metadata")
    return True


def read_links(csv_path: Path, video_filter: List[str] = None) -> List[Tuple[str, str]]:
    """Read CSV with optional video filtering for academic-grade processing"""
    
    # If video filter provided, create filtered CSV first
    if video_filter:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_csv:
            filtered_csv_path = Path(tmp_csv.name)
        
        if not filter_csv_for_videos(csv_path, video_filter, filtered_csv_path):
            return []
        
        # Use filtered CSV for reading
        csv_path = filtered_csv_path
    
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
    
    # Clean up temporary file if created
    if video_filter:
        try:
            os.unlink(filtered_csv_path)
        except:
            pass
    
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
        last_print_mb = 0
        tmp = outfile.with_suffix(outfile.suffix + ".part")
        with open(tmp, "wb") as f:
            while True:
                b = r.read(chunk)
                if not b:
                    break
                f.write(b)
                read += len(b)
                if total:
                    current_mb = read / 1e6
                    # Print progress every 10MB
                    if current_mb - last_print_mb >= 10 or read == total:
                        pct = 100.0 * read / total
                        print(f"\r  → {read/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)", end="")
                        last_print_mb = current_mb
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
    """
    Sort extracted files into AIC 2025 dataset structure using bulk directory operations.
    Much faster than checking individual files.
    """
    # Create target directories
    ensure_dir(dataset_root / "videos")
    ensure_dir(dataset_root / "keyframes") 
    ensure_dir(dataset_root / "features")
    ensure_dir(dataset_root / "map_keyframes")
    ensure_dir(dataset_root / "media_info")
    ensure_dir(dataset_root / "objects")
    
    # Find and bulk copy directory structures
    for item in extracted_root.rglob("*"):
        if not item.is_dir():
            continue
            
        item_name = item.name.lower()
        
        # Videos: Copy all Videos_* directories (they contain a 'video' subfolder)
        if item_name.startswith("videos_"):
            # Look for 'video' subfolder inside Videos_* directory
            video_subdir = item / "video"
            if video_subdir.exists():
                _copy_all_files(video_subdir, dataset_root / "videos", "*.mp4")
            else:
                # Fallback: copy directly from Videos_* if no video subfolder
                _copy_all_files(item, dataset_root / "videos", "*.mp4")
            
        # Keyframes: Copy from Keyframes_*/keyframes/ structure  
        elif item_name == "keyframes" and any("keyframes_" in p.name.lower() for p in item.parents):
            _copy_keyframe_structure(item, dataset_root / "keyframes")
            
        # Features: Copy from clip-features-* directories
        elif "clip-features" in item_name:
            _copy_all_files(item, dataset_root / "features", "*.npy")
            
        # Map keyframes: Copy from map-keyframes directories
        elif "map-keyframes" in item_name:
            _copy_all_files(item, dataset_root / "map_keyframes", "*.csv")
            
        # Media info: Copy from media-info directories  
        elif "media-info" in item_name:
            _copy_all_files(item, dataset_root / "media_info", "*.json")
            
        # Objects: Copy maintaining video subfolder structure
        elif item_name == "objects":
            _copy_objects_structure(item, dataset_root / "objects")
    
    # Cleanup
    _remove_empty_dirs(extracted_root)


def _copy_all_files(src_dir: Path, dst_dir: Path, pattern: str = "*") -> None:
    """Copy all files matching pattern from src to dst directory (flattened)"""
    ensure_dir(dst_dir)
    for file_path in src_dir.rglob(pattern):
        if file_path.is_file():
            dst = dst_dir / file_path.name
            _move_if_needed(file_path, dst)


def _copy_keyframe_structure(keyframes_dir: Path, dst_dir: Path) -> None:
    """Copy keyframe directory structure maintaining video subfolders"""
    ensure_dir(dst_dir)
    for video_dir in keyframes_dir.iterdir():
        if video_dir.is_dir() and looks_like_vid_folder(video_dir.name):
            video_dst = dst_dir / video_dir.name
            ensure_dir(video_dst)
            for img_file in video_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    dst = video_dst / img_file.name
                    _move_if_needed(img_file, dst)


def _copy_objects_structure(objects_dir: Path, dst_dir: Path) -> None:
    """Copy objects directory structure maintaining video subfolders"""
    ensure_dir(dst_dir)
    for video_dir in objects_dir.iterdir():
        if video_dir.is_dir() and looks_like_vid_folder(video_dir.name):
            video_dst = dst_dir / video_dir.name
            ensure_dir(video_dst)
            for json_file in video_dir.iterdir():
                if json_file.is_file() and json_file.suffix.lower() == ".json":
                    dst = video_dst / json_file.name
                    _move_if_needed(json_file, dst)


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
    # Use full filename to avoid conflicts when multiple archives have same stem
    dest = out_dir / archive.name.replace('.', '_')
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


def filter_rows(rows: List[Tuple[str, str]], only: str, videos: List[str] | None = None) -> List[Tuple[str, str]]:
    """Filter CSV rows by type and optional L## collections (e.g., ['L21','L22']).

    Always include essential metadata archives (map-keyframes, media-info, objects, clip-features).
    """
    def type_ok(name: str) -> bool:
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

    if videos is None or len(videos) == 0:
        return [(n, u) for n, u in rows if type_ok(n)]

    vids_upper = {v.upper() for v in videos}
    essentials = ("MAP-KEYFRAMES", "MEDIA-INFO", "OBJECTS", "CLIP-FEATURES")

    def video_ok(name: str) -> bool:
        up = name.upper()
        # Always include essential bundles
        if any(e in up for e in essentials):
            return True
        # Otherwise require an L## tag match in the filename
        return any(v in up for v in vids_upper)

    return [(n, u) for n, u in rows if type_ok(n) and video_ok(n)]


def main():
    ap = argparse.ArgumentParser(description="Academic-grade AIC 2025 dataset downloader with CSV filtering")
    ap.add_argument("--dataset_root", type=Path, required=True, help="Root directory for dataset")
    ap.add_argument("--csv", type=Path, default=Path("AIC_2025_dataset_download_link.csv"), help="CSV file with download links")
    ap.add_argument("--only", choices=["all", "videos", "keyframes", "features", "meta", "objects"], default="all")
    ap.add_argument("--videos", nargs="*", default=None, help="Restrict to L-collections (e.g., L21 L22)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip downloads if file already exists")
    ap.add_argument("--keep-downloads", action="store_true", help="Keep archives after extraction")
    ap.add_argument("--test-mode", action="store_true", help="Enable test mode (L21 L22 only)")
    args = ap.parse_args()

    # Handle test mode
    if args.test_mode and not args.videos:
        args.videos = ['L21', 'L22']
        print("🧪 TEST MODE ENABLED: Only processing L21-L22")

    print("🏗️ Academic-Grade AIC Dataset Downloader")
    print("=" * 50)
    print(f"📁 Dataset root: {args.dataset_root}")
    print(f"📄 CSV file: {args.csv}")
    print(f"🎯 Video filter: {args.videos or 'All videos'}")
    print(f"📦 Content filter: {args.only}")

    if not args.csv.exists():
        print(f"❌ CSV file not found: {args.csv}")
        print("Make sure the AIC_2025_dataset_download_link.csv file exists in the current directory")
        return 1

    root = args.dataset_root
    ensure_dir(root)
    downloads = root / "downloads"
    extracted_tmp = root / "_extracted_tmp"
    ensure_dir(downloads)
    ensure_dir(extracted_tmp)

    # Use enhanced CSV reading with video filtering
    print(f"📊 Reading and filtering CSV...")
    rows = read_links(args.csv, args.videos)
    rows = filter_rows(rows, args.only, args.videos)
    
    if not rows:
        print(f"⚠️ No matching files found for --only={args.only} --videos={args.videos}")
        return 0

    print(f"📦 Found {len(rows)} files to download")

    # Download and extract with progress
    import time
    start_time = time.time()
    
    for i, (name, url) in enumerate(rows, 1):
        print(f"\n📥 [{i}/{len(rows)}] Processing: {name}")
        
        out = downloads / name
        if out.exists() and args.skip_existing:
            print(f"⏭️ Skipping (already exists)")
        else:
            try:
                download(url, out)
            except Exception as e:
                print(f"❌ Download failed: {e}")
                continue

        try:
            print(f"📦 Extracting...")
            extracted = extract_archive(out, extracted_tmp)
            sort_extracted_to_layout(extracted, root)
            print(f"✅ Extracted and organized")
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            continue
            
        if not args.keep_downloads:
            try:
                out.unlink()
            except Exception:
                pass

    elapsed = time.time() - start_time
    print(f"\n🎉 Dataset preparation completed in {elapsed:.1f} seconds")

    print("[OK] Dataset prepared at:", root)
    print("Layout:")
    print(" - videos/ (mp4)")
    print(" - keyframes/<VID>/*.{jpg,jpeg,png} (organized by video)")
    print(" - map_keyframes/*.csv (keyframe mapping files)")
    print(" - media_info/*.json (video metadata)")
    print(" - objects/<VID>/*.json (object detection per video)")
    print(" - features/*.npy (precomputed CLIP features)")
    print(" - misc/ (other unhandled files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

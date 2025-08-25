#!/usr/bin/env python3
"""
Automatic video discovery and processing for AIC dataset.
Discovers video IDs from dataset structure and returns them for processing.
"""

import os
import sys
import argparse
from pathlib import Path

def discover_video_ids(dataset_root):
    """Discover video IDs in dataset"""
    dataset_root = Path(dataset_root)
    video_ids = set()
    
    # Check videos directory
    videos_dir = dataset_root / "videos"
    if videos_dir.exists():
        for video_file in videos_dir.glob("*.mp4"):
            video_id = video_file.stem.split('_')[0]  # L21_V001 -> L21
            video_ids.add(video_id)
    
    # Check keyframes directory
    keyframes_dir = dataset_root / "keyframes"
    if keyframes_dir.exists():
        for subdir in keyframes_dir.iterdir():
            if subdir.is_dir():
                video_id = subdir.name.split('_')[0]  # L21_V001 -> L21
                video_ids.add(video_id)
    
    # Check for ZIP files
    for zip_file in dataset_root.glob("*L*.zip"):
        try:
            # Extract L## pattern from various naming conventions
            name = zip_file.stem
            if "L" in name:
                parts = name.split('_')
                for part in parts:
                    if part.startswith('L') and len(part) >= 3:
                        video_ids.add(part[:3])  # L21, L22, etc.
        except Exception:
            continue
    
    return sorted(list(video_ids))

def main():
    parser = argparse.ArgumentParser(description="Discover video IDs in AIC dataset")
    parser.add_argument("dataset_root", help="Root directory of AIC dataset")
    parser.add_argument("--output", help="Output file to save video IDs")
    
    args = parser.parse_args()
    
    if not Path(args.dataset_root).exists():
        print(f"Error: Dataset root does not exist: {args.dataset_root}")
        return 1
    
    video_ids = discover_video_ids(args.dataset_root)
    
    if video_ids:
        print(f"Discovered {len(video_ids)} video collections: {video_ids}")
        
        if args.output:
            with open(args.output, 'w') as f:
                for video_id in video_ids:
                    f.write(f"{video_id}\n")
            print(f"Video IDs saved to {args.output}")
        else:
            # Print to stdout for shell capture
            for video_id in video_ids:
                print(video_id)
    else:
        print("No video collections found in dataset")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

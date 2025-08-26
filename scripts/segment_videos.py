#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import pandas as pd
import sys

# Ensure repo root in path so 'src' is importable
sys.path.append(str(Path(__file__).parent.parent))
from src.segmentation.transnetv2 import segment_video


def expand_videos(dataset_root: Path, videos_arg: list[str]) -> list[str]:
    all_ids: list[str] = []
    for v in videos_arg:
        if re.match(r"^L\d{2}$", v):
            mk = dataset_root / "map_keyframes"
            if not mk.exists():
                raise FileNotFoundError(f"map_keyframes directory not found: {mk}")
            coll = sorted([p.stem for p in mk.glob(f"{v}_V*.csv")])
            all_ids.extend(coll)
        else:
            all_ids.append(v)
    # dedup preserving order
    seen = set()
    out = []
    for vid in all_ids:
        if vid not in seen:
            out.append(vid)
            seen.add(vid)
    if not out:
        raise SystemExit("No videos found to segment")
    return out


def main():
    ap = argparse.ArgumentParser(description="Segment videos using TransNetV2 if available, else OpenCV fallback. Writes artifacts/segments.parquet")
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--videos", nargs="+", required=True, help="Video collections (L21) or IDs (L21_V001)")
    ap.add_argument("--model_path", type=Path, default=None, help="Optional path to TransNetV2 TorchScript model (.pt)")
    ap.add_argument("--rep_count", type=int, default=3, help="Representative frames per segment to record")
    ap.add_argument("--stride", type=int, default=5, help="Frame stride for OpenCV histogram-based detector")
    ap.add_argument("--z", type=float, default=3.0, help="Z-score multiplier for cut detection threshold")
    ap.add_argument("--min_len_sec", type=float, default=1.0, help="Minimum segment duration in seconds")
    ap.add_argument("--artifact_dir", type=Path, default=Path("./artifacts"))
    args = ap.parse_args()

    vids = expand_videos(args.dataset_root, args.videos)
    rows = []
    for vid in vids:
        segs, fps, reps = segment_video(
            dataset_root=args.dataset_root,
            video_id=vid,
            model_path=args.model_path,
            rep_count=args.rep_count,
            stride=args.stride,
            z=args.z,
            min_len_sec=args.min_len_sec,
        )
        for sid, ((s, e), rf) in enumerate(zip(segs, reps)):
            rows.append({
                "video_id": vid,
                "seg_id": sid,
                "start_frame": int(s),
                "end_frame": int(e),
                "start_sec": float(s / fps),
                "end_sec": float(e / fps),
                "rep_frames": rf,
            })

    out = pd.DataFrame(rows)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.artifact_dir / "segments.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[OK] wrote {len(out)} segments → {out_path}")


if __name__ == "__main__":
    main()

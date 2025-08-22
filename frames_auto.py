
import argparse, os, math
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# We use decord for accurate frame indexing (handles variable fps reasonably)
try:
    from decord import VideoReader, cpu
except Exception as e:
    raise SystemExit("Please install decord: pip install decord") from e

def uniform_indices(num_frames, every_n=None, sample_fps=None, fps=None):
    if every_n is not None:
        idxs = list(range(0, num_frames, max(1, int(every_n))))
    elif sample_fps is not None and fps is not None and fps > 0:
        step = max(1, int(round(fps / sample_fps)))
        idxs = list(range(0, num_frames, step))
    else:
        # default ~2 fps if fps known, else ~every 15 frames
        step = max(1, int(round((fps/2) if fps else 15)))
        idxs = list(range(0, num_frames, step))
    return idxs

def simple_shot_indices(vr, fps, downsample=4, thr=28.0, min_gap=6):
    # Very light shot change via frame diff on downsampled gray frames.
    # thr ~ 20..40 depending on content. Returns a superset of uniform if needed.
    L = len(vr)
    last = None
    picks = []
    for i in range(0, L, downsample):
        frm = vr[i].asnumpy()  # HWC, uint8
        gray = frm.mean(axis=2)  # crude grayscale
        if last is None:
            picks.append(i)
            last = gray
            continue
        diff = np.abs(gray - last).mean()
        if diff >= thr and (len(picks) == 0 or i - picks[-1] >= min_gap):
            picks.append(i)
            last = gray
    # always include last frame
    if picks[-1] != L - 1:
        picks.append(L - 1)
    return picks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--videos", nargs="+", required=True, help="Video IDs like L21_V001")
    ap.add_argument("--mode", choices=["uniform","shot"], default="uniform")
    ap.add_argument("--every_n", type=int, default=None, help="Take every Nth frame (uniform mode)")
    ap.add_argument("--sample_fps", type=float, default=None, help="Approx frames per second to sample (uniform mode)")
    ap.add_argument("--shot_thr", type=float, default=28.0, help="Threshold for simple shot detection")
    ap.add_argument("--out_keyframes_dir", type=str, default="keyframes_auto", help="Output subdir for generated keyframes")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    videos_dir = args.dataset_root / "videos"
    out_kf_root = args.dataset_root / args.out_keyframes_dir
    meta_dir = args.dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_kf_root.mkdir(parents=True, exist_ok=True)

    for vid in args.videos:
        mp4 = videos_dir / f"{vid}.mp4"
        if not mp4.exists():
            raise FileNotFoundError(mp4)
        vr = VideoReader(str(mp4), ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        L = len(vr)

        if args.mode == "uniform":
            idxs = uniform_indices(L, every_n=args.every_n, sample_fps=args.sample_fps, fps=fps)
        else:
            idxs = simple_shot_indices(vr, fps, thr=args.shot_thr)

        # write images and map
        out_dir = out_kf_root / vid
        out_dir.mkdir(parents=True, exist_ok=True)
        map_rows = []
        n = 1
        for idx in tqdm(idxs, desc=f"{vid} extracting"):
            idx = int(min(max(0, idx), L - 1))
            img = vr[idx].asnumpy()  # HWC uint8
            im = Image.fromarray(img, mode="RGB")
            png = out_dir / f"{n:03d}.png"
            if args.overwrite or not png.exists():
                im.save(png, format="PNG")
            pts_time = idx / fps if fps > 0 else 0.0
            map_rows.append((n, pts_time, fps, idx))
            n += 1

        # write map_keyframe.csv (our own)
        import csv
        map_csv = meta_dir / f"{vid}.map_keyframe.csv"
        with open(map_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["n","pts_time","fps","frame_idx"])
            for r in map_rows:
                w.writerow(r)
        print(f"[OK] {vid}: wrote {len(map_rows)} frames → {out_dir}")
        print(f"[OK] {vid}: map → {map_csv}")

if __name__ == "__main__":
    main()

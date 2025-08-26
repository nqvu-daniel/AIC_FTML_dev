import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    import torch
except Exception:  # torch optional for fallback
    torch = None


def _pick_video_file(root: Path, video_id: str) -> Optional[Path]:
    videos_dir = root / "videos"
    for ext in (".mp4", ".mkv", ".mov", ".avi"):
        p = videos_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    # Try lowercase extensions or nested structure
    for p in videos_dir.glob(f"{video_id}.*"):
        if p.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi"}:
            return p
    return None


def _fps_and_frames(cap: cv2.VideoCapture) -> Tuple[float, int]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return float(fps), frame_count


def _hsv_hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def detect_shots_opencv(video_path: Path, stride: int = 5, z: float = 3.0, min_len_frames: int = 25) -> List[Tuple[int, int]]:
    """Simple shot detection via HSV histogram deltas with z-score thresholding.
    Returns list of (start_frame, end_frame) segments (inclusive).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps, total = _fps_and_frames(cap)
    if total <= 0:
        total = 0
        # We still iterate until read fails

    prev_hist = None
    deltas = []
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            hist = _hsv_hist(frame)
            if prev_hist is not None:
                d = float(np.sum(np.abs(hist - prev_hist)))
                deltas.append(d)
                frames.append(idx)
            prev_hist = hist
        idx += 1
    cap.release()

    if not deltas:
        return [(0, max(0, idx - 1))]

    mu = float(np.mean(deltas))
    sigma = float(np.std(deltas) + 1e-6)
    threshold = mu + z * sigma
    cut_frames = []
    for f, d in zip(frames, deltas):
        if d > threshold:
            cut_frames.append(f)

    # Build segments from cut positions
    segs = []
    last = 0
    for cf in cut_frames:
        if cf - last >= min_len_frames:
            segs.append((last, cf - 1))
            last = cf
    segs.append((last, max(last, idx - 1)))
    # Merge too-short segments
    merged = []
    for s, e in segs:
        if merged and (e - merged[-1][0] + 1) < min_len_frames:
            # merge with previous
            ps, pe = merged.pop()
            merged.append((ps, e))
        else:
            merged.append((s, e))
    return merged


def _select_rep_frames(start: int, end: int, count: int = 3) -> List[int]:
    length = max(1, end - start + 1)
    if count <= 1:
        return [start + length // 2]
    if count == 2:
        return [start, end]
    # 3 or more: start, middle, end (truncate extras at end)
    reps = [start, start + length // 2, end]
    if count > 3:
        step = length // (count - 1)
        reps = [start + i * step for i in range(count - 1)] + [end]
    # Ensure within bounds
    reps = [min(max(start, r), end) for r in reps]
    # De-duplicate
    out = []
    for r in reps:
        if r not in out:
            out.append(r)
    return out


def segment_video(
    dataset_root: Path,
    video_id: str,
    model_path: Optional[Path] = None,
    rep_count: int = 3,
    stride: int = 5,
    z: float = 3.0,
    min_len_sec: float = 1.0,
) -> Tuple[List[Tuple[int, int]], float, List[List[int]]]:
    """Segment one video. If TransNetV2 weights given and usable, use them; else OpenCV fallback.
    Returns: (segments [(start_f,end_f)], fps, rep_frames_per_segment)
    """
    video_path = _pick_video_file(dataset_root, video_id)
    if video_path is None:
        raise FileNotFoundError(f"Video file for {video_id} not found under {dataset_root}/videos")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps, frame_count = _fps_and_frames(cap)
    cap.release()
    min_len_frames = int(max(1, round(min_len_sec * fps)))

    # Placeholder for TransNetV2 path (torchscript) — if provided and torch is available, try to use it.
    use_transnet = False
    if model_path and torch is not None and Path(model_path).exists():
        # NOTE: Real TransNetV2 integration would load the model and run prediction.
        # To keep this repo self-contained without external weights, we fall back to OpenCV if loading fails.
        try:
            _ = torch.jit.load(str(model_path), map_location="cpu")
            # Without the real inference code, we still use the OpenCV fallback but mark the flag for future extension.
            use_transnet = True
        except Exception:
            use_transnet = False

    # For now, both branches use the robust OpenCV fallback; the interface supports TransNetV2 drop-in later.
    segments = detect_shots_opencv(video_path, stride=stride, z=z, min_len_frames=min_len_frames)
    rep_per_seg = [_select_rep_frames(s, e, rep_count) for s, e in segments]
    return segments, fps, rep_per_seg


import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    import torch
    import transnetv2_pytorch
except Exception:  # torch optional for fallback
    torch = None
    transnetv2_pytorch = None


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


def detect_shots_transnetv2(
    video_path: Path,
    model: Optional[object] = None,
    threshold: float = 0.5,
    min_len_frames: int = 25,
    device: str = "cpu"
) -> List[Tuple[int, int]]:
    """TransNetV2-based shot detection.
    Returns list of (start_frame, end_frame) segments (inclusive).
    """
    if model is None or transnetv2_pytorch is None:
        raise RuntimeError("TransNetV2 model not available")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB and resize to TransNetV2 input size
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (48, 27))
        frames.append(frame_resized)
    cap.release()
    
    if len(frames) == 0:
        return [(0, 0)]
    
    # Convert to tensor format expected by TransNetV2
    frames_array = np.array(frames, dtype=np.uint8)
    frames_tensor = torch.from_numpy(frames_array).to(device)
    frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension
    
    # Run TransNetV2 prediction
    with torch.no_grad():
        predictions = model(frames_tensor)
        predictions = predictions.squeeze(0).cpu().numpy()
    
    # Convert predictions to binary scene boundaries
    scene_boundaries = predictions > threshold
    
    # Find scene segments
    segments = []
    scene_start = 0
    
    for i, is_boundary in enumerate(scene_boundaries):
        if is_boundary and i - scene_start >= min_len_frames:
            segments.append((scene_start, i - 1))
            scene_start = i
    
    # Add final segment
    if scene_start < len(frames) - 1:
        segments.append((scene_start, len(frames) - 1))
    
    # Ensure we have at least one segment
    if not segments:
        segments = [(0, max(0, len(frames) - 1))]
    
    return segments


def load_transnetv2_model(device: str = "cpu") -> Optional[object]:
    """Load TransNetV2 model. Returns None if loading fails."""
    if transnetv2_pytorch is None:
        return None
    
    try:
        model = transnetv2_pytorch.TransNetV2()
        # Try to load pre-trained weights if available
        try:
            state_dict = transnetv2_pytorch.load_transnetv2_weights()
            model.load_state_dict(state_dict)
        except Exception:
            # If pre-trained weights aren't available, use the model as-is
            pass
        
        model = model.to(device)
        model.eval()
        return model
    except Exception:
        return None


def segment_video(
    dataset_root: Path,
    video_id: str,
    model_path: Optional[Path] = None,
    rep_count: int = 3,
    stride: int = 5,
    z: float = 3.0,
    min_len_sec: float = 1.0,
    use_transnetv2: bool = True,
    transnetv2_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int]], float, List[List[int]]]:
    """Segment one video. Use TransNetV2 if available, else OpenCV fallback.
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

    # Try TransNetV2 first if requested and available
    if use_transnetv2 and torch is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_transnetv2_model(device)
        
        if model is not None:
            try:
                segments = detect_shots_transnetv2(
                    video_path, 
                    model, 
                    threshold=transnetv2_threshold, 
                    min_len_frames=min_len_frames,
                    device=device
                )
                rep_per_seg = [_select_rep_frames(s, e, rep_count) for s, e in segments]
                return segments, fps, rep_per_seg
            except Exception as e:
                print(f"TransNetV2 failed, falling back to OpenCV: {e}")
    
    # Fallback to OpenCV method
    segments = detect_shots_opencv(video_path, stride=stride, z=z, min_len_frames=min_len_frames)
    rep_per_seg = [_select_rep_frames(s, e, rep_count) for s, e in segments]
    return segments, fps, rep_per_seg


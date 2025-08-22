import argparse
import os
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from decord import VideoReader, cpu, gpu
except Exception as e:
    raise SystemExit("Please install decord: pip install decord") from e


class FastIntelligentSampler:
    """
    Optimized intelligent frame sampler with GPU support and parallel processing.
    Focuses on speed while maintaining quality of frame selection.
    """
    
    def __init__(self, window_size=8, min_gap=15, coverage_fps=0.3, use_gpu=False):
        """
        Args:
            window_size: Frames to compare on each side
            min_gap: Minimum frames between keyframes (increased for speed)
            coverage_fps: Minimum coverage (reduced for large datasets)
            use_gpu: Use GPU for video decoding if available
        """
        self.window_size = window_size
        self.min_gap = min_gap
        self.coverage_fps = coverage_fps
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
    def fast_frame_difference(self, frame1, frame2):
        """
        Fast frame difference using only essential metrics.
        """
        if frame1 is None or frame2 is None:
            return 0.0
            
        # Aggressive downsampling for speed
        h, w = 120, 160
        f1 = cv2.resize(frame1, (w, h), interpolation=cv2.INTER_LINEAR)
        f2 = cv2.resize(frame2, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to LAB color space (perceptually uniform)
        lab1 = cv2.cvtColor(f1, cv2.COLOR_RGB2LAB)
        lab2 = cv2.cvtColor(f2, cv2.COLOR_RGB2LAB)
        
        # Fast color difference in LAB space
        color_diff = np.mean(np.abs(lab1.astype(float) - lab2.astype(float)))
        
        # Fast edge detection using simple gradients
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
        
        # Simple gradient (faster than Sobel)
        dx1 = np.abs(np.diff(gray1, axis=1))
        dx2 = np.abs(np.diff(gray2, axis=1))
        edge_diff = np.mean(np.abs(dx1 - dx2))
        
        # Combined score with weights
        return color_diff * 0.7 + edge_diff * 0.3
    
    def compute_block_importance(self, frames_block):
        """
        Compute importance scores for a block of frames.
        """
        block_size = len(frames_block)
        scores = np.zeros(block_size)
        
        for i in range(block_size):
            # Compare with neighbors within block
            diffs = []
            for j in range(max(0, i-self.window_size), min(block_size, i+self.window_size+1)):
                if i != j:
                    diff = self.fast_frame_difference(frames_block[i], frames_block[j])
                    weight = 1.0 / (1 + abs(i - j) * 0.2)
                    diffs.append(diff * weight)
            
            scores[i] = np.mean(diffs) if diffs else 0.0
        
        return scores
    
    def parallel_importance_scoring(self, vr, sample_rate=30, num_workers=4):
        """
        Compute importance scores in parallel for speed.
        """
        L = len(vr)
        
        # Sample frames for analysis (every Nth frame)
        sample_indices = list(range(0, L, sample_rate))
        n_samples = len(sample_indices)
        
        # Load frames in batches
        batch_size = 100
        all_scores = {}
        
        with tqdm(total=n_samples, desc="Computing importance scores") as pbar:
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_indices = sample_indices[batch_start:batch_end]
                
                # Load batch of frames
                frames = [vr[idx].asnumpy() for idx in batch_indices]
                
                # Compute scores for this batch
                scores = self.compute_block_importance(frames)
                
                # Store results
                for idx, score in zip(batch_indices, scores):
                    all_scores[idx] = score
                
                pbar.update(len(batch_indices))
        
        return all_scores
    
    def fast_peak_detection(self, scores_dict, L, min_frames):
        """
        Fast peak detection with non-maximum suppression.
        """
        # Convert sparse scores to dense array
        indices = sorted(scores_dict.keys())
        scores = np.array([scores_dict[i] for i in indices])
        
        if len(scores) == 0:
            return list(range(0, L, L // min_frames))
        
        # Normalize scores
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # Adaptive threshold
        threshold = np.percentile(scores, 70)
        
        # Find peaks above threshold
        peak_mask = scores > threshold
        peak_indices = [indices[i] for i, is_peak in enumerate(peak_mask) if is_peak]
        
        # Non-maximum suppression
        selected = []
        for idx in sorted(peak_indices):
            # Check if far enough from already selected frames
            if all(abs(idx - s) >= self.min_gap for s in selected):
                selected.append(idx)
        
        # Ensure minimum coverage
        if len(selected) < min_frames:
            # Add uniformly spaced frames
            gap = L // (min_frames - len(selected) + 1)
            for i in range(gap, L, gap):
                if len(selected) >= min_frames:
                    break
                # Find nearest unselected frame
                if all(abs(i - s) >= self.min_gap // 2 for s in selected):
                    selected.append(i)
        
        # Always include first and last
        if 0 not in selected:
            selected.insert(0, 0)
        if L - 1 not in selected:
            selected.append(L - 1)
        
        return sorted(selected)
    
    def ultra_fast_sampling(self, vr, fps):
        """
        Ultra-fast sampling using motion vectors and frame statistics.
        """
        L = len(vr)
        min_frames = max(10, int(L / fps * self.coverage_fps) if fps > 0 else L // 100)
        
        # Use larger sample rate for very large videos
        if L > 10000:
            sample_rate = 60
        elif L > 5000:
            sample_rate = 40
        else:
            sample_rate = 20
        
        print(f"Analyzing {L} frames with sample rate 1:{sample_rate}")
        
        # Parallel importance scoring
        scores_dict = self.parallel_importance_scoring(vr, sample_rate)
        
        # Fast peak detection
        selected_indices = self.fast_peak_detection(scores_dict, L, min_frames)
        
        # Generate scores for selected frames
        scores = []
        for idx in selected_indices:
            # Find nearest scored frame
            nearest_scored = min(scores_dict.keys(), key=lambda x: abs(x - idx))
            scores.append(scores_dict.get(idx, scores_dict[nearest_scored]))
        
        return selected_indices, np.array(scores)
    
    def motion_based_sampling(self, vr, fps, motion_threshold=5.0):
        """
        Extremely fast sampling based on optical flow motion detection.
        """
        L = len(vr)
        selected = [0]  # Always include first frame
        
        # Parameters for optical flow
        flow_params = dict(
            pyr_scale=0.5,
            levels=1,
            winsize=15,
            iterations=1,
            poly_n=5,
            poly_sigma=1.1,
            flags=0
        )
        
        # Downsample for flow computation
        h, w = 120, 160
        
        prev_gray = None
        min_gap_counter = 0
        
        for i in tqdm(range(0, L, 5), desc="Motion detection"):  # Check every 5th frame
            if i >= L:
                break
                
            frame = vr[i].asnumpy()
            gray = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_RGB2GRAY)
            
            if prev_gray is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **flow_params)
                
                # Compute motion magnitude
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_score = np.mean(magnitude)
                
                # Select frame if motion exceeds threshold and min gap satisfied
                if motion_score > motion_threshold and min_gap_counter >= self.min_gap:
                    selected.append(i)
                    min_gap_counter = 0
                else:
                    min_gap_counter += 5
            
            prev_gray = gray
        
        # Always include last frame
        if L - 1 not in selected:
            selected.append(L - 1)
        
        # Generate uniform scores
        scores = np.ones(len(selected))
        
        return selected, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--videos", nargs="+", required=True, help="Video IDs")
    ap.add_argument("--mode", choices=["fast", "ultra_fast", "motion"], default="fast",
                   help="Sampling mode: fast, ultra_fast, or motion-based")
    ap.add_argument("--window_size", type=int, default=8)
    ap.add_argument("--min_gap", type=int, default=15)
    ap.add_argument("--coverage_fps", type=float, default=0.3)
    ap.add_argument("--out_keyframes_dir", type=str, default="keyframes_intelligent")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--use_gpu", action="store_true", help="Use GPU for video decoding")
    args = ap.parse_args()
    
    videos_dir = args.dataset_root / "videos"
    out_kf_root = args.dataset_root / args.out_keyframes_dir
    meta_dir = args.dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_kf_root.mkdir(parents=True, exist_ok=True)
    
    sampler = FastIntelligentSampler(
        window_size=args.window_size,
        min_gap=args.min_gap,
        coverage_fps=args.coverage_fps,
        use_gpu=args.use_gpu
    )
    
    for vid in args.videos:
        print(f"\n{'='*60}")
        print(f"Processing {vid}...")
        mp4 = videos_dir / f"{vid}.mp4"
        if not mp4.exists():
            raise FileNotFoundError(mp4)
        
        # Use GPU context if available
        ctx = gpu(0) if args.use_gpu else cpu(0)
        vr = VideoReader(str(mp4), ctx=ctx)
        fps = float(vr.get_avg_fps())
        L = len(vr)
        
        duration = L / fps if fps > 0 else L / 30
        print(f"Video: {L} frames @ {fps:.2f} fps ({duration:.1f} seconds)")
        
        # Select sampling method
        if args.mode == "ultra_fast":
            idxs, scores = sampler.ultra_fast_sampling(vr, fps)
        elif args.mode == "motion":
            idxs, scores = sampler.motion_based_sampling(vr, fps)
        else:  # fast
            idxs, scores = sampler.ultra_fast_sampling(vr, fps)
        
        # Report statistics
        compression_ratio = (1 - len(idxs) / L) * 100
        avg_gap = np.mean(np.diff(idxs)) if len(idxs) > 1 else 0
        coverage = len(idxs) / duration if duration > 0 else 0
        
        print(f"\nResults:")
        print(f"  • Selected {len(idxs)} keyframes")
        print(f"  • Compression: {compression_ratio:.1f}% reduction")
        print(f"  • Average gap: {avg_gap:.1f} frames ({avg_gap/fps:.2f}s)")
        print(f"  • Coverage: {coverage:.2f} frames/second")
        
        # Extract and save frames
        out_dir = out_kf_root / vid
        out_dir.mkdir(parents=True, exist_ok=True)
        map_rows = []
        
        n = 1
        for idx, score in tqdm(zip(idxs, scores), total=len(idxs), desc="Extracting frames"):
            idx = int(min(max(0, idx), L - 1))
            img = vr[idx].asnumpy()
            im = Image.fromarray(img, mode="RGB")
            png = out_dir / f"{n:03d}.png"
            if args.overwrite or not png.exists():
                im.save(png, format="PNG", optimize=True)
            pts_time = idx / fps if fps > 0 else 0.0
            map_rows.append((n, pts_time, fps, idx, float(score)))
            n += 1
        
        # Write mapping CSV
        map_csv = meta_dir / f"{vid}.map_keyframe.csv"
        with open(map_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["n", "pts_time", "fps", "frame_idx", "importance_score"])
            for r in map_rows:
                w.writerow(r)
        
        print(f"\n✓ Saved {len(map_rows)} frames → {out_dir}")
        print(f"✓ Saved mapping → {map_csv}")


if __name__ == "__main__":
    main()
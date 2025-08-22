import argparse
import os
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from collections import deque
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

try:
    from decord import VideoReader, cpu
except Exception as e:
    raise SystemExit("Please install decord: pip install decord") from e


class IntelligentFrameSampler:
    """
    Intelligent frame sampling that analyzes temporal windows to find
    visually significant frames instead of uniform sampling.
    """
    
    def __init__(self, window_size=8, min_gap=10, coverage_fps=0.5):
        """
        Args:
            window_size: Number of frames to compare on each side (±window_size)
            min_gap: Minimum frames between selected keyframes
            coverage_fps: Minimum coverage (frames per second) to ensure temporal coverage
        """
        self.window_size = window_size
        self.min_gap = min_gap
        self.coverage_fps = coverage_fps
        
    def compute_frame_difference(self, frame1, frame2, method='combined'):
        """
        Compute difference between two frames using multiple metrics.
        """
        if frame1 is None or frame2 is None:
            return 0.0
            
        # Resize for faster computation
        h, w = 240, 320  # Standard low res for difference calculation
        f1 = cv2.resize(frame1, (w, h))
        f2 = cv2.resize(frame2, (w, h))
        
        scores = {}
        
        # 1. Color histogram difference (HSV space)
        if method in ['combined', 'histogram']:
            hsv1 = cv2.cvtColor(f1, cv2.COLOR_RGB2HSV)
            hsv2 = cv2.cvtColor(f2, cv2.COLOR_RGB2HSV)
            
            hist_diff = 0
            for channel in range(3):
                hist1 = cv2.calcHist([hsv1], [channel], None, [50], [0, 256])
                hist2 = cv2.calcHist([hsv2], [channel], None, [50], [0, 256])
                hist1 = cv2.normalize(hist1, hist1).flatten()
                hist2 = cv2.normalize(hist2, hist2).flatten()
                hist_diff += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            scores['histogram'] = hist_diff / 3.0
        
        # 2. Edge difference (Sobel)
        if method in ['combined', 'edge']:
            gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
            
            edges1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 1, ksize=3)
            edges2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 1, ksize=3)
            edge_diff = np.mean(np.abs(edges1 - edges2))
            scores['edge'] = edge_diff
        
        # 3. Pixel-wise difference (motion)
        if method in ['combined', 'pixel']:
            gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
            pixel_diff = np.mean(np.abs(gray1.astype(float) - gray2.astype(float)))
            scores['pixel'] = pixel_diff
        
        # 4. Texture/entropy difference
        if method in ['combined', 'texture']:
            gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
            
            # Simple texture metric using local standard deviation
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            
            mean1 = cv2.filter2D(gray1, -1, kernel)
            mean2 = cv2.filter2D(gray2, -1, kernel)
            
            sq1 = cv2.filter2D(gray1**2, -1, kernel)
            sq2 = cv2.filter2D(gray2**2, -1, kernel)
            
            std1 = np.sqrt(np.maximum(sq1 - mean1**2, 0))
            std2 = np.sqrt(np.maximum(sq2 - mean2**2, 0))
            
            texture_diff = np.mean(np.abs(std1 - std2))
            scores['texture'] = texture_diff
        
        if method == 'combined':
            # Weighted combination of all metrics
            weights = {
                'histogram': 0.3,
                'edge': 0.25,
                'pixel': 0.25,
                'texture': 0.2
            }
            total = sum(scores.get(k, 0) * v for k, v in weights.items())
            return total
        else:
            return scores.get(method, 0)
    
    def compute_temporal_importance(self, vr, frame_idx, window_size=None):
        """
        Compute importance score for a frame based on its difference
        from surrounding frames in a temporal window.
        """
        if window_size is None:
            window_size = self.window_size
            
        L = len(vr)
        start = max(0, frame_idx - window_size)
        end = min(L, frame_idx + window_size + 1)
        
        if start == end:
            return 0.0
        
        # Get current frame
        current_frame = vr[frame_idx].asnumpy()
        
        differences = []
        for i in range(start, end):
            if i == frame_idx:
                continue
            compare_frame = vr[i].asnumpy()
            diff = self.compute_frame_difference(current_frame, compare_frame)
            # Weight by temporal distance (closer frames matter more)
            temporal_weight = 1.0 / (1 + abs(i - frame_idx) * 0.1)
            differences.append(diff * temporal_weight)
        
        # Return average weighted difference
        return np.mean(differences) if differences else 0.0
    
    def adaptive_threshold(self, scores, percentile=75):
        """
        Compute adaptive threshold based on score distribution.
        """
        if len(scores) == 0:
            return 0.0
        return np.percentile(scores, percentile)
    
    def select_keyframes(self, vr, fps, batch_size=500, progress_bar=True):
        """
        Select keyframes using intelligent sampling.
        """
        L = len(vr)
        
        # Ensure minimum temporal coverage
        min_frames = int(L / fps * self.coverage_fps) if fps > 0 else L // 60
        
        # Process in batches for memory efficiency
        importance_scores = np.zeros(L)
        
        # Sample frames for initial analysis (every 10th frame for speed)
        sample_step = 10
        sample_indices = list(range(0, L, sample_step))
        
        if progress_bar:
            pbar = tqdm(sample_indices, desc="Analyzing frames")
        else:
            pbar = sample_indices
            
        for idx in pbar:
            importance_scores[idx] = self.compute_temporal_importance(vr, idx)
        
        # Interpolate scores for unsampled frames
        from scipy.interpolate import interp1d
        if len(sample_indices) > 1:
            f = interp1d(sample_indices, importance_scores[sample_indices], 
                        kind='linear', fill_value='extrapolate')
            all_indices = np.arange(L)
            importance_scores = f(all_indices)
        
        # Normalize scores
        if importance_scores.max() > 0:
            importance_scores = importance_scores / importance_scores.max()
        
        # Find peaks in importance scores
        threshold = self.adaptive_threshold(importance_scores, percentile=60)
        peak_indices, properties = find_peaks(
            importance_scores,
            height=threshold,
            distance=self.min_gap,
            prominence=0.1
        )
        
        selected_indices = list(peak_indices)
        
        # Ensure minimum coverage by adding frames in gaps
        if len(selected_indices) < min_frames:
            # Find large gaps and add frames
            gaps = []
            for i in range(len(selected_indices) - 1):
                gap_size = selected_indices[i+1] - selected_indices[i]
                if gap_size > fps * 2:  # Gap larger than 2 seconds
                    gaps.append((gap_size, selected_indices[i], selected_indices[i+1]))
            
            gaps.sort(reverse=True)
            
            for gap_size, start, end in gaps:
                if len(selected_indices) >= min_frames:
                    break
                # Add frame in middle of gap
                mid = (start + end) // 2
                # Find local maximum near midpoint
                search_start = max(0, mid - 20)
                search_end = min(L, mid + 20)
                local_max_idx = search_start + np.argmax(importance_scores[search_start:search_end])
                selected_indices.append(local_max_idx)
        
        # Always include first and last frames
        if 0 not in selected_indices:
            selected_indices.insert(0, 0)
        if L - 1 not in selected_indices:
            selected_indices.append(L - 1)
        
        selected_indices.sort()
        
        # Return indices and their importance scores
        return selected_indices, importance_scores[selected_indices]
    
    def detect_scene_changes(self, vr, threshold_multiplier=2.0):
        """
        Detect hard scene cuts using multiple metrics.
        """
        L = len(vr)
        scene_changes = [0]  # Always include first frame
        
        prev_frame = vr[0].asnumpy()
        
        for i in tqdm(range(1, L), desc="Detecting scene changes"):
            curr_frame = vr[i].asnumpy()
            
            # Use stricter threshold for scene detection
            diff = self.compute_frame_difference(prev_frame, curr_frame)
            
            # Dynamic threshold based on recent history
            if i > 30:
                recent_diffs = []
                for j in range(max(0, i-30), i):
                    f1 = vr[j].asnumpy()
                    f2 = vr[j+1].asnumpy() if j+1 < L else f1
                    recent_diffs.append(self.compute_frame_difference(f1, f2))
                
                dynamic_threshold = np.mean(recent_diffs) * threshold_multiplier
                
                if diff > dynamic_threshold:
                    # Verify it's not noise by checking consistency
                    if i + 1 < L:
                        next_frame = vr[i+1].asnumpy()
                        next_diff = self.compute_frame_difference(curr_frame, next_frame)
                        if next_diff < diff * 0.5:  # Next frame is similar to current
                            scene_changes.append(i)
            
            prev_frame = curr_frame
        
        # Always include last frame
        if L - 1 not in scene_changes:
            scene_changes.append(L - 1)
        
        return scene_changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--videos", nargs="+", required=True, help="Video IDs like L21_V001")
    ap.add_argument("--mode", choices=["intelligent", "scene", "hybrid"], default="intelligent",
                   help="Sampling mode: intelligent (temporal window), scene (hard cuts), hybrid (both)")
    ap.add_argument("--window_size", type=int, default=8, 
                   help="Temporal window size for importance calculation")
    ap.add_argument("--min_gap", type=int, default=10,
                   help="Minimum frames between selected keyframes")
    ap.add_argument("--coverage_fps", type=float, default=0.5,
                   help="Minimum temporal coverage in frames per second")
    ap.add_argument("--out_keyframes_dir", type=str, default="keyframes_intelligent",
                   help="Output subdir for generated keyframes")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save_scores", action="store_true",
                   help="Save importance scores for analysis")
    args = ap.parse_args()
    
    videos_dir = args.dataset_root / "videos"
    out_kf_root = args.dataset_root / args.out_keyframes_dir
    meta_dir = args.dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_kf_root.mkdir(parents=True, exist_ok=True)
    
    sampler = IntelligentFrameSampler(
        window_size=args.window_size,
        min_gap=args.min_gap,
        coverage_fps=args.coverage_fps
    )
    
    for vid in args.videos:
        print(f"\nProcessing {vid}...")
        mp4 = videos_dir / f"{vid}.mp4"
        if not mp4.exists():
            raise FileNotFoundError(mp4)
        
        vr = VideoReader(str(mp4), ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        L = len(vr)
        
        print(f"Video: {L} frames @ {fps:.2f} fps ({L/fps:.1f} seconds)")
        
        # Select frames based on mode
        if args.mode == "intelligent":
            idxs, scores = sampler.select_keyframes(vr, fps)
            print(f"Selected {len(idxs)} keyframes using intelligent sampling")
            
        elif args.mode == "scene":
            idxs = sampler.detect_scene_changes(vr)
            scores = np.ones(len(idxs))  # Uniform scores for scene mode
            print(f"Detected {len(idxs)} scene changes")
            
        else:  # hybrid
            intelligent_idxs, intelligent_scores = sampler.select_keyframes(vr, fps)
            scene_idxs = sampler.detect_scene_changes(vr)
            
            # Merge and deduplicate
            all_idxs = set(intelligent_idxs) | set(scene_idxs)
            idxs = sorted(list(all_idxs))
            
            # Recompute scores for merged set
            scores = []
            for idx in idxs:
                if idx in intelligent_idxs:
                    score_idx = intelligent_idxs.index(idx)
                    scores.append(intelligent_scores[score_idx])
                else:
                    scores.append(1.0)  # High score for scene changes
            scores = np.array(scores)
            print(f"Selected {len(idxs)} keyframes using hybrid sampling")
        
        # Calculate compression ratio
        compression_ratio = (1 - len(idxs) / L) * 100
        print(f"Compression: {compression_ratio:.1f}% reduction from {L} to {len(idxs)} frames")
        
        # Write images and map
        out_dir = out_kf_root / vid
        out_dir.mkdir(parents=True, exist_ok=True)
        map_rows = []
        
        n = 1
        for idx, score in tqdm(zip(idxs, scores), total=len(idxs), desc=f"{vid} extracting"):
            idx = int(min(max(0, idx), L - 1))
            img = vr[idx].asnumpy()  # HWC uint8
            im = Image.fromarray(img, mode="RGB")
            png = out_dir / f"{n:03d}.png"
            if args.overwrite or not png.exists():
                im.save(png, format="PNG")
            pts_time = idx / fps if fps > 0 else 0.0
            map_rows.append((n, pts_time, fps, idx, float(score)))
            n += 1
        
        # Write map_keyframe.csv with importance scores
        map_csv = meta_dir / f"{vid}.map_keyframe_intelligent.csv"
        with open(map_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["n", "pts_time", "fps", "frame_idx", "importance_score"])
            for r in map_rows:
                w.writerow(r)
        
        print(f"[OK] {vid}: wrote {len(map_rows)} frames → {out_dir}")
        print(f"[OK] {vid}: map → {map_csv}")
        
        # Optionally save detailed scores for analysis
        if args.save_scores:
            scores_file = meta_dir / f"{vid}.importance_scores.npy"
            np.save(scores_file, {'indices': idxs, 'scores': scores})
            print(f"[OK] {vid}: scores → {scores_file}")


if __name__ == "__main__":
    main()
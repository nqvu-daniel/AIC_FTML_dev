#!/usr/bin/env python3
"""
Advanced intelligent frame sampling with multiple smart algorithms.
Uses semantic analysis, motion detection, and visual importance scoring.
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import utils
import config

class AdvancedFrameSampler:
    def __init__(self, dataset_root, use_gpu=False, batch_size=32):
        self.dataset_root = Path(dataset_root)
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
        # Initialize GPU if available
        if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.gpu_available = True
            print("GPU acceleration enabled for OpenCV")
        else:
            self.gpu_available = False
            print("Using CPU processing")
    
    def compute_visual_complexity(self, frame):
        """Compute visual complexity using multiple metrics"""
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        metrics = {}
        
        # 1. Edge density (structural complexity)
        edges = cv2.Canny(gray, 50, 150)
        metrics['edge_density'] = np.mean(edges) / 255.0
        
        # 2. Color diversity (histogram entropy)
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_h = hist_h.flatten() + 1e-10  # Avoid log(0)
        hist_h = hist_h / np.sum(hist_h)
        metrics['color_entropy'] = -np.sum(hist_h * np.log(hist_h))
        
        # 3. Texture complexity (local variance)
        kernel = np.ones((5,5), np.float32) / 25
        mean_filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        variance = cv2.filter2D((gray.astype(np.float32) - mean_filtered) ** 2, -1, kernel)
        metrics['texture_complexity'] = np.mean(variance) / 255.0
        
        # 4. Brightness distribution (contrast)
        metrics['brightness_std'] = np.std(gray) / 255.0
        
        # 5. Saturation variance (color intensity variation)
        metrics['saturation_std'] = np.std(hsv[:,:,1]) / 255.0
        
        return metrics
    
    def detect_scene_changes(self, frames, threshold=0.3):
        """Detect scene changes using multiple methods"""
        scene_changes = []
        
        if len(frames) < 2:
            return scene_changes
        
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            
            # Method 1: Histogram difference
            hist_prev = cv2.calcHist([prev_frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist_curr = cv2.calcHist([curr_frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            
            hist_diff = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
            
            # Method 2: Structural similarity (SSIM approximation)
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Simple SSIM using mean and variance
            mu1 = np.mean(gray_prev)
            mu2 = np.mean(gray_curr)
            var1 = np.var(gray_prev)
            var2 = np.var(gray_curr)
            covar = np.mean((gray_prev - mu1) * (gray_curr - mu2))
            
            ssim = (2 * mu1 * mu2 + 0.01) * (2 * covar + 0.03) / ((mu1**2 + mu2**2 + 0.01) * (var1 + var2 + 0.03))
            
            # Combine metrics
            scene_score = (1 - hist_diff) * 0.6 + (1 - ssim) * 0.4
            
            if scene_score > threshold:
                scene_changes.append(i)
        
        return scene_changes
    
    def compute_motion_intensity(self, frame1, frame2):
        """Compute motion intensity between frames using LK optical flow"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        prev_pts = cv2.goodFeaturesToTrack(gray1, maxCorners=200, qualityLevel=0.01, minDistance=8)
        if prev_pts is None or len(prev_pts) == 0:
            return 0.0

        next_pts, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, prev_pts, None)
        if next_pts is None or status is None:
            return 0.0
        mask = status.reshape(-1) == 1
        if not np.any(mask):
            return 0.0
        disp = next_pts[mask] - prev_pts[mask]
        motion_magnitude = np.linalg.norm(disp, axis=2)
        return float(np.mean(motion_magnitude)) if motion_magnitude.size else 0.0
    
    def score_frame_importance(self, frame, context_frames, frame_idx):
        """Score frame importance using multiple criteria"""
        scores = {}
        
        # 1. Visual complexity
        complexity_metrics = self.compute_visual_complexity(frame)
        scores['complexity'] = np.mean(list(complexity_metrics.values()))
        
        # 2. Uniqueness within context
        if context_frames:
            uniqueness_scores = []
            for ctx_frame in context_frames:
                # Simple pixel difference
                diff = np.mean(np.abs(frame.astype(np.float32) - ctx_frame.astype(np.float32)))
                uniqueness_scores.append(diff / 255.0)
            scores['uniqueness'] = np.mean(uniqueness_scores) if uniqueness_scores else 0.5
        else:
            scores['uniqueness'] = 0.5
        
        # 3. Motion context
        if len(context_frames) >= 2:
            # Check motion before and after this frame
            motion_scores = []
            for i, ctx_frame in enumerate(context_frames):
                if i < len(context_frames) - 1:
                    motion = self.compute_motion_intensity(ctx_frame, context_frames[i+1])
                    motion_scores.append(motion)
            scores['motion_context'] = np.mean(motion_scores) if motion_scores else 0.0
        else:
            scores['motion_context'] = 0.0
        
        # 4. Position-based importance (middle frames often more important)
        total_frames = len(context_frames) + 1
        center_distance = abs(frame_idx - total_frames // 2) / (total_frames // 2)
        scores['position'] = 1.0 - center_distance
        
        # Weighted combination
        weights = {
            'complexity': 0.3,
            'uniqueness': 0.4,
            'motion_context': 0.2,
            'position': 0.1
        }
        
        final_score = sum(scores[key] * weights[key] for key in scores)
        return final_score, scores
    
    def smart_sampling(self, video_path, target_fps=0.5, window_size=16, min_gap=8):
        """Perform smart frame sampling on video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"Video stats: {frame_count} frames, {fps:.1f} fps, {duration:.1f}s")
        
        # Safety checks to prevent hanging
        if frame_count <= 0:
            print("ERROR: Invalid video - no frames detected")
            cap.release()
            return []
        
        if frame_count > 100000:  # ~55 minutes at 30fps
            print("WARNING: Very long video detected, may take time to process")
        
        if duration > 3600:  # 1 hour
            print("ERROR: Video too long (>1 hour), skipping for safety")
            cap.release()
            return []
        
        # Read all frames (for smaller videos) or sample strategically
        frames = []
        frame_indices = []
        
        if frame_count < 1000:  # Small video - read all frames
            step = 1
        else:  # Large video - sample every N frames initially
            step = max(1, int(fps / 4))  # Sample 4 times per second initially
        
        frame_positions = list(range(0, frame_count, step))
        with tqdm(frame_positions, desc=f"Loading frames from {video_path.name}", unit="frame") as pbar:
            for i in pbar:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    frame_indices.append(i)
                pbar.set_postfix({"loaded": len(frames), "pos": f"{i}/{frame_count}"})
        
        cap.release()
        
        if not frames:
            return []
        
        print(f"Loaded {len(frames)} frames for analysis")
        
        # Score all frames
        importance_scores = []
        
        with tqdm(enumerate(frames), desc="Scoring frames", total=len(frames), unit="frame") as pbar:
            for i, frame in pbar:
                # Get context window
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(frames), i + window_size // 2 + 1)
                context_frames = frames[start_idx:i] + frames[i+1:end_idx]
                
                score, detailed_scores = self.score_frame_importance(frame, context_frames, i)
                importance_scores.append((frame_indices[i], score, detailed_scores))
                pbar.set_postfix({"score": f"{score:.3f}"})
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select frames ensuring minimum gap
        selected_frames = []
        selected_indices = set()
        
        target_count = max(10, int(duration * target_fps))
        
        for frame_idx, score, details in importance_scores:
            if len(selected_frames) >= target_count:
                break
            
            # Check minimum gap constraint
            too_close = False
            for selected_idx in selected_indices:
                if abs(frame_idx - selected_idx) < min_gap * step:
                    too_close = True
                    break
            
            if not too_close:
                selected_frames.append((frame_idx, score))
                selected_indices.add(frame_idx)
        
        # Sort selected frames by frame index
        selected_frames.sort(key=lambda x: x[0])
        
        print(f"Selected {len(selected_frames)} frames with avg importance {np.mean([s[1] for s in selected_frames]):.3f}")
        
        return selected_frames
    
    def process_video(self, video_id):
        """Process a single video with advanced intelligent sampling"""
        print(f"\n=== Processing video collection: {video_id} ===")
        
        # Find video files for this collection - expand collection ID to individual videos
        video_dir = self.dataset_root / "videos"
        if not video_dir.exists():
            print(f"ERROR: Videos directory not found: {video_dir}")
            return False
        
        # If video_id is a collection (L21), find all videos in that collection
        if len(video_id) == 3 and video_id.startswith('L'):
            video_files = sorted(list(video_dir.glob(f"{video_id}_V*.mp4")))
        else:
            # Individual video ID
            video_files = [video_dir / f"{video_id}.mp4"]
            
        if not video_files:
            print(f"ERROR: No video files found for collection {video_id}")
            return False
            
        print(f"Found {len(video_files)} videos in collection {video_id}")
        
        success_count = 0
        with tqdm(video_files, desc=f"Processing {video_id}", unit="video") as pbar:
            for video_file in pbar:
                pbar.set_postfix({"file": video_file.stem})
                if self.process_single_video(video_file):
                    success_count += 1
                pbar.set_postfix({"file": video_file.stem, "success": f"{success_count}/{len(video_files)}"})
                
        print(f"Successfully processed {success_count}/{len(video_files)} videos in collection {video_id}")
        return success_count > 0
    
    def process_single_video(self, video_path):
        """Process a single video file"""
        video_path = Path(video_path)
        video_id = video_path.stem  # L21_V001
        
        print(f"Processing video: {video_id}")
        
        if not video_path.exists():
            print(f"ERROR: Video file not found: {video_path}")
            return False
        
        # Run intelligent sampling on this video
        selected_frames = self.smart_sampling(video_path)
        
        if not selected_frames:
            print(f"No frames selected for {video_id}")
            return False
        
        # Save keyframes and mapping
        output_dir = self.dataset_root / "keyframes_intelligent" / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mapping_data = []
        
        # Extract and save frames
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        with tqdm(enumerate(selected_frames), desc=f"Saving keyframes for {video_id}", total=len(selected_frames), unit="frame") as pbar:
            for i, (frame_idx, importance_score) in pbar:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"WARNING: Could not read frame {frame_idx}")
                    continue
                    
                frame_filename = f"{i+1:03d}.png"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                pts_time = (frame_idx / fps) if fps else 0.0
                mapping_data.append({
                    'n': i + 1,
                    'pts_time': pts_time,
                    'fps': fps,
                    'frame_idx': frame_idx,
                    'importance_score': importance_score
                })
                pbar.set_postfix({"frame": frame_filename, "score": f"{importance_score:.3f}"})
        
        cap.release()
        
        # Save mapping file
        mapping_file = self.dataset_root / "map_keyframes" / f"{video_id}.csv"
        mapping_file.parent.mkdir(exist_ok=True)
        
        df = pd.DataFrame(mapping_data)
        df.to_csv(mapping_file, index=False)
        
        print(f"Saved {len(mapping_data)} intelligent keyframes for {video_id}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Advanced intelligent frame sampling")
    parser.add_argument("--dataset_root", required=True, help="Dataset root directory")
    parser.add_argument("--videos", nargs="+", required=True, help="Video IDs to process")
    parser.add_argument("--target_fps", type=float, default=0.5, help="Target frames per second")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--window_size", type=int, default=16, help="Context window size")
    parser.add_argument("--min_gap", type=int, default=8, help="Minimum gap between selected frames")
    
    args = parser.parse_args()
    
    sampler = AdvancedFrameSampler(
        args.dataset_root, 
        use_gpu=args.use_gpu
    )
    
    success_count = 0
    with tqdm(args.videos, desc="Processing video collections", unit="collection") as pbar:
        for video_id in pbar:
            pbar.set_postfix({"collection": video_id})
            if sampler.process_video(video_id):
                success_count += 1
            pbar.set_postfix({"collection": video_id, "success": f"{success_count}/{len(args.videos)}"})
    
    print(f"\nCompleted processing {success_count}/{len(args.videos)} video collections")
    
    return 0 if success_count == len(args.videos) else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
CLIP/SigLIP-guided intelligent frame sampling for optimal retrieval performance.
Uses semantic diversity, scene detection, and query-relevance scoring.
"""

import os
import sys
import argparse
from pathlib import Path
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Conservative dependency handling - only warn about missing dependencies
try:
    import cv2
    import numpy as np
    from tqdm import tqdm
    import torch
    from PIL import Image
    import open_clip
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import pdist, squareform
except ImportError as e:
    missing_dep = str(e).split("'")[1] if "'" in str(e) else str(e)
    print(f"Missing dependency: {missing_dep}")
    print("Please install dependencies with one of:")
    print("  pip install -r requirements.txt")
    print("  conda env create -f environment.yml") 
    print("  or set AIC_FORCE_INSTALL=1 environment variable")
    sys.exit(1)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import utils
import config

class CLIPGuidedFrameSampler:
    def __init__(self, dataset_root, use_gpu=True, batch_size=32, model_name=None):
        self.dataset_root = Path(dataset_root)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.batch_size = batch_size
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # Initialize CLIP model (use config model or override)
        model_name = model_name or config.MODEL_NAME
        pretrained = config.MODEL_PRETRAINED
        
        print(f"Loading {model_name} with {pretrained} on {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        
        # Common query templates for relevance scoring
        self.query_templates = [
            "a person",
            "an object", 
            "an action",
            "a scene",
            "people interacting",
            "text or writing",
            "a face",
            "hands or gestures",
            "movement or motion",
            "indoor scene",
            "outdoor scene",
            "close-up view",
            "wide shot view"
        ]
        
        # Precompute query embeddings
        print("Precomputing query embeddings...")
        self.query_embeddings = self._encode_text_batch(self.query_templates)
        
    def _encode_text_batch(self, texts):
        """Encode text queries to embeddings"""
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embeddings = self.model.encode_text(tokens)
            else:
                embeddings = self.model.encode_text(tokens)
        return embeddings.cpu().numpy()
    
    def _encode_image_batch(self, images):
        """Encode batch of PIL images to embeddings"""
        if not images:
            return np.array([])
            
        batch_tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensor = self.preprocess(img).unsqueeze(0)
            batch_tensors.append(tensor)
        
        batch = torch.cat(batch_tensors).to(self.device)
        
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embeddings = self.model.encode_image(batch)
            else:
                embeddings = self.model.encode_image(batch)
        
        return embeddings.cpu().numpy()
    
    def detect_scene_boundaries(self, frames, threshold=0.4):
        """Detect scene boundaries using CLIP embeddings"""
        if len(frames) < 2:
            return []
        
        print("Detecting scene boundaries with CLIP...")
        
        # Process frames in batches
        embeddings = []
        for i in tqdm(range(0, len(frames), self.batch_size), desc="Encoding frames"):
            batch = frames[i:i+self.batch_size]
            batch_embeddings = self._encode_image_batch(batch)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Compute similarity between consecutive frames
        scene_boundaries = []
        similarities = []
        
        for i in range(1, len(embeddings)):
            similarity = cosine_similarity(
                embeddings[i-1:i], embeddings[i:i+1]
            )[0, 0]
            similarities.append(similarity)
            
            if similarity < threshold:
                scene_boundaries.append(i)
        
        print(f"Detected {len(scene_boundaries)} scene boundaries")
        return scene_boundaries, similarities
    
    def compute_frame_relevance(self, frame_embeddings):
        """Score frames by relevance to common queries"""
        # Compute similarity to all query templates
        similarities = cosine_similarity(frame_embeddings, self.query_embeddings)
        
        # Aggregate scores (max relevance to any query)
        relevance_scores = np.max(similarities, axis=1)
        
        return relevance_scores
    
    def compute_diversity_score(self, frame_embeddings, selected_indices):
        """Score frames by diversity from already selected frames"""
        if not selected_indices:
            return np.ones(len(frame_embeddings))
        
        selected_embeddings = frame_embeddings[selected_indices]
        
        diversity_scores = []
        for i, embedding in enumerate(frame_embeddings):
            if i in selected_indices:
                diversity_scores.append(0)  # Already selected
            else:
                # Minimum distance to any selected frame
                distances = 1 - cosine_similarity(
                    embedding.reshape(1, -1), selected_embeddings
                )[0]
                diversity_scores.append(np.min(distances))
        
        return np.array(diversity_scores)
    
    def sample_frames_intelligent(self, video_path, target_frames=30, 
                                 diversity_weight=0.4, relevance_weight=0.4, 
                                 temporal_weight=0.2, min_gap_seconds=1.0):
        """
        Intelligent frame sampling using CLIP guidance
        
        Args:
            video_path: Path to video file
            target_frames: Number of frames to sample
            diversity_weight: Weight for semantic diversity
            relevance_weight: Weight for query relevance  
            temporal_weight: Weight for temporal distribution
            min_gap_seconds: Minimum time gap between selected frames
        """
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"Processing {video_path.name}: {frame_count} frames, {fps:.1f} FPS, {duration:.1f}s")
        
        if frame_count == 0:
            cap.release()
            return []
        
        # Sample initial frames more densely for analysis
        if frame_count > 3000:  # Long video
            initial_step = max(1, int(fps / 2))  # 2 FPS
        else:
            initial_step = max(1, int(fps / 4))  # 4 FPS
        
        # Load frames for analysis
        frames = []
        frame_indices = []
        frame_times = []
        
        positions = list(range(0, frame_count, initial_step))
        
        with tqdm(positions, desc=f"Loading frames from {video_path.name}") as pbar:
            for pos in pbar:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
                    frame_indices.append(pos)
                    frame_times.append(pos / fps)
                pbar.set_postfix({"loaded": len(frames)})
        
        cap.release()
        
        if len(frames) == 0:
            return []
        
        print(f"Loaded {len(frames)} frames for analysis")
        
        # Encode all frames to CLIP embeddings
        print("Computing CLIP embeddings...")
        all_embeddings = []
        for i in tqdm(range(0, len(frames), self.batch_size), desc="Encoding frames"):
            batch = frames[i:i+self.batch_size]
            batch_embeddings = self._encode_image_batch(batch)
            all_embeddings.append(batch_embeddings)
        
        all_embeddings = np.vstack(all_embeddings)
        
        # Detect scene boundaries
        scene_boundaries, similarities = self.detect_scene_boundaries(frames, threshold=0.4)
        
        # Compute relevance scores
        print("Computing relevance scores...")
        relevance_scores = self.compute_frame_relevance(all_embeddings)
        
        # Greedy selection with multiple criteria
        selected_indices = []
        target_frames = min(target_frames, len(frames))
        min_gap_frames = max(1, int(min_gap_seconds * fps / initial_step))
        
        print(f"Selecting {target_frames} frames with min gap of {min_gap_frames} frames...")
        
        with tqdm(total=target_frames, desc="Selecting frames") as pbar:
            for _ in range(target_frames):
                if len(selected_indices) >= len(frames):
                    break
                
                # Compute diversity scores
                diversity_scores = self.compute_diversity_score(all_embeddings, selected_indices)
                
                # Compute temporal distribution scores
                temporal_scores = np.ones(len(frames))
                if len(selected_indices) > 0:
                    selected_times = np.array([frame_times[i] for i in selected_indices])
                    for i, t in enumerate(frame_times):
                        if i in selected_indices:
                            temporal_scores[i] = 0
                        else:
                            # Prefer frames that fill temporal gaps
                            min_time_dist = np.min(np.abs(selected_times - t))
                            temporal_scores[i] = min_time_dist / duration
                
                # Combined score
                combined_scores = (
                    diversity_weight * diversity_scores +
                    relevance_weight * relevance_scores +
                    temporal_weight * temporal_scores
                )
                
                # Apply minimum gap constraint
                for selected_idx in selected_indices:
                    gap_start = max(0, selected_idx - min_gap_frames)
                    gap_end = min(len(frames), selected_idx + min_gap_frames + 1)
                    combined_scores[gap_start:gap_end] = 0
                
                # Select best frame
                best_idx = np.argmax(combined_scores)
                if combined_scores[best_idx] > 0:
                    selected_indices.append(best_idx)
                    pbar.update(1)
                    pbar.set_postfix({
                        "diversity": f"{diversity_scores[best_idx]:.3f}",
                        "relevance": f"{relevance_scores[best_idx]:.3f}",
                        "frame": frame_indices[best_idx]
                    })
                else:
                    break
        
        # Convert to original frame indices and sort by time
        result = [(frame_indices[i], frame_times[i], {
            'relevance': float(relevance_scores[i]),
            'diversity': float(self.compute_diversity_score(all_embeddings, [i])[i]),
            'scene_boundary': i in scene_boundaries
        }) for i in selected_indices]
        
        result.sort(key=lambda x: x[0])  # Sort by frame index
        
        print(f"Selected {len(result)} frames spanning {result[-1][1] - result[0][1]:.1f}s")
        return result
    
    def process_video_batch(self, video_paths, target_frames=30, output_dir=None):
        """Process multiple videos"""
        results = {}
        
        for video_path in tqdm(video_paths, desc="Processing videos"):
            try:
                selected = self.sample_frames_intelligent(
                    video_path, target_frames=target_frames
                )
                results[str(video_path)] = selected
                
                if output_dir:
                    output_file = Path(output_dir) / f"{video_path.stem}_frames.json"
                    with open(output_file, 'w') as f:
                        json.dump(selected, f, indent=2)
                        
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results[str(video_path)] = []
        
        return results

def main():
    parser = argparse.ArgumentParser(description="CLIP-guided intelligent frame sampling")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Root directory of video dataset")
    parser.add_argument("--output_dir", type=Path, default="./sampled_frames", help="Output directory for results")
    parser.add_argument("--target_frames", type=int, default=30, help="Target number of frames per video")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for CLIP encoding")
    parser.add_argument("--video_pattern", type=str, default="*.mp4", help="Video file pattern")
    parser.add_argument("--model_name", type=str, default=None, help="Override model name from config")
    parser.add_argument("--min_gap_seconds", type=float, default=1.0, help="Minimum gap between frames (seconds)")
    parser.add_argument("--diversity_weight", type=float, default=0.4, help="Weight for diversity scoring")
    parser.add_argument("--relevance_weight", type=float, default=0.4, help="Weight for relevance scoring")
    parser.add_argument("--temporal_weight", type=float, default=0.2, help="Weight for temporal distribution")
    
    args = parser.parse_args()
    
    # Find video files
    video_files = list(args.dataset_root.rglob(args.video_pattern))
    print(f"Found {len(video_files)} video files")
    
    if not video_files:
        print(f"No videos found matching {args.video_pattern} in {args.dataset_root}")
        return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize sampler
    sampler = CLIPGuidedFrameSampler(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        model_name=args.model_name
    )
    
    # Process videos
    results = sampler.process_video_batch(
        video_files,
        target_frames=args.target_frames,
        output_dir=args.output_dir
    )
    
    # Save summary
    summary_file = args.output_dir / "sampling_summary.json"
    with open(summary_file, 'w') as f:
        summary = {
            'total_videos': len(video_files),
            'target_frames_per_video': args.target_frames,
            'model_used': args.model_name or config.MODEL_NAME,
            'parameters': {
                'diversity_weight': args.diversity_weight,
                'relevance_weight': args.relevance_weight, 
                'temporal_weight': args.temporal_weight,
                'min_gap_seconds': args.min_gap_seconds
            },
            'results': {str(k): len(v) for k, v in results.items()}
        }
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {args.output_dir}")
    print(f"Summary: {summary_file}")

if __name__ == "__main__":
    main()
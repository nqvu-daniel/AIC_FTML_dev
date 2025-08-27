"""TransNet-V2 based academic-grade keyframe extraction"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import torch

from ..core.base import DataProcessor, VideoData

try:
    import transnetv2_pytorch as tnv2
    TRANSNET_AVAILABLE = True
    print("✅ TransNet-V2 available for academic-grade shot boundary detection")
except ImportError:
    TRANSNET_AVAILABLE = False
    print("⚠️ TransNet-V2 not available. Install: pip install transnetv2-pytorch")


class TransNetKeyframeExtractor(DataProcessor):
    """Academic-grade keyframe extraction using TransNet-V2 shot boundary detection + intelligent sampling"""
    
    def __init__(self, 
                 target_frames: int = 50,
                 min_gap_seconds: float = 1.0,
                 transnet_threshold: float = 0.5,
                 use_intelligent_refinement: bool = True,
                 complexity_weight: float = 0.3,
                 motion_weight: float = 0.2,
                 boundary_weight: float = 0.5,  # Higher weight for TransNet boundaries
                 device: str = None,
                 config: Dict[str, Any] = None):
        """
        Initialize TransNet-V2 + intelligent sampling keyframe extractor
        
        Args:
            target_frames: Target number of keyframes to extract
            min_gap_seconds: Minimum gap between selected keyframes
            transnet_threshold: TransNet-V2 detection threshold (paper uses 0.5)
            use_intelligent_refinement: Add intelligent sampling refinement
            complexity_weight: Weight for visual complexity scoring
            motion_weight: Weight for motion analysis
            boundary_weight: Weight for TransNet boundaries (academic focus)
            device: Device for TransNet-V2 ('cuda', 'cpu', 'mps', or None for auto)
            config: Additional configuration
        """
        super().__init__("TransNetKeyframeExtractor", config)
        
        self.target_frames = target_frames
        self.min_gap_seconds = min_gap_seconds
        self.transnet_threshold = transnet_threshold
        self.use_intelligent_refinement = use_intelligent_refinement
        self.complexity_weight = complexity_weight
        self.motion_weight = motion_weight  
        self.boundary_weight = boundary_weight
        
        # Initialize TransNet-V2 model
        if TRANSNET_AVAILABLE:
            if device is None:
                # Auto-detect device
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            self.device = device
            print(f"🎬 Loading TransNet-V2 on {device}")
            
            try:
                # Initialize TransNet-V2 model
                self.model = tnv2.TransNetV2(device=device)
                self.available = True
                print("✅ TransNet-V2 loaded successfully - academic-grade shot boundary detection ready")
            except Exception as e:
                print(f"❌ Failed to load TransNet-V2: {e}")
                self.model = None
                self.available = False
        else:
            self.model = None
            self.available = False
            print("❌ TransNet-V2 not available - falling back to intelligent sampling")
    
    def process(self, video_data: VideoData) -> VideoData:
        """Extract keyframes using TransNet-V2 shot boundaries + intelligent sampling"""
        
        if not self.available:
            print("⚠️ TransNet-V2 not available, using intelligent sampling fallback")
            return self._fallback_intelligent_sampling(video_data)
        
        print(f"🎬 Processing {video_data.video_id} with TransNet-V2")
        
        try:
            # Step 1: TransNet-V2 shot boundary detection
            predictions = self.model.predict_frames(str(video_data.video_path))
            scene_boundaries = self._extract_scene_boundaries(predictions)
            
            print(f"🔍 TransNet-V2 detected {len(scene_boundaries)} shot boundaries")
            
            # Step 2: Extract candidate keyframes around boundaries
            candidates = self._extract_boundary_candidates(video_data, scene_boundaries)
            
            # Step 3: Apply intelligent refinement if enabled
            if self.use_intelligent_refinement and candidates:
                candidates = self._apply_intelligent_refinement(video_data, candidates)
            
            # Step 4: Select final keyframes with temporal constraints
            keyframes = self._select_final_keyframes(candidates)
            
            print(f"✅ Selected {len(keyframes)} keyframes using TransNet-V2 + intelligent sampling")
            
            video_data.keyframes = keyframes
            return video_data
            
        except Exception as e:
            print(f"❌ TransNet-V2 processing failed: {e}")
            print("🔄 Falling back to intelligent sampling")
            return self._fallback_intelligent_sampling(video_data)
    
    def _extract_scene_boundaries(self, predictions: np.ndarray) -> List[int]:
        """Extract scene boundary frame indices from TransNet-V2 predictions"""
        # Apply threshold (paper uses 0.5)
        binary_predictions = (predictions >= self.transnet_threshold).astype(np.uint8)
        
        if binary_predictions.sum() == 0:
            return []
        
        # Find transition spans (consecutive frames above threshold)
        diff = np.diff(np.pad(binary_predictions, (1, 1)))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        
        # Take center of each span as boundary (as per paper)
        boundaries = []
        for start, end in zip(starts, ends):
            center = int(round((start + end) / 2))
            boundaries.append(center)
        
        return boundaries
    
    def _extract_boundary_candidates(self, video_data: VideoData, boundaries: List[int]) -> List[Dict[str, Any]]:
        """Extract candidate keyframes around TransNet-V2 detected boundaries"""
        cap = cv2.VideoCapture(str(video_data.video_path))
        fps = video_data.metadata.get("fps", 30)
        frame_count = video_data.metadata.get("frame_count", 0)
        
        if not cap.isOpened() or frame_count == 0:
            cap.release()
            return []
        
        candidates = []
        
        # If no boundaries detected, fall back to uniform sampling
        if not boundaries:
            print("🔄 No TransNet boundaries found, using uniform candidates")
            step = max(1, frame_count // self.target_frames)
            boundary_candidates = list(range(0, frame_count, step))
        else:
            # Use detected boundaries + add some uniform samples for coverage
            boundary_candidates = boundaries.copy()
            
            # Add uniform samples between boundaries for better coverage
            if len(boundaries) < self.target_frames:
                uniform_step = max(1, frame_count // (self.target_frames - len(boundaries)))
                uniform_candidates = list(range(0, frame_count, uniform_step))
                boundary_candidates.extend(uniform_candidates)
        
        # Remove duplicates and sort
        boundary_candidates = sorted(list(set(boundary_candidates)))
        
        # Extract frames and compute initial scores
        prev_frame = None
        for frame_idx in tqdm(boundary_candidates, desc="Extracting boundary candidates"):
            if frame_idx >= frame_count:
                continue
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            timestamp = frame_idx / fps
            
            # Check if this is a TransNet boundary
            is_boundary = frame_idx in boundaries
            boundary_score = 1.0 if is_boundary else 0.0
            
            # Compute additional metrics for intelligent refinement
            complexity_score = self._compute_visual_complexity(frame)
            motion_score = self._compute_motion_score(frame, prev_frame) if prev_frame is not None else 0.5
            
            # Combined importance score with TransNet boundary bias
            importance = (self.boundary_weight * boundary_score + 
                         self.complexity_weight * complexity_score +
                         self.motion_weight * motion_score)
            
            candidates.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "frame": frame.copy(),
                "frame_path": None,
                "importance": importance,
                "is_transnet_boundary": is_boundary,
                "boundary_score": boundary_score,
                "complexity": complexity_score,
                "motion": motion_score,
                "sampling_method": "transnet_v2"
            })
            
            prev_frame = frame
        
        cap.release()
        return candidates
    
    def _apply_intelligent_refinement(self, video_data: VideoData, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply intelligent sampling refinement to TransNet candidates"""
        print("🧠 Applying intelligent refinement to TransNet candidates")
        
        # Re-score candidates with updated metrics
        for i, candidate in enumerate(candidates):
            frame = candidate["frame"]
            
            # Enhanced visual complexity with edge density
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges) / 255.0
            
            # Color diversity (histogram entropy)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist = hist.flatten()
            hist = hist[hist > 0]
            if len(hist) > 1:
                hist = hist / np.sum(hist)
                color_entropy = -np.sum(hist * np.log(hist)) / np.log(180)
            else:
                color_entropy = 0.0
            
            enhanced_complexity = (edge_density * 0.6 + color_entropy * 0.4)
            
            # Update candidate with enhanced metrics
            candidate["enhanced_complexity"] = enhanced_complexity
            
            # Recompute importance with enhanced metrics
            candidate["importance"] = (
                self.boundary_weight * candidate["boundary_score"] + 
                self.complexity_weight * enhanced_complexity +
                self.motion_weight * candidate["motion"]
            )
        
        return candidates
    
    def _select_final_keyframes(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select final keyframes with temporal constraints"""
        if not candidates:
            return []
        
        # Sort by importance (highest first)
        candidates.sort(key=lambda x: x["importance"], reverse=True)
        
        min_gap_frames = int(self.min_gap_seconds * 30)  # Assume 30 fps average
        
        selected = []
        for candidate in candidates:
            # Check temporal constraints
            valid = True
            for selected_frame in selected:
                frame_gap = abs(candidate["frame_idx"] - selected_frame["frame_idx"])
                if frame_gap < min_gap_frames:
                    valid = False
                    break
            
            if valid:
                # Add metadata tags
                if candidate.get("is_transnet_boundary", False):
                    candidate["is_scene_boundary"] = True
                
                if candidate["importance"] > 0.8:
                    candidate["relevance_score"] = candidate["importance"]
                
                # Remove frame data to save memory
                if "frame" in candidate:
                    del candidate["frame"]
                
                selected.append(candidate)
                
                # Stop when we have enough frames
                if len(selected) >= self.target_frames:
                    break
        
        # Sort by timestamp for temporal order
        selected.sort(key=lambda x: x["frame_idx"])
        
        return selected
    
    def _compute_visual_complexity(self, frame: np.ndarray) -> float:
        """Compute visual complexity score"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        return np.clip(edge_density, 0.0, 1.0)
    
    def _compute_motion_score(self, current_frame: np.ndarray, prev_frame: np.ndarray) -> float:
        """Compute motion score using frame difference"""
        if prev_frame is None:
            return 0.5
        
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_gray, prev_gray)
        motion_score = np.mean(diff) / 255.0
        return np.clip(motion_score, 0.0, 1.0)
    
    def _fallback_intelligent_sampling(self, video_data: VideoData) -> VideoData:
        """Fallback to basic intelligent sampling if TransNet-V2 unavailable"""
        from .video_processor import KeyframeExtractor
        
        print("🔄 Using intelligent sampling fallback")
        fallback_extractor = KeyframeExtractor(
            target_frames=self.target_frames,
            min_gap_seconds=self.min_gap_seconds
        )
        return fallback_extractor.process(video_data)


# Alias for backward compatibility
AcademicKeyframeExtractor = TransNetKeyframeExtractor
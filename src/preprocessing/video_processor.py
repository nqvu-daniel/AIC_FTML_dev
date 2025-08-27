"""Video processing and keyframe extraction"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from ..core.base import DataProcessor, VideoData


class VideoProcessor(DataProcessor):
    """Handles video loading and basic processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("VideoProcessor", config)
        
    def process(self, video_path: Path) -> VideoData:
        """Process a video file and extract basic metadata"""
        video_id = video_path.stem
        
        # Extract video metadata
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        metadata = {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        
        cap.release()
        
        return VideoData(video_id, video_path, metadata)


class KeyframeExtractor(DataProcessor):
    """Extract keyframes from video using intelligent sampling"""
    
    def __init__(self, target_frames: int = 50, min_gap_seconds: float = 1.0, config: Dict[str, Any] = None):
        super().__init__("KeyframeExtractor", config)
        self.target_frames = target_frames
        self.min_gap_seconds = min_gap_seconds
        
    def process(self, video_data: VideoData) -> VideoData:
        """Extract keyframes and add them to video_data"""
        cap = cv2.VideoCapture(str(video_data.video_path))
        
        if not cap.isOpened():
            return video_data
            
        fps = video_data.metadata.get("fps", 30)
        frame_count = video_data.metadata.get("frame_count", 0)
        
        # Simple uniform sampling for now (can be replaced with intelligent sampling)
        if frame_count == 0:
            cap.release()
            return video_data
            
        step = max(1, frame_count // self.target_frames)
        min_gap_frames = max(1, int(self.min_gap_seconds * fps))
        
        keyframes = []
        for frame_idx in range(0, frame_count, max(step, min_gap_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                timestamp = frame_idx / fps
                keyframes.append({
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "frame": frame,
                    "frame_path": None  # Will be set if saved
                })
                
        cap.release()
        video_data.keyframes = keyframes
        return video_data


class KeyframeSaver(DataProcessor):
    """Save extracted keyframes to disk"""
    
    def __init__(self, output_dir: Path, quality: int = 95, config: Dict[str, Any] = None):
        super().__init__("KeyframeSaver", config)
        self.output_dir = Path(output_dir)
        self.quality = quality
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process(self, video_data: VideoData) -> VideoData:
        """Save keyframes and update paths in video_data"""
        for i, keyframe in enumerate(video_data.keyframes):
            if "frame" in keyframe and keyframe["frame"] is not None:
                # Create filename
                filename = f"{video_data.video_id}_frame_{keyframe['frame_idx']:06d}.jpg"
                frame_path = self.output_dir / filename
                
                # Save frame
                cv2.imwrite(
                    str(frame_path), 
                    keyframe["frame"], 
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                
                # Update keyframe data
                keyframe["frame_path"] = str(frame_path)
                # Remove frame data to save memory
                del keyframe["frame"]
                
        return video_data
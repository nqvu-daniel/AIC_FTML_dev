"""Text processing and corpus building"""

import re
from typing import List, Dict, Any
from pathlib import Path

from ..core.base import DataProcessor, VideoData


class MetadataProcessor(DataProcessor):
    """Process video metadata into text descriptions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("MetadataProcessor", config)
        
    def process(self, video_data: VideoData) -> VideoData:
        """Generate text descriptions from video metadata and keyframes"""
        descriptions = []
        
        for keyframe in video_data.keyframes:
            # Extract collection and video number from video ID
            collection_match = re.match(r'(L\d+)_V(\d+)', video_data.video_id)
            if collection_match:
                collection = collection_match.group(1)
                video_num = collection_match.group(2)
            else:
                collection = "unknown"
                video_num = "unknown"
            
            # Create descriptive text
            text_parts = [
                f"video {video_data.video_id}",
                f"collection {collection}",
                f"video number {video_num}",
                f"timestamp {keyframe['timestamp']:.1f} seconds"
            ]
            
            # Add frame-specific info if available
            if "relevance_score" in keyframe:
                if keyframe["relevance_score"] > 0.7:
                    text_parts.append("high relevance frame")
                elif keyframe["relevance_score"] > 0.5:
                    text_parts.append("medium relevance frame")
                    
            if keyframe.get("is_scene_boundary", False):
                text_parts.append("scene boundary frame")
                
            description = " ".join(text_parts)
            descriptions.append(description)
            
        video_data.descriptions = descriptions
        return video_data


class TextCorpusBuilder(DataProcessor):
    """Build searchable text corpus from video descriptions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("TextCorpusBuilder", config)
        
    def process(self, video_data_list: List[VideoData]) -> List[Dict[str, Any]]:
        """Build corpus from multiple video data objects"""
        corpus_entries = []
        
        for video_data in video_data_list:
            for i, description in enumerate(video_data.descriptions):
                if i < len(video_data.keyframes):
                    keyframe = video_data.keyframes[i]
                    
                    # Combine multiple text sources
                    text_sources = [description]
                    
                    # Add OCR text if available
                    if hasattr(video_data, 'ocr_texts') and i < len(video_data.ocr_texts):
                        ocr_text = video_data.ocr_texts[i]
                        if ocr_text.strip():
                            text_sources.append(f"text: {ocr_text}")
                            
                    # Add captions if available
                    if hasattr(video_data, 'captions') and i < len(video_data.captions):
                        caption = video_data.captions[i]
                        if caption.strip():
                            text_sources.append(f"caption: {caption}")
                    
                    # Combine all text sources
                    combined_text = " ".join(text_sources)
                    
                    # Simple tokenization
                    tokens = combined_text.lower().replace(",", "").split()
                    
                    corpus_entries.append({
                        "video_id": video_data.video_id,
                        "frame_idx": keyframe["frame_idx"],
                        "timestamp": keyframe["timestamp"],
                        "raw": combined_text,
                        "tokens": tokens,
                        "has_ocr": bool(hasattr(video_data, 'ocr_texts') and i < len(video_data.ocr_texts) and video_data.ocr_texts[i].strip()),
                        "has_caption": bool(hasattr(video_data, 'captions') and i < len(video_data.captions) and video_data.captions[i].strip())
                    })
                    
        return corpus_entries


class OCRProcessor(DataProcessor):
    """Process OCR text from keyframes using EasyOCR"""
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = True, config: Dict[str, Any] = None):
        super().__init__("OCRProcessor", config)
        
        # Initialize EasyOCR encoder
        try:
            from ..encoders.ocr_encoder import EasyOCREncoder
            self.ocr_encoder = EasyOCREncoder(languages=languages, gpu=use_gpu)
        except ImportError:
            print("Warning: Could not import EasyOCREncoder")
            self.ocr_encoder = None
        
    def process(self, video_data: VideoData) -> VideoData:
        """Extract OCR text from keyframes using EasyOCR"""
        if not self.ocr_encoder or not self.ocr_encoder.available:
            video_data.ocr_texts = ["" for _ in video_data.keyframes]
            return video_data
            
        ocr_texts = []
        frame_paths = []
        
        # Collect frame paths
        for keyframe in video_data.keyframes:
            if "frame_path" in keyframe and keyframe["frame_path"]:
                frame_paths.append(keyframe["frame_path"])
            else:
                frame_paths.append("")
                
        if frame_paths:
            try:
                # Extract OCR text from all frames
                valid_paths = [p for p in frame_paths if p]
                if valid_paths:
                    extracted_texts = self.ocr_encoder.encode(valid_paths)
                    
                    # Map back to original frame order
                    text_idx = 0
                    for path in frame_paths:
                        if path:
                            ocr_texts.append(extracted_texts[text_idx] if text_idx < len(extracted_texts) else "")
                            text_idx += 1
                        else:
                            ocr_texts.append("")
                else:
                    ocr_texts = ["" for _ in frame_paths]
                    
            except Exception as e:
                print(f"OCR processing error: {e}")
                ocr_texts = ["" for _ in frame_paths]
        else:
            ocr_texts = []
            
        video_data.ocr_texts = ocr_texts
        return video_data


class CaptionProcessor(DataProcessor):
    """Generate image captions using BLIP-2"""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", 
                 use_gpu: bool = True, config: Dict[str, Any] = None):
        super().__init__("CaptionProcessor", config)
        
        # Initialize BLIP captioner
        try:
            from ..encoders.blip_encoder import BLIPCaptioner
            device = "cuda" if use_gpu else "cpu"
            self.captioner = BLIPCaptioner(model_name=model_name, device=device)
        except ImportError:
            print("Warning: Could not import BLIPCaptioner")
            self.captioner = None
            
    def process(self, video_data: VideoData) -> VideoData:
        """Generate captions for keyframes using BLIP-2"""
        if not self.captioner or not self.captioner.available:
            video_data.captions = ["" for _ in video_data.keyframes]
            return video_data
            
        frame_paths = []
        
        # Collect frame paths
        for keyframe in video_data.keyframes:
            if "frame_path" in keyframe and keyframe["frame_path"]:
                frame_paths.append(keyframe["frame_path"])
            else:
                frame_paths.append("")
                
        if frame_paths:
            try:
                # Generate captions for all frames
                valid_paths = [p for p in frame_paths if p]
                if valid_paths:
                    captions = self.captioner.encode(valid_paths)
                    
                    # Map back to original frame order
                    caption_idx = 0
                    final_captions = []
                    for path in frame_paths:
                        if path:
                            final_captions.append(captions[caption_idx] if caption_idx < len(captions) else "")
                            caption_idx += 1
                        else:
                            final_captions.append("")
                else:
                    final_captions = ["" for _ in frame_paths]
                    
            except Exception as e:
                print(f"Caption processing error: {e}")
                final_captions = ["" for _ in frame_paths]
        else:
            final_captions = []
            
        video_data.captions = final_captions
        return video_data
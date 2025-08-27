"""EasyOCR encoder for text extraction (Best balance of accuracy/speed)"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
from pathlib import Path

from ..core.base import Encoder


class EasyOCREncoder(Encoder):
    """EasyOCR encoder for text extraction from images (supports 70+ languages)"""
    
    def __init__(self, languages: List[str] = None, gpu: bool = True, config_dict: Dict[str, Any] = None):
        super().__init__("EasyOCREncoder", config_dict)
        self.languages = languages or ['en', 'vi']  # English and Vietnamese for AIC
        self.gpu = gpu
        
        try:
            import easyocr
            print(f"Loading EasyOCR with languages: {self.languages}, GPU: {self.gpu}")
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            self.available = True
            print("EasyOCR loaded successfully")
        except ImportError:
            print("Warning: easyocr not installed. Run: pip install easyocr")
            self.reader = None
            self.available = False
        except Exception as e:
            print(f"Warning: Could not initialize EasyOCR: {e}")
            self.reader = None
            self.available = False
        
    def encode(self, data: Union[List[Image.Image], List[str]]) -> List[str]:
        """Extract text from images as simple strings"""
        if not self.available:
            return ["" for _ in (data if isinstance(data, list) else [data])]
            
        if not isinstance(data, list):
            data = [data]
            
        texts = []
        for item in data:
            try:
                if isinstance(item, (str, Path)):
                    # Read from file path
                    results = self.reader.readtext(str(item))
                else:
                    # PIL Image - convert to numpy array
                    image_array = np.array(item)
                    results = self.reader.readtext(image_array)
                    
                # Extract text from results (EasyOCR returns [bbox, text, confidence])
                extracted_text = " ".join([result[1] for result in results if len(result) > 1])
                texts.append(extracted_text.strip())
                
            except Exception as e:
                print(f"EasyOCR encoding error: {e}")
                texts.append("")
                
        return texts
        
    def process(self, images: Union[List[Union[Image.Image, str]], Union[Image.Image, str]]) -> List[Dict[str, Any]]:
        """Process images and return detailed OCR results with positions and confidence"""
        if not self.available:
            return [{"text": "", "texts": [], "boxes": [], "scores": [], "num_detections": 0}]
            
        if not isinstance(images, list):
            images = [images]
            
        results = []
        for image in images:
            try:
                if isinstance(image, (str, Path)):
                    ocr_results = self.reader.readtext(str(image))
                else:
                    image_array = np.array(image)
                    ocr_results = self.reader.readtext(image_array)
                    
                # Format results (EasyOCR format: [bbox, text, confidence])
                texts = []
                boxes = []
                scores = []
                
                for result in ocr_results:
                    if len(result) >= 3:
                        bbox, text, confidence = result
                        texts.append(text)
                        boxes.append(bbox)  # List of 4 corner points
                        scores.append(float(confidence))
                        
                # Create combined text
                combined_text = " ".join(texts).strip()
                        
                results.append({
                    "text": combined_text,
                    "texts": texts,
                    "boxes": boxes,
                    "scores": scores,
                    "num_detections": len(texts),
                    "avg_confidence": np.mean(scores) if scores else 0.0,
                    "languages": self.languages
                })
                
            except Exception as e:
                print(f"EasyOCR processing error: {e}")
                results.append({
                    "text": "",
                    "texts": [],
                    "boxes": [],
                    "scores": [],
                    "num_detections": 0,
                    "avg_confidence": 0.0,
                    "languages": self.languages
                })
                
        return results
        
    def extract_from_keyframes(self, keyframe_paths: List[str]) -> Dict[str, str]:
        """Convenience method to extract text from keyframe paths"""
        if not self.available:
            return {path: "" for path in keyframe_paths}
            
        ocr_results = {}
        print(f"Extracting OCR from {len(keyframe_paths)} keyframes...")
        
        for i, path in enumerate(keyframe_paths):
            try:
                text = self.encode([path])[0]
                ocr_results[path] = text
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(keyframe_paths)} keyframes")
            except Exception as e:
                print(f"Error processing {path}: {e}")
                ocr_results[path] = ""
                
        return ocr_results


# Alias for backward compatibility
OCREncoder = EasyOCREncoder
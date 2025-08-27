"""BLIP-2 encoder for image captioning (SOTA performance, easy implementation)"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
from pathlib import Path
import torch

from ..core.base import Encoder


class BLIPCaptioner(Encoder):
    """BLIP-2 encoder for generating image captions (HuggingFace integration)"""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", 
                 device: str = None, max_length: int = 50, config_dict: Dict[str, Any] = None):
        super().__init__("BLIPCaptioner", config_dict)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print(f"Loading BLIP model: {model_name} on {self.device}")
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.available = True
            print("BLIP-2 loaded successfully")
        except ImportError:
            print("Warning: transformers not installed. Run: pip install transformers")
            self.processor = None
            self.model = None
            self.available = False
        except Exception as e:
            print(f"Warning: Could not load BLIP model: {e}")
            self.processor = None
            self.model = None
            self.available = False
            
    def encode(self, data: Union[List[Image.Image], List[str], str, Image.Image]) -> List[str]:
        """Generate captions from images"""
        if not self.available:
            return ["" for _ in (data if isinstance(data, list) else [data])]
            
        # Convert to list format
        if isinstance(data, (str, Path)):
            images = [Image.open(str(data)).convert('RGB')]
        elif isinstance(data, Image.Image):
            images = [data]
        elif isinstance(data, list):
            images = []
            for item in data:
                if isinstance(item, (str, Path)):
                    images.append(Image.open(str(item)).convert('RGB'))
                elif isinstance(item, Image.Image):
                    images.append(item)
                else:
                    images.append(item)
        else:
            images = [data]
            
        captions = []
        
        try:
            # Process images in batches for efficiency
            batch_size = 4  # Adjust based on GPU memory
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                
                # Preprocess images
                inputs = self.processor(batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate captions
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False
                    )
                
                # Decode captions
                batch_captions = self.processor.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                captions.extend(batch_captions)
                
        except Exception as e:
            print(f"BLIP captioning error: {e}")
            captions = ["" for _ in images]
            
        return captions
        
    def process(self, images: Union[List[Union[Image.Image, str]], Union[Image.Image, str]]) -> List[Dict[str, Any]]:
        """Process images and return detailed captioning results"""
        if not self.available:
            return [{"caption": "", "confidence": 0.0, "model": self.model_name}]
            
        if not isinstance(images, list):
            images = [images]
            
        # Generate captions
        captions = self.encode(images)
        
        results = []
        for i, (image, caption) in enumerate(zip(images, captions)):
            try:
                # Get image info
                if isinstance(image, (str, Path)):
                    img_pil = Image.open(str(image)).convert('RGB')
                    image_path = str(image)
                else:
                    img_pil = image
                    image_path = None
                    
                results.append({
                    "caption": caption.strip(),
                    "model": self.model_name,
                    "image_path": image_path,
                    "image_size": img_pil.size,
                    "max_length": self.max_length,
                    "confidence": 1.0,  # BLIP doesn't provide confidence scores directly
                    "processing_device": self.device
                })
                
            except Exception as e:
                print(f"BLIP processing error for image {i}: {e}")
                results.append({
                    "caption": "",
                    "model": self.model_name,
                    "image_path": None,
                    "image_size": None,
                    "max_length": self.max_length,
                    "confidence": 0.0,
                    "processing_device": self.device
                })
                
        return results
        
    def caption_keyframes(self, keyframe_paths: List[str]) -> Dict[str, str]:
        """Convenience method to caption keyframe images"""
        if not self.available:
            return {path: "" for path in keyframe_paths}
            
        caption_results = {}
        print(f"Generating captions for {len(keyframe_paths)} keyframes...")
        
        # Process in batches for efficiency
        batch_size = 8
        for i in range(0, len(keyframe_paths), batch_size):
            batch_paths = keyframe_paths[i:i+batch_size]
            
            try:
                batch_captions = self.encode(batch_paths)
                for path, caption in zip(batch_paths, batch_captions):
                    caption_results[path] = caption.strip()
                    
                if (i + batch_size) % 50 == 0:
                    print(f"Processed {min(i + batch_size, len(keyframe_paths))}/{len(keyframe_paths)} keyframes")
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                for path in batch_paths:
                    caption_results[path] = ""
                    
        return caption_results
        
    def conditional_caption(self, images: Union[List[Image.Image], List[str]], 
                           prompts: Union[str, List[str]]) -> List[str]:
        """Generate conditional captions with prompts (BLIP-2 feature)"""
        if not self.available:
            return ["" for _ in (images if isinstance(images, list) else [images])]
            
        # Convert to list format
        if not isinstance(images, list):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
            
        # Convert paths to PIL Images
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(str(img)).convert('RGB'))
            else:
                pil_images.append(img)
                
        captions = []
        
        try:
            for image, prompt in zip(pil_images, prompts):
                # Use prompt as conditional text
                inputs = self.processor(image, text=prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
                captions.append(caption.strip())
                
        except Exception as e:
            print(f"BLIP conditional captioning error: {e}")
            captions = ["" for _ in images]
            
        return captions


# Alias for backward compatibility  
ImageCaptioner = BLIPCaptioner
MAGICEncoder = BLIPCaptioner
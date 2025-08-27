"""CLIP-based encoders for images and text"""

import torch
import numpy as np
import open_clip
from PIL import Image
from pathlib import Path
from typing import List, Union, Dict, Any
from tqdm import tqdm

from ..core.base import Encoder, VideoData
from utils import load_image, normalize_rows
import config


class CLIPImageEncoder(Encoder):
    """CLIP encoder for images/frames"""
    
    def __init__(self, model_name: str = None, pretrained: str = None, device: str = None, config_dict: Dict[str, Any] = None):
        super().__init__("CLIPImageEncoder", config_dict)
        
        self.model_name = model_name or config.MODEL_NAME
        self.pretrained = pretrained or config.MODEL_PRETRAINED
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        print(f"Loading CLIP model: {self.model_name} ({self.pretrained}) on {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.model.eval()
        
    def encode(self, data: Union[List[str], List[Image.Image], str, Image.Image]) -> np.ndarray:
        """Encode images to CLIP embeddings"""
        if isinstance(data, (str, Path)):
            data = [data]
        elif isinstance(data, Image.Image):
            data = [data]
            
        if isinstance(data[0], (str, Path)):
            # Load images from paths
            images = []
            for path in data:
                try:
                    img = load_image(path)
                    images.append(img)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    # Create a dummy image
                    images.append(Image.new('RGB', (224, 224), color='black'))
        else:
            images = data
            
        return self._encode_image_batch(images)
        
    def _encode_image_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Encode batch of PIL images"""
        embeddings = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            batch = images[i:i+batch_size]
            
            # Preprocess images
            batch_tensors = []
            for img in batch:
                tensor = self.preprocess(img).unsqueeze(0)
                batch_tensors.append(tensor)
                
            if batch_tensors:
                batch_tensor = torch.cat(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    if self.device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            batch_embeddings = self.model.encode_image(batch_tensor)
                    else:
                        batch_embeddings = self.model.encode_image(batch_tensor)
                        
                embeddings.append(batch_embeddings.cpu().numpy())
                
        if embeddings:
            all_embeddings = np.concatenate(embeddings, axis=0)
            return normalize_rows(all_embeddings)
        else:
            return np.array([])
            
    def process(self, video_data: VideoData) -> VideoData:
        """Process video data and add frame embeddings"""
        if not video_data.keyframes:
            return video_data
            
        # Get frame paths
        frame_paths = []
        for keyframe in video_data.keyframes:
            if "frame_path" in keyframe and keyframe["frame_path"]:
                frame_paths.append(keyframe["frame_path"])
            else:
                # Skip keyframes without saved images
                frame_paths.append(None)
                
        # Filter out None paths for encoding
        valid_paths = [p for p in frame_paths if p is not None]
        
        if valid_paths:
            embeddings = self.encode(valid_paths)
            video_data.embeddings = embeddings
        else:
            video_data.embeddings = np.array([])
            
        return video_data


class CLIPTextEncoder(Encoder):
    """CLIP encoder for text queries"""
    
    def __init__(self, model_name: str = None, pretrained: str = None, device: str = None, config_dict: Dict[str, Any] = None):
        super().__init__("CLIPTextEncoder", config_dict)
        
        self.model_name = model_name or config.MODEL_NAME
        self.pretrained = pretrained or config.MODEL_PRETRAINED
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        print(f"Loading CLIP model: {self.model_name} ({self.pretrained}) on {self.device}")
        self.model, _, _ = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()
        
    def encode(self, data: Union[str, List[str]]) -> np.ndarray:
        """Encode text to CLIP embeddings"""
        if isinstance(data, str):
            texts = [data]
        else:
            texts = data
            
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            if self.device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    embeddings = self.model.encode_text(tokens)
            else:
                embeddings = self.model.encode_text(tokens)
                
        embeddings = embeddings.float().cpu().numpy()
        return normalize_rows(embeddings)
        
    def process(self, query_text: str) -> np.ndarray:
        """Process single query text"""
        return self.encode(query_text)
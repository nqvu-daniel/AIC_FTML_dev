"""FastSAM encoder for fast object segmentation"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
from pathlib import Path
import torch

from ..core.base import Encoder


class FastSAMEncoder(Encoder):
    """FastSAM encoder for fast object segmentation (50x faster than SAM2)"""
    
    def __init__(self, model_name: str = "FastSAM-x.pt", device: str = None, config_dict: Dict[str, Any] = None):
        super().__init__("FastSAMEncoder", config_dict)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            from ultralytics import FastSAM
            print(f"Loading FastSAM model: {model_name} on {self.device}")
            self.model = FastSAM(model_name)
            self.available = True
        except ImportError:
            print("Warning: ultralytics not installed. Run: pip install ultralytics")
            self.model = None
            self.available = False
        except Exception as e:
            print(f"Warning: Could not load FastSAM model: {e}")
            self.model = None
            self.available = False
            
    def encode(self, data: Union[List[Image.Image], List[str], str, Image.Image]) -> np.ndarray:
        """Extract segmentation features from images"""
        if not self.available:
            # Return placeholder embeddings if model not available
            batch_size = len(data) if isinstance(data, list) else 1
            return np.random.randn(batch_size, 256)
            
        # Convert to format FastSAM expects
        if isinstance(data, (str, Path)):
            images = [str(data)]
        elif isinstance(data, Image.Image):
            images = [data]
        elif isinstance(data, list):
            images = []
            for item in data:
                if isinstance(item, (str, Path)):
                    images.append(str(item))
                elif isinstance(item, Image.Image):
                    images.append(item)
        else:
            images = data
            
        try:
            results = self.model(images, device=self.device)
            
            # Extract features from results
            features = []
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    # Use mask features as embeddings
                    mask_features = result.masks.data.cpu().numpy()
                    if len(mask_features.shape) > 2:
                        # Flatten and pool masks to create feature vector
                        feature_vector = np.mean(mask_features.reshape(mask_features.shape[0], -1), axis=0)
                    else:
                        feature_vector = mask_features.flatten()
                        
                    # Ensure consistent feature dimension
                    if len(feature_vector) > 256:
                        feature_vector = feature_vector[:256]
                    elif len(feature_vector) < 256:
                        feature_vector = np.pad(feature_vector, (0, 256 - len(feature_vector)))
                        
                    features.append(feature_vector)
                else:
                    # Fallback: create zero feature vector
                    features.append(np.zeros(256))
                    
            return np.array(features) if features else np.array([])
            
        except Exception as e:
            print(f"FastSAM encoding error: {e}")
            batch_size = len(images) if isinstance(images, list) else 1
            return np.random.randn(batch_size, 256)
        
    def process(self, images: Union[List[Image.Image], List[str]]) -> List[Dict[str, Any]]:
        """Process images and return detailed segmentation results"""
        if not self.available:
            return [{"masks": [], "boxes": [], "scores": [], "features": np.zeros(256)} 
                   for _ in range(len(images) if isinstance(images, list) else 1)]
            
        try:
            results = self.model(images, device=self.device)
            
            processed_results = []
            for result in results:
                masks = []
                boxes = []
                scores = []
                features = np.zeros(256)
                
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy().tolist()
                    # Create feature from masks
                    mask_data = result.masks.data.cpu().numpy()
                    if len(mask_data.shape) > 2:
                        features = np.mean(mask_data.reshape(mask_data.shape[0], -1), axis=0)
                        if len(features) > 256:
                            features = features[:256]
                        elif len(features) < 256:
                            features = np.pad(features, (0, 256 - len(features)))
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy().tolist()
                    scores = result.boxes.conf.cpu().numpy().tolist() if hasattr(result.boxes, 'conf') else []
                    
                processed_results.append({
                    "masks": masks,
                    "boxes": boxes, 
                    "scores": scores,
                    "features": features,
                    "num_masks": len(masks),
                    "image_shape": getattr(result, 'orig_shape', None)
                })
                
            return processed_results
            
        except Exception as e:
            print(f"FastSAM processing error: {e}")
            return [{"masks": [], "boxes": [], "scores": [], "features": np.zeros(256)}
                   for _ in range(len(images) if isinstance(images, list) else 1)]


# Alias for backward compatibility
SAM2Encoder = FastSAMEncoder
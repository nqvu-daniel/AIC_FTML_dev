"""Base classes for pipeline components"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pathlib import Path


class PipelineComponent(ABC):
    """Base class for all pipeline components"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return output"""
        pass
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class DataProcessor(PipelineComponent):
    """Base class for data preprocessing components"""
    pass


class QueryProcessor(PipelineComponent):
    """Base class for query processing components"""
    pass


class Encoder(PipelineComponent):
    """Base class for encoders (CLIP, OCR, etc.)"""
    
    @abstractmethod
    def encode(self, data: Any) -> np.ndarray:
        """Encode data to embeddings/features"""
        pass


class Index(PipelineComponent):
    """Base class for search indexes"""
    
    @abstractmethod
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add vectors to index with metadata"""
        pass
        
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10) -> tuple:
        """Search index and return (distances, indices)"""
        pass


class VideoData:
    """Standardized video data container"""
    
    def __init__(self, video_id: str, video_path: Path, metadata: Dict[str, Any] = None):
        self.video_id = video_id
        self.video_path = Path(video_path)
        self.metadata = metadata or {}
        
        # Extracted data
        self.keyframes: List[Dict[str, Any]] = []
        self.descriptions: List[str] = []
        self.ocr_texts: List[str] = []
        self.captions: List[str] = []  # Image captions from BLIP
        self.embeddings: Optional[np.ndarray] = None
        self.segmentation_features: Optional[np.ndarray] = None  # FastSAM features


class QueryData:
    """Standardized query data container"""
    
    def __init__(self, query: str, query_type: str = "text", metadata: Dict[str, Any] = None):
        self.query = query
        self.query_type = query_type  # "text", "image", "multimodal"
        self.metadata = metadata or {}
        
        # Processed data
        self.text_embedding: Optional[np.ndarray] = None
        self.image_embedding: Optional[np.ndarray] = None


class SearchResult:
    """Standardized search result container"""
    
    def __init__(self, video_id: str, frame_idx: int, score: float, metadata: Dict[str, Any] = None):
        self.video_id = video_id
        self.frame_idx = frame_idx
        self.score = score
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "score": self.score,
            **self.metadata
        }
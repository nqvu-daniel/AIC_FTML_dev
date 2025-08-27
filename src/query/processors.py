"""Query processing components"""

from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image

from ..core.base import QueryProcessor, QueryData, SearchResult
from ..encoders.clip_encoder import CLIPTextEncoder, CLIPImageEncoder


class TextQueryProcessor(QueryProcessor):
    """Process text queries"""
    
    def __init__(self, encoder: CLIPTextEncoder, config_dict: Dict[str, Any] = None):
        super().__init__("TextQueryProcessor", config_dict)
        self.encoder = encoder
        
    def process(self, query: str, metadata: Dict[str, Any] = None) -> QueryData:
        """Process text query into QueryData"""
        query_data = QueryData(query, "text", metadata)
        
        # Encode text to embedding
        query_data.text_embedding = self.encoder.process(query)
        
        return query_data


class ImageQueryProcessor(QueryProcessor):
    """Process image queries"""
    
    def __init__(self, encoder: CLIPImageEncoder, config_dict: Dict[str, Any] = None):
        super().__init__("ImageQueryProcessor", config_dict)
        self.encoder = encoder
        
    def process(self, image: Image.Image, metadata: Dict[str, Any] = None) -> QueryData:
        """Process image query into QueryData"""
        query_data = QueryData(str(image), "image", metadata)
        
        # Encode image to embedding
        query_data.image_embedding = self.encoder.encode([image])
        
        return query_data


class MultimodalQueryProcessor(QueryProcessor):
    """Process multimodal queries (text + image)"""
    
    def __init__(self, text_encoder: CLIPTextEncoder, image_encoder: CLIPImageEncoder, 
                 fusion_weight: float = 0.5, config_dict: Dict[str, Any] = None):
        super().__init__("MultimodalQueryProcessor", config_dict)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion_weight = fusion_weight
        
    def process(self, query: str, image: Optional[Image.Image] = None, 
                metadata: Dict[str, Any] = None) -> QueryData:
        """Process multimodal query"""
        query_data = QueryData(query, "multimodal", metadata)
        
        # Encode text
        if query:
            query_data.text_embedding = self.text_encoder.process(query)
            
        # Encode image
        if image:
            query_data.image_embedding = self.image_encoder.encode([image])
            
        return query_data


class QueryExpander(QueryProcessor):
    """Expand queries with synonyms, related terms, etc."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        super().__init__("QueryExpander", config_dict)
        
        # Common query expansions for video retrieval
        self.expansions = {
            "person": ["people", "human", "individual", "man", "woman"],
            "object": ["item", "thing", "element"],
            "action": ["activity", "movement", "motion", "behavior"],
            "indoor": ["inside", "interior", "room"],
            "outdoor": ["outside", "exterior", "street", "nature"],
        }
        
    def process(self, query: str) -> str:
        """Expand query with related terms"""
        words = query.lower().split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            if word in self.expansions:
                expanded_words.extend(self.expansions[word][:2])  # Add top 2 expansions
                
        return " ".join(expanded_words)
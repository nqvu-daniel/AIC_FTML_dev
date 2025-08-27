"""Search engine combining vector and text search"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from ..core.base import QueryProcessor, SearchResult, QueryData
from ..indexing.vector_index import FAISSIndex
from ..indexing.text_index import BM25Index


class VectorSearchEngine(QueryProcessor):
    """Vector-based search engine"""
    
    def __init__(self, vector_index: FAISSIndex, config_dict: Dict[str, Any] = None):
        super().__init__("VectorSearchEngine", config_dict)
        self.vector_index = vector_index
        
    def process(self, query_data: QueryData, k: int = 100) -> List[SearchResult]:
        """Perform vector search"""
        if query_data.text_embedding is None and query_data.image_embedding is None:
            return []
            
        # Use text embedding if available, otherwise image embedding
        query_vector = query_data.text_embedding
        if query_vector is None:
            query_vector = query_data.image_embedding
            
        if query_vector is None:
            return []
            
        # Search vector index
        distances, indices = self.vector_index.search(query_vector, k)
        
        # Convert to SearchResult objects
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.vector_index.metadata):
                metadata = self.vector_index.metadata[idx]
                result = SearchResult(
                    video_id=metadata.get('video_id', ''),
                    frame_idx=metadata.get('frame_idx', 0),
                    score=float(distance),
                    metadata={**metadata, 'search_type': 'vector', 'rank': i}
                )
                results.append(result)
                
        return results


class TextSearchEngine(QueryProcessor):
    """Text-based search engine using BM25"""
    
    def __init__(self, text_index: BM25Index, config_dict: Dict[str, Any] = None):
        super().__init__("TextSearchEngine", config_dict)
        self.text_index = text_index
        
    def process(self, query_data: QueryData, k: int = 100) -> List[SearchResult]:
        """Perform text search"""
        if query_data.query_type != "text" and not query_data.query:
            return []
            
        # Search text index
        scores, indices = self.text_index.search(query_data.query, k)
        
        # Convert to SearchResult objects
        results = []
        for i, (score, idx) in enumerate(zip(scores, indices)):
            if idx < len(self.text_index.documents):
                doc = self.text_index.documents[idx]
                result = SearchResult(
                    video_id=doc.get('video_id', ''),
                    frame_idx=doc.get('frame_idx', 0),
                    score=float(score),
                    metadata={**doc, 'search_type': 'text', 'rank': i}
                )
                results.append(result)
                
        return results


class HybridSearchEngine(QueryProcessor):
    """Hybrid search engine combining vector and text search"""
    
    def __init__(self, vector_engine: VectorSearchEngine, text_engine: TextSearchEngine, 
                 config_dict: Dict[str, Any] = None):
        super().__init__("HybridSearchEngine", config_dict)
        self.vector_engine = vector_engine
        self.text_engine = text_engine
        
    def process(self, query_data: QueryData, k: int = 100, 
                vector_weight: float = 0.6, text_weight: float = 0.4) -> List[SearchResult]:
        """Perform hybrid search with both vector and text"""
        
        # Get results from both engines
        vector_results = self.vector_engine.process(query_data, k * 2)  # Over-fetch for fusion
        text_results = self.text_engine.process(query_data, k * 2)
        
        # Combine using Reciprocal Rank Fusion (RRF)
        return self._reciprocal_rank_fusion(vector_results, text_results, k, vector_weight, text_weight)
        
    def _reciprocal_rank_fusion(self, vector_results: List[SearchResult], text_results: List[SearchResult],
                               k: int, vector_weight: float, text_weight: float) -> List[SearchResult]:
        """Combine results using Reciprocal Rank Fusion"""
        
        # Create score dictionaries keyed by (video_id, frame_idx)
        vector_scores = {}
        text_scores = {}
        all_items = set()
        
        # Process vector results
        for i, result in enumerate(vector_results):
            key = (result.video_id, result.frame_idx)
            vector_scores[key] = {
                'rrf_score': vector_weight / (60 + i + 1),  # RRF with k=60
                'original_score': result.score,
                'rank': i,
                'result': result
            }
            all_items.add(key)
            
        # Process text results
        for i, result in enumerate(text_results):
            key = (result.video_id, result.frame_idx)
            text_scores[key] = {
                'rrf_score': text_weight / (60 + i + 1),  # RRF with k=60
                'original_score': result.score,
                'rank': i,
                'result': result
            }
            all_items.add(key)
            
        # Combine scores
        combined_results = []
        for key in all_items:
            vector_score = vector_scores.get(key, {'rrf_score': 0, 'original_score': 0, 'rank': float('inf'), 'result': None})
            text_score = text_scores.get(key, {'rrf_score': 0, 'original_score': 0, 'rank': float('inf'), 'result': None})
            
            # Combined RRF score
            combined_rrf_score = vector_score['rrf_score'] + text_score['rrf_score']
            
            # Use the result with higher original score, or vector if tied
            if vector_score['result'] and text_score['result']:
                if vector_score['original_score'] >= text_score['original_score']:
                    base_result = vector_score['result']
                else:
                    base_result = text_score['result']
            elif vector_score['result']:
                base_result = vector_score['result']
            elif text_score['result']:
                base_result = text_score['result']
            else:
                continue
                
            # Create combined result
            combined_result = SearchResult(
                video_id=base_result.video_id,
                frame_idx=base_result.frame_idx,
                score=combined_rrf_score,
                metadata={
                    **base_result.metadata,
                    'search_type': 'hybrid',
                    'vector_score': vector_score['original_score'],
                    'text_score': text_score['original_score'],
                    'vector_rank': vector_score['rank'],
                    'text_rank': text_score['rank'],
                    'rrf_score': combined_rrf_score
                }
            )
            combined_results.append(combined_result)
            
        # Sort by combined score and return top k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:k]
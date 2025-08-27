"""Result reranking and post-processing"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from ..core.base import QueryProcessor, SearchResult


class TemporalDeduplicator(QueryProcessor):
    """Remove temporally close duplicates from the same video"""
    
    def __init__(self, radius: int = 1, config_dict: Dict[str, Any] = None):
        super().__init__("TemporalDeduplicator", config_dict)
        self.radius = radius
        
    def process(self, results: List[SearchResult]) -> List[SearchResult]:
        """Deduplicate temporally close results"""
        # Convert to DataFrame for easier processing
        result_data = []
        for result in results:
            result_data.append({
                'video_id': result.video_id,
                'frame_idx': result.frame_idx,
                'score': result.score,
                'result': result
            })
            
        df = pd.DataFrame(result_data)
        if df.empty:
            return []
            
        # Sort by video and frame index
        df = df.sort_values(['video_id', 'frame_idx']).reset_index(drop=True)
        
        # Deduplicate within each video
        kept_results = []
        last_frame_by_video = {}
        
        for _, row in df.sort_values('score', ascending=False).iterrows():
            video_id = row['video_id']
            frame_idx = row['frame_idx']
            
            # Check if too close to previously selected frame in same video
            if video_id in last_frame_by_video:
                if abs(frame_idx - last_frame_by_video[video_id]) <= self.radius:
                    continue  # Skip this frame
                    
            kept_results.append(row['result'])
            last_frame_by_video[video_id] = frame_idx
            
        return kept_results


class ScoreNormalizer(QueryProcessor):
    """Normalize scores across different search methods"""
    
    def __init__(self, method: str = "minmax", config_dict: Dict[str, Any] = None):
        super().__init__("ScoreNormalizer", config_dict)
        self.method = method
        
    def process(self, results: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores"""
        if not results:
            return results
            
        scores = np.array([r.score for r in results])
        
        if self.method == "minmax":
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = scores
        elif self.method == "zscore":
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            if std_score > 0:
                normalized_scores = (scores - mean_score) / std_score
            else:
                normalized_scores = scores
        else:
            normalized_scores = scores
            
        # Update results with normalized scores
        normalized_results = []
        for result, norm_score in zip(results, normalized_scores):
            new_result = SearchResult(
                video_id=result.video_id,
                frame_idx=result.frame_idx,
                score=float(norm_score),
                metadata={**result.metadata, 'original_score': result.score}
            )
            normalized_results.append(new_result)
            
        return normalized_results


class DiversityReranker(QueryProcessor):
    """Rerank results to promote diversity"""
    
    def __init__(self, diversity_weight: float = 0.3, config_dict: Dict[str, Any] = None):
        super().__init__("DiversityReranker", config_dict)
        self.diversity_weight = diversity_weight
        
    def process(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results considering diversity"""
        if len(results) <= 1:
            return results
            
        # Simple diversity: prefer results from different videos
        video_counts = {}
        reranked_results = []
        
        for result in results:
            video_id = result.video_id
            video_count = video_counts.get(video_id, 0)
            
            # Apply diversity penalty
            diversity_penalty = video_count * self.diversity_weight
            adjusted_score = result.score * (1 - diversity_penalty)
            
            new_result = SearchResult(
                video_id=result.video_id,
                frame_idx=result.frame_idx,
                score=adjusted_score,
                metadata={
                    **result.metadata,
                    'diversity_penalty': diversity_penalty,
                    'original_score': result.score
                }
            )
            reranked_results.append(new_result)
            
            video_counts[video_id] = video_count + 1
            
        # Resort by adjusted scores
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results


class MultiFrameContextReranker(QueryProcessor):
    """Rerank considering multi-frame context"""
    
    def __init__(self, context_window: int = 3, context_weight: float = 0.2, config_dict: Dict[str, Any] = None):
        super().__init__("MultiFrameContextReranker", config_dict)
        self.context_window = context_window
        self.context_weight = context_weight
        
    def process(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank using neighboring frame context"""
        if len(results) <= 1:
            return results
            
        # Group results by video
        video_groups = {}
        for result in results:
            video_id = result.video_id
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(result)
            
        # Rerank within each video group
        reranked_results = []
        for video_id, video_results in video_groups.items():
            # Sort by frame index
            video_results.sort(key=lambda x: x.frame_idx)
            
            # Calculate context scores
            for i, result in enumerate(video_results):
                # Find neighbors
                neighbors = []
                for j in range(max(0, i - self.context_window), 
                              min(len(video_results), i + self.context_window + 1)):
                    if j != i:
                        neighbors.append(video_results[j])
                        
                # Calculate average neighbor score
                if neighbors:
                    neighbor_score = np.mean([n.score for n in neighbors])
                    context_boost = neighbor_score * self.context_weight
                else:
                    context_boost = 0
                    
                # Apply context boost
                boosted_score = result.score + context_boost
                
                new_result = SearchResult(
                    video_id=result.video_id,
                    frame_idx=result.frame_idx,
                    score=boosted_score,
                    metadata={
                        **result.metadata,
                        'context_boost': context_boost,
                        'original_score': result.score,
                        'neighbor_count': len(neighbors)
                    }
                )
                reranked_results.append(new_result)
                
        # Sort all results by final score
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results
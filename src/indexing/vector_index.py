"""Vector indexing using FAISS"""

import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

from ..core.base import Index

# Import utilities from root directory
import os
import sys
root_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, root_path)

# Import utils from root utils.py file
import importlib.util
utils_spec = importlib.util.spec_from_file_location("utils", os.path.join(root_path, "utils.py"))
utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(utils)

save_faiss = utils.save_faiss
load_faiss = utils.load_faiss
to_parquet = utils.to_parquet
from_parquet = utils.from_parquet
normalize_rows = utils.normalize_rows


class FAISSIndex(Index):
    """FAISS-based vector index"""
    
    def __init__(self, dimension: int, use_flat: bool = False, config_dict: Dict[str, Any] = None):
        super().__init__("FAISSIndex", config_dict)
        self.dimension = dimension
        self.use_flat = use_flat
        self.index = self._create_index()
        self.metadata: List[Dict[str, Any]] = []
        
    def _create_index(self) -> faiss.Index:
        """Create FAISS index"""
        if self.use_flat:
            # Exact search
            index = faiss.IndexFlatIP(self.dimension)
            print(f"Created exact IndexFlatIP (dim={self.dimension})")
        else:
            # HNSW for faster approximate search
            index = faiss.IndexHNSWFlat(self.dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 100
            print(f"Created HNSW index (dim={self.dimension})")
            
        return index
        
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add vectors to index with metadata"""
        if len(vectors) == 0:
            return
            
        # Normalize vectors for cosine similarity
        vectors = normalize_rows(vectors)
        
        # Add to FAISS index
        self.index.add(vectors.astype('float32'))
        
        # Store metadata
        self.metadata.extend(metadata)
        
        print(f"Added {len(vectors)} vectors to index (total: {self.index.ntotal})")
        
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search index and return (distances, indices)"""
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # Normalize query
        query_vector = normalize_rows(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        return distances, indices
        
    def save(self, index_path: Path, mapping_path: Path):
        """Save index and metadata to disk"""
        save_faiss(self.index, index_path)
        
        # Convert metadata to DataFrame and save
        if self.metadata:
            df = pd.DataFrame(self.metadata)
            to_parquet(df, mapping_path)
            
        print(f"Saved index: {index_path}")
        print(f"Saved mapping: {mapping_path}")
        
    def load(self, index_path: Path, mapping_path: Path):
        """Load index and metadata from disk"""
        self.index = load_faiss(index_path)
        
        if mapping_path.exists():
            df = from_parquet(mapping_path)
            self.metadata = df.to_dict('records')
            
        print(f"Loaded index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.metadata)} metadata entries")
        
    def process(self, input_data: Any) -> Any:
        """Process method for pipeline compatibility"""
        # This is mainly used for building the index from processed data
        return input_data


class VideoSemanticIndex(FAISSIndex):
    """Specialized index for video-level semantic search"""
    
    def __init__(self, dimension: int, window: int = 8, stride: int = 4, pooling: str = "mean", use_flat: bool = False):
        super().__init__(dimension, use_flat)
        self.window = window
        self.stride = stride
        self.pooling = pooling
        
    def build_from_frames(self, frame_embeddings: np.ndarray, frame_mapping: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Build video-level segments from frame embeddings"""
        segments = []
        segment_metadata = []
        
        # Group by video
        for video_id in frame_mapping['video_id'].unique():
            video_frames = frame_mapping[frame_mapping['video_id'] == video_id].sort_values('timestamp')
            video_embeddings = frame_embeddings[video_frames.index.values]
            
            # Create sliding windows
            for i in range(0, len(video_embeddings) - self.window + 1, self.stride):
                window_embeddings = video_embeddings[i:i+self.window]
                
                # Pool embeddings
                if self.pooling == "mean":
                    segment_embedding = np.mean(window_embeddings, axis=0)
                elif self.pooling == "max":
                    segment_embedding = np.max(window_embeddings, axis=0)
                elif self.pooling == "meanmax":
                    mean_emb = np.mean(window_embeddings, axis=0)
                    max_emb = np.max(window_embeddings, axis=0)
                    segment_embedding = np.concatenate([mean_emb, max_emb])
                else:
                    segment_embedding = np.mean(window_embeddings, axis=0)
                    
                segments.append(segment_embedding)
                
                # Metadata for this segment
                window_frames = video_frames.iloc[i:i+self.window]
                segment_metadata.append({
                    'video_id': video_id,
                    'start_frame': int(window_frames.iloc[0]['frame_idx']),
                    'end_frame': int(window_frames.iloc[-1]['frame_idx']),
                    'start_timestamp': float(window_frames.iloc[0]['timestamp']),
                    'end_timestamp': float(window_frames.iloc[-1]['timestamp']),
                    'segment_id': f"{video_id}_{i//self.stride:04d}"
                })
                
        if segments:
            segment_vectors = np.stack(segments)
            segment_df = pd.DataFrame(segment_metadata)
            return segment_vectors, segment_df
        else:
            return np.array([]), pd.DataFrame()
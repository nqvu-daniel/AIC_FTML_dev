"""Query processing pipeline orchestrator"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from ..core.base import SearchResult, QueryData
from ..query.processors import TextQueryProcessor, ImageQueryProcessor, QueryExpander
from ..query.search_engine import VectorSearchEngine, TextSearchEngine, HybridSearchEngine
from ..fusion.reranker import TemporalDeduplicator, ScoreNormalizer, DiversityReranker, MultiFrameContextReranker
from ..encoders.clip_encoder import CLIPTextEncoder, CLIPImageEncoder
from ..indexing.vector_index import FAISSIndex
from ..indexing.text_index import BM25Index
import config


class QueryProcessingPipeline:
    """Orchestrates the complete query processing pipeline"""
    
    def __init__(self, 
                 artifact_dir: Path,
                 model_name: str = None,
                 pretrained: str = None,
                 enable_reranking: bool = True,
                 enable_deduplication: bool = True):
        
        self.artifact_dir = Path(artifact_dir)
        self.enable_reranking = enable_reranking
        self.enable_deduplication = enable_deduplication
        
        # Initialize encoders
        self.text_encoder = CLIPTextEncoder(
            model_name=model_name or config.MODEL_NAME,
            pretrained=pretrained or config.MODEL_PRETRAINED
        )
        self.image_encoder = CLIPImageEncoder(
            model_name=model_name or config.MODEL_NAME, 
            pretrained=pretrained or config.MODEL_PRETRAINED
        )
        
        # Initialize query processors
        self.text_query_processor = TextQueryProcessor(self.text_encoder)
        self.image_query_processor = ImageQueryProcessor(self.image_encoder)
        self.query_expander = QueryExpander()
        
        # Load indexes
        self._load_indexes()
        
        # Initialize search engines
        self.vector_engine = VectorSearchEngine(self.vector_index)
        self.text_engine = TextSearchEngine(self.text_index)
        self.hybrid_engine = HybridSearchEngine(self.vector_engine, self.text_engine)

        # Initialize rerankers
        if self.enable_reranking:
            self.temporal_dedup = TemporalDeduplicator(radius=1)
            self.score_normalizer = ScoreNormalizer(method="minmax")
            self.diversity_reranker = DiversityReranker(diversity_weight=0.2)
            self.context_reranker = MultiFrameContextReranker(context_window=3, context_weight=0.1)

        # Optional ML reranker (trained model)
        self.ml_reranker = None
        try:
            import joblib  # noqa: F401
            model_path = self.artifact_dir / "reranker.joblib"
            if model_path.exists():
                from joblib import load
                self.ml_reranker = load(model_path)
                print(f"Loaded ML reranker: {model_path}")
        except Exception as e:
            print(f"Warning: could not load ML reranker: {e}")
        
    def _load_indexes(self):
        """Load pre-built indexes"""
        print("Loading indexes...")
        
        # Load vector index
        index_path = self.artifact_dir / "index.faiss"
        mapping_path = self.artifact_dir / "mapping.parquet"
        
        if index_path.exists() and mapping_path.exists():
            # Determine dimension from mapping
            mapping_df = pd.read_parquet(mapping_path)
            # We need to infer dimension - use config default
            embedding_dim = 512  # Default, should be read from pipeline_info.json
            
            # Try to read actual dimension from pipeline info
            info_path = self.artifact_dir / "pipeline_info.json"
            if info_path.exists():
                import json
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    embedding_dim = info.get('embedding_dim', 512)
            
            self.vector_index = FAISSIndex(embedding_dim)
            self.vector_index.load(index_path, mapping_path)
        else:
            raise FileNotFoundError(f"Vector index files not found: {index_path}, {mapping_path}")
            
        # Load text index
        bm25_path = self.artifact_dir / "bm25_index.json"
        if bm25_path.exists():
            self.text_index = BM25Index()
            self.text_index.load(bm25_path)
        else:
            print(f"Warning: BM25 index not found: {bm25_path}")
            self.text_index = BM25Index()  # Empty index
            
        print("Indexes loaded successfully")
        
    def search(self, 
               query: str,
               search_mode: str = "hybrid",  # "vector", "text", "hybrid"
               k: int = 100,
                vector_weight: float = 0.6,
                text_weight: float = 0.4,
                expand_query: bool = False,
                enable_reranking: bool = None) -> List[SearchResult]:
        """Process query and return search results"""
        
        if enable_reranking is None:
            enable_reranking = self.enable_reranking
            
        print(f"\n=== Query Processing Pipeline ===")
        print(f"Query: '{query}'")
        print(f"Search mode: {search_mode}")
        print(f"Top-k: {k}")
        
        # Step 1: Query Processing
        if expand_query:
            expanded_query = self.query_expander.process(query)
            print(f"Expanded query: '{expanded_query}'")
            query = expanded_query
            
        query_data = self.text_query_processor.process(query)
        
        # Step 2: Search
        print(f"\n--- Step 2: Search ({search_mode}) ---")
        
        overfetch = k * 2
        if search_mode == "vector":
            results = self.vector_engine.process(query_data, overfetch)
        elif search_mode == "text":
            results = self.text_engine.process(query_data, overfetch)
        elif search_mode == "hybrid":
            results = self.hybrid_engine.process(query_data, overfetch, vector_weight, text_weight)
        else:
            raise ValueError(f"Unknown search mode: {search_mode}")
            
        print(f"Retrieved {len(results)} initial results")

        # Step 3: Reranking and Post-processing
        if enable_reranking and results:
            print(f"\n--- Step 3: Reranking ---")

            # Apply optional ML reranker first if available
            if self.ml_reranker is not None:
                try:
                    results = self._apply_ml_reranker(query, query_data, results, overfetch)
                    print("Applied ML reranker")
                except Exception as e:
                    print(f"Warning: ML reranker failed: {e}")
            
            # Temporal deduplication
            if self.enable_deduplication:
                results = self.temporal_dedup.process(results)
                print(f"After deduplication: {len(results)} results")
            
            # Score normalization
            results = self.score_normalizer.process(results)
            
            # Diversity reranking
            results = self.diversity_reranker.process(results)
            print(f"Applied diversity reranking")
            
            # Multi-frame context reranking
            results = self.context_reranker.process(results)
            print(f"Applied context reranking")
        
        # Step 4: Final filtering
        final_results = results[:k]
        
        print(f"\n✅ Query processing complete: {len(final_results)} final results")
        return final_results

    def _apply_ml_reranker(self, query: str, query_data: QueryData, results: List[SearchResult], overfetch: int) -> List[SearchResult]:
        """Re-score results using trained ML reranker if present"""
        if self.ml_reranker is None:
            return results

        # Build lookup tables for vector/text ranks and scores for a consistent feature set
        vec_lookup: Dict[tuple, tuple] = {}
        txt_lookup: Dict[tuple, tuple] = {}

        # Always compute both modalities for feature completeness
        vec_candidates = self.vector_engine.process(query_data, overfetch)
        for r, item in enumerate(vec_candidates):
            vec_lookup[(item.video_id, item.frame_idx)] = (r, float(item.score))

        txt_candidates = self.text_engine.process(query_data, overfetch)
        for r, item in enumerate(txt_candidates):
            txt_lookup[(item.video_id, item.frame_idx)] = (r, float(item.score))

        # Build features for current result set
        X: List[List[float]] = []
        keys: List[tuple] = []
        for res in results:
            key = (res.video_id, int(res.frame_idx))
            vrank, vscore = vec_lookup.get(key, (None, None))
            trank, tscore = txt_lookup.get(key, (None, None))
            # Feature ordering must match the training script
            vr = float(vrank if vrank is not None else 1e6)
            tr = float(trank if trank is not None else 1e6)
            vs = float(vscore if vscore is not None else 0.0)
            ts = float(tscore if tscore is not None else 0.0)
            both = 1.0 if (vrank is not None and trank is not None) else 0.0
            rrf = (1.0 / (60.0 + (vrank if vrank is not None else 1e6))) + \
                  (1.0 / (60.0 + (trank if trank is not None else 1e6)))
            X.append([vs, ts, vr, tr, both, rrf])
            keys.append(key)

        # Predict scores (use predict_proba if available)
        try:
            if hasattr(self.ml_reranker, "predict_proba"):
                scores = self.ml_reranker.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
            else:
                scores = self.ml_reranker.decision_function(np.asarray(X, dtype=np.float32))
        except Exception as e:
            print(f"ML reranker scoring error: {e}")
            return results

        # Attach ML score and sort
        rescored: List[SearchResult] = []
        for res, s in zip(results, scores):
            rescored.append(
                SearchResult(
                    video_id=res.video_id,
                    frame_idx=res.frame_idx,
                    score=float(s),
                    metadata={**res.metadata, "ml_score": float(s)}
                )
            )

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored
        
    def export_results(self, results: List[SearchResult], output_path: Path, 
                      format: str = "csv", include_metadata: bool = False) -> Path:
        """Export search results to file"""
        
        if not results:
            print("No results to export")
            return output_path
            
        # Convert results to DataFrame
        result_data = []
        for i, result in enumerate(results):
            row = {
                'rank': i + 1,
                'video_id': result.video_id,
                'frame_idx': result.frame_idx,
                'score': result.score
            }
            
            if include_metadata:
                row.update(result.metadata)
                
            result_data.append(row)
            
        df = pd.DataFrame(result_data)
        
        # Save in requested format
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient='records', indent=2)
        elif format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"Results exported to: {output_path}")
        return output_path
        
    def batch_search(self, queries: List[str], **kwargs) -> Dict[str, List[SearchResult]]:
        """Process multiple queries"""
        results = {}
        
        for query in queries:
            print(f"\nProcessing query: {query}")
            query_results = self.search(query, **kwargs)
            results[query] = query_results
            
        return results

"""Data preprocessing pipeline orchestrator"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pandas as pd
import numpy as np

from ..core.base import VideoData
from ..preprocessing.video_processor import VideoProcessor, KeyframeExtractor, KeyframeSaver
from ..preprocessing.transnet_processor import TransNetKeyframeExtractor
from ..preprocessing.text_processor import MetadataProcessor, TextCorpusBuilder, OCRProcessor, CaptionProcessor
from ..encoders.clip_encoder import CLIPImageEncoder
from ..indexing.vector_index import FAISSIndex, VideoSemanticIndex
from ..indexing.text_index import BM25Index
import config


class DataPreprocessingPipeline:
    """Orchestrates the complete data preprocessing pipeline"""
    
    def __init__(self, 
                 output_dir: Path,
                 artifact_dir: Path,
                 target_frames: int = 50,
                 batch_size: int = 32,
                 use_flat: bool = False,
                 model_name: str = None,
                 pretrained: str = None,
                 enable_ocr: bool = True,
                 enable_captions: bool = True,
                 enable_segmentation: bool = False,
                 use_transnet: bool = True):
        
        self.output_dir = Path(output_dir)
        self.artifact_dir = Path(artifact_dir)
        self.target_frames = target_frames
        self.batch_size = batch_size
        self.use_flat = use_flat
        self.enable_ocr = enable_ocr
        self.enable_captions = enable_captions
        self.enable_segmentation = enable_segmentation
        self.use_transnet = use_transnet
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.keyframes_dir = self.output_dir / "keyframes"
        
        # Initialize components
        self.video_processor = VideoProcessor()
        
        # Choose keyframe extractor: TransNet-V2 for academic excellence or basic intelligent sampling
        if self.use_transnet:
            print("🎬 Using TransNet-V2 for academic-grade shot boundary detection")
            self.keyframe_extractor = TransNetKeyframeExtractor(target_frames=target_frames)
        else:
            print("🧠 Using intelligent sampling approach")
            self.keyframe_extractor = KeyframeExtractor(target_frames=target_frames)
            
        self.keyframe_saver = KeyframeSaver(self.keyframes_dir)
        self.metadata_processor = MetadataProcessor()
        self.corpus_builder = TextCorpusBuilder()
        
        # Initialize optional processors
        if self.enable_ocr:
            self.ocr_processor = OCRProcessor()
        else:
            self.ocr_processor = None
            
        if self.enable_captions:
            self.caption_processor = CaptionProcessor()
        else:
            self.caption_processor = None
        
        # Initialize CLIP encoder
        self.clip_encoder = CLIPImageEncoder(
            model_name=model_name or config.MODEL_NAME,
            pretrained=pretrained or config.MODEL_PRETRAINED
        )
        
        # Will be initialized after we know embedding dimension
        self.vector_index: Optional[FAISSIndex] = None
        self.text_index = BM25Index()
        self.video_index: Optional[VideoSemanticIndex] = None
        
    def process_videos(self, video_paths: List[Path]) -> Dict[str, Any]:
        """Process all videos through the complete pipeline"""
        
        print(f"\n=== Data Preprocessing Pipeline ===")
        print(f"Processing {len(video_paths)} videos")
        print(f"Target frames per video: {self.target_frames}")
        print(f"Output directory: {self.output_dir}")
        print(f"Artifact directory: {self.artifact_dir}")
        
        # Step 1: Video Processing and Keyframe Extraction
        video_data_list = []
        all_embeddings = []
        embedding_metadata = []
        
        print(f"\n--- Step 1: Video Processing and Keyframe Extraction ---")
        
        for video_path in tqdm(video_paths, desc="Processing videos"):
            try:
                # Process video
                video_data = self.video_processor.process(video_path)
                
                # Extract keyframes
                video_data = self.keyframe_extractor.process(video_data)
                
                if not video_data.keyframes:
                    print(f"Warning: No keyframes extracted from {video_path.name}")
                    continue
                    
                # Save keyframes
                video_data = self.keyframe_saver.process(video_data)
                
                video_data_list.append(video_data)
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
                
        if not video_data_list:
            raise ValueError("No videos were successfully processed")
            
        print(f"Successfully processed {len(video_data_list)} videos")
        
        # Step 2: Feature Extraction
        print(f"\n--- Step 2: Feature Extraction ---")
        
        for video_data in tqdm(video_data_list, desc="Extracting features"):
            try:
                # CLIP encoding
                video_data = self.clip_encoder.process(video_data)
                
                if video_data.embeddings is not None and len(video_data.embeddings) > 0:
                    all_embeddings.append(video_data.embeddings)
                    
                    # Create metadata for each embedding
                    for i, keyframe in enumerate(video_data.keyframes):
                        if i < len(video_data.embeddings):
                            embedding_metadata.append({
                                'global_idx': len(embedding_metadata),
                                'video_id': video_data.video_id,
                                'frame_idx': keyframe['frame_idx'],
                                'timestamp': keyframe['timestamp'],
                                'frame_path': keyframe.get('frame_path', ''),
                                'n': i  # Frame number within video
                            })
                
                # Text processing
                video_data = self.metadata_processor.process(video_data)
                
                # OCR processing (optional)
                if self.enable_ocr and self.ocr_processor:
                    print(f"  OCR processing for {video_data.video_id}...")
                    video_data = self.ocr_processor.process(video_data)
                
                # Image captioning (optional)
                if self.enable_captions and self.caption_processor:
                    print(f"  Generating captions for {video_data.video_id}...")
                    video_data = self.caption_processor.process(video_data)
                
            except Exception as e:
                print(f"Error extracting features from {video_data.video_id}: {e}")
                continue
        
        # Combine all embeddings
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
        else:
            combined_embeddings = np.array([])
            
        print(f"Extracted {len(combined_embeddings)} total embeddings")
        
        # Step 3: Text Corpus Building
        print(f"\n--- Step 3: Text Corpus Building ---")
        
        corpus_entries = self.corpus_builder.process(video_data_list)
        print(f"Built text corpus with {len(corpus_entries)} entries")
        
        # Step 4: Index Building
        print(f"\n--- Step 4: Index Building ---")
        
        # Initialize vector index with correct dimension
        if len(combined_embeddings) > 0:
            embedding_dim = combined_embeddings.shape[1]
            self.vector_index = FAISSIndex(embedding_dim, use_flat=self.use_flat)
            
            # Add embeddings to vector index
            self.vector_index.add(combined_embeddings, embedding_metadata)
        
        # Build text index
        self.text_index.process(corpus_entries)
        
        # Step 5: Save Artifacts
        print(f"\n--- Step 5: Saving Artifacts ---")
        
        self._save_artifacts(video_data_list, combined_embeddings, embedding_metadata, corpus_entries)
        
        # Return summary
        return {
            'total_videos': len(video_data_list),
            'total_keyframes': len(embedding_metadata),
            'total_embeddings': len(combined_embeddings),
            'embedding_dimension': combined_embeddings.shape[1] if len(combined_embeddings) > 0 else 0,
            'corpus_size': len(corpus_entries),
            'model_name': self.clip_encoder.model_name,
            'model_pretrained': self.clip_encoder.pretrained
        }
        
    def _save_artifacts(self, video_data_list: List[VideoData], embeddings: np.ndarray, 
                       metadata: List[Dict[str, Any]], corpus: List[Dict[str, Any]]):
        """Save all pipeline artifacts"""
        
        # Save vector index and mapping
        if self.vector_index is not None:
            index_path = self.artifact_dir / "index.faiss" 
            mapping_path = self.artifact_dir / "mapping.parquet"
            self.vector_index.save(index_path, mapping_path)
        
        # Save text corpus
        import json
        corpus_path = self.artifact_dir / "text_corpus.jsonl"
        with open(corpus_path, 'w') as f:
            for entry in corpus:
                f.write(json.dumps(entry) + '\n')
        print(f"Saved text corpus: {corpus_path}")
        
        # Save BM25 index
        bm25_path = self.artifact_dir / "bm25_index.json"
        self.text_index.save(bm25_path)
        
        # Save pipeline info
        info = {
            "model_name": self.clip_encoder.model_name,
            "model_pretrained": self.clip_encoder.pretrained,
            "embedding_dim": int(embeddings.shape[1]) if len(embeddings) > 0 else 0,
            "total_keyframes": len(metadata),
            "total_videos": len(video_data_list),
            "target_frames_per_video": self.target_frames,
            "index_type": "flat" if self.use_flat else "hnsw",
            "sampling_method": "intelligent"
        }
        
        info_path = self.artifact_dir / "pipeline_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Saved pipeline info: {info_path}")
        
        print(f"\n✅ All artifacts saved to: {self.artifact_dir}")
        
    def build_video_semantics(self, window: int = 8, stride: int = 4, pooling: str = "mean"):
        """Build optional video-level semantic index"""
        if self.vector_index is None:
            print("Warning: No vector index available for video semantics")
            return
            
        print(f"\n--- Building Video Semantics Index ---")
        print(f"Window: {window}, Stride: {stride}, Pooling: {pooling}")
        
        # Load mapping
        mapping_path = self.artifact_dir / "mapping.parquet"
        if not mapping_path.exists():
            print("Warning: No mapping file found")
            return
            
        mapping_df = pd.read_parquet(mapping_path)
        
        # Load embeddings (reconstruct from index or reload)
        # For simplicity, we'll skip this and assume embeddings are available
        print("Video semantics building skipped - requires embedding reconstruction")
        # TODO: Implement video semantics building
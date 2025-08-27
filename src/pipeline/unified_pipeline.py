"""Unified pipeline combining data preprocessing and query processing"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import argparse

from .data_pipeline import DataPreprocessingPipeline  
from .query_pipeline import QueryProcessingPipeline
from ..core.base import SearchResult
import config


class UnifiedVideoPipeline:
    """Main pipeline orchestrator that combines all components"""
    
    def __init__(self, 
                 output_dir: Path = None,
                 artifact_dir: Path = None,
                 model_name: str = None,
                 pretrained: str = None,
                 use_transnet: bool = True):
        
        # Set default directories
        self.output_dir = Path(output_dir) if output_dir else Path("./pipeline_output")
        self.artifact_dir = Path(artifact_dir) if artifact_dir else (self.output_dir / "artifacts")
        
        # Model configuration
        self.model_name = model_name or config.MODEL_NAME
        self.pretrained = pretrained or config.MODEL_PRETRAINED
        self.use_transnet = use_transnet
        
        # Pipeline components (initialized on demand)
        self.data_pipeline: Optional[DataPreprocessingPipeline] = None
        self.query_pipeline: Optional[QueryProcessingPipeline] = None
        
    def build_index(self, 
                    video_paths: List[Path],
                    target_frames: int = 50,
                    batch_size: int = 32,
                    use_flat: bool = False,
                    enable_ocr: bool = True,
                    enable_captions: bool = True,
                    enable_segmentation: bool = False,
                    use_transnet: bool = None) -> Dict[str, Any]:
        """Build search index from video dataset"""
        
        print(f"\n🚀 Starting Data Preprocessing Pipeline")
        print(f"Video paths: {len(video_paths)} videos")
        print(f"Model: {self.model_name} ({self.pretrained})")
        
        # Use TransNet setting from parameter or instance default
        transnet_setting = use_transnet if use_transnet is not None else self.use_transnet
        
        print(f"🎬 Keyframe extraction: {'TransNet-V2 (academic)' if transnet_setting else 'Intelligent sampling'}")
        
        # Initialize data pipeline
        self.data_pipeline = DataPreprocessingPipeline(
            output_dir=self.output_dir,
            artifact_dir=self.artifact_dir,
            target_frames=target_frames,
            batch_size=batch_size,
            use_flat=use_flat,
            model_name=self.model_name,
            pretrained=self.pretrained,
            enable_ocr=enable_ocr,
            enable_captions=enable_captions,
            enable_segmentation=enable_segmentation,
            use_transnet=transnet_setting
        )
        
        # Process videos
        summary = self.data_pipeline.process_videos(video_paths)
        
        print(f"\n✅ Index building complete!")
        print(f"   - Processed {summary['total_videos']} videos")
        print(f"   - Extracted {summary['total_keyframes']} keyframes") 
        print(f"   - Built {summary['embedding_dimension']}D vector index")
        print(f"   - Created text corpus with {summary['corpus_size']} entries")
        
        return summary
        
    def search(self, 
               query: str,
               search_mode: str = "hybrid",
               k: int = 100,
               **kwargs) -> List[SearchResult]:
        """Search the built index"""
        
        # Initialize query pipeline if not already done
        if self.query_pipeline is None:
            if not self.artifact_dir.exists():
                raise FileNotFoundError(f"No artifacts found at {self.artifact_dir}. Run build_index() first.")
                
            self.query_pipeline = QueryProcessingPipeline(
                artifact_dir=self.artifact_dir,
                model_name=self.model_name,
                pretrained=self.pretrained
            )
        
        # Process query
        results = self.query_pipeline.search(
            query=query,
            search_mode=search_mode,
            k=k,
            **kwargs
        )
        
        return results
        
    def export_results(self, 
                      results: List[SearchResult], 
                      output_path: Path,
                      format: str = "csv",
                      **kwargs) -> Path:
        """Export search results"""
        
        if self.query_pipeline is None:
            raise RuntimeError("Query pipeline not initialized. Run search() first.")
            
        return self.query_pipeline.export_results(results, output_path, format, **kwargs)
        
    def run_end_to_end(self,
                      video_paths: List[Path],
                      queries: List[str],
                      output_dir: Path = None,
                      **pipeline_kwargs) -> Dict[str, Any]:
        """Run complete end-to-end pipeline"""
        
        output_dir = Path(output_dir) if output_dir else (self.output_dir / "results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎯 Running End-to-End Pipeline")
        print(f"Videos: {len(video_paths)}")
        print(f"Queries: {len(queries)}")
        
        # Step 1: Build index
        build_summary = self.build_index(video_paths, **pipeline_kwargs)
        
        # Step 2: Process queries
        all_results = {}
        for i, query in enumerate(queries):
            print(f"\n--- Query {i+1}/{len(queries)}: {query} ---")
            
            results = self.search(query, **pipeline_kwargs)
            all_results[query] = results
            
            # Export results
            query_filename = f"query_{i+1:03d}_{query[:30].replace(' ', '_')}.csv"
            query_output = output_dir / query_filename
            self.export_results(results, query_output)
            
        print(f"\n🎉 End-to-end pipeline complete!")
        print(f"Results saved to: {output_dir}")
        
        return {
            'build_summary': build_summary,
            'query_results': all_results,
            'output_directory': output_dir
        }


def main():
    """Command-line interface for the unified pipeline"""
    
    parser = argparse.ArgumentParser(description="Unified Video Retrieval Pipeline")
    parser.add_argument("command", choices=["build", "search", "end2end"], help="Command to run")
    
    # Common arguments
    parser.add_argument("--video_dir", type=Path, help="Directory containing videos")
    parser.add_argument("--video_pattern", type=str, default="*.mp4", help="Video file pattern")
    parser.add_argument("--output_dir", type=Path, default="./pipeline_output", help="Output directory")
    parser.add_argument("--artifact_dir", type=Path, help="Artifact directory (default: output_dir/artifacts)")
    parser.add_argument("--model_name", type=str, help="Override model name")
    parser.add_argument("--pretrained", type=str, help="Override pretrained weights")
    
    # Build arguments
    parser.add_argument("--target_frames", type=int, default=50, help="Frames per video")
    parser.add_argument("--batch_size", type=int, default=32, help="Processing batch size")
    parser.add_argument("--use_flat", action="store_true", help="Use flat FAISS index")
    
    # Academic keyframe extraction
    parser.add_argument("--use_transnet", action="store_true", default=True, help="Use TransNet-V2 for academic-grade shot boundary detection (default)")
    parser.add_argument("--disable_transnet", action="store_true", help="Disable TransNet-V2, use intelligent sampling instead")
    
    # NEW: Enable new near-SOTA features  
    parser.add_argument("--enable_ocr", action="store_true", help="Enable EasyOCR text extraction")
    parser.add_argument("--enable_captions", action="store_true", help="Enable BLIP-2 image captioning")
    parser.add_argument("--enable_segmentation", action="store_true", help="Enable FastSAM segmentation")
    parser.add_argument("--disable_ocr", action="store_true", help="Disable OCR (default: enabled)")
    parser.add_argument("--disable_captions", action="store_true", help="Disable captions (default: enabled)")
    
    # Search arguments  
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--queries_file", type=Path, help="File with multiple queries")
    parser.add_argument("--search_mode", type=str, default="hybrid", choices=["vector", "text", "hybrid"])
    parser.add_argument("--k", type=int, default=100, help="Number of results")
    parser.add_argument("--expand_query", action="store_true", help="Enable query expansion")
    
    args = parser.parse_args()
    
    # Determine TransNet-V2 setting  
    use_transnet = not args.disable_transnet  # Default: True (academic excellence)
    
    # Initialize pipeline
    pipeline = UnifiedVideoPipeline(
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        use_transnet=use_transnet
    )
    
    if args.command == "build":
        # Build index
        if not args.video_dir:
            raise ValueError("--video_dir required for build command")
            
        video_paths = list(Path(args.video_dir).rglob(args.video_pattern))
        if not video_paths:
            raise ValueError(f"No videos found in {args.video_dir} matching {args.video_pattern}")
            
        # Determine feature flags
        enable_ocr = not args.disable_ocr  # Default: enabled
        enable_captions = not args.disable_captions  # Default: enabled
        
        if args.enable_ocr:
            enable_ocr = True
        if args.enable_captions:
            enable_captions = True
            
        pipeline.build_index(
            video_paths=video_paths,
            target_frames=args.target_frames,
            batch_size=args.batch_size,
            use_flat=args.use_flat,
            enable_ocr=enable_ocr,
            enable_captions=enable_captions,
            enable_segmentation=args.enable_segmentation,
            use_transnet=use_transnet
        )
        
    elif args.command == "search":
        # Search
        queries = []
        if args.query:
            queries.append(args.query)
        elif args.queries_file and args.queries_file.exists():
            with open(args.queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("Either --query or --queries_file required for search command")
            
        for query in queries:
            results = pipeline.search(
                query=query,
                search_mode=args.search_mode,
                k=args.k,
                expand_query=args.expand_query
            )
            
            # Print results
            print(f"\nTop {min(10, len(results))} results for: {query}")
            for i, result in enumerate(results[:10]):
                print(f"{i+1:2d}. {result.video_id} frame {result.frame_idx} (score: {result.score:.3f})")
                
    elif args.command == "end2end":
        # End-to-end
        if not args.video_dir:
            raise ValueError("--video_dir required for end2end command")
            
        video_paths = list(Path(args.video_dir).rglob(args.video_pattern))
        
        queries = []
        if args.query:
            queries.append(args.query)
        elif args.queries_file and args.queries_file.exists():
            with open(args.queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
        else:
            queries = ["person walking", "outdoor scene", "text or writing"]  # Default queries
            
        pipeline.run_end_to_end(
            video_paths=video_paths,
            queries=queries,
            target_frames=args.target_frames,
            batch_size=args.batch_size,
            use_flat=args.use_flat,
            search_mode=args.search_mode,
            k=args.k,
            expand_query=args.expand_query
        )


if __name__ == "__main__":
    main()
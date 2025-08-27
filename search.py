#!/usr/bin/env python3
"""
New Search Interface
Clean interface for searching the built index
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline.query_pipeline import QueryProcessingPipeline
import config


def main():
    parser = argparse.ArgumentParser(description="Search video index")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--artifact_dir", type=Path, default=config.ARTIFACT_DIR, help="Artifact directory")
    parser.add_argument("--k", type=int, default=100, help="Number of results")
    parser.add_argument("--search_mode", type=str, default="hybrid", choices=["vector", "text", "hybrid"])
    parser.add_argument("--expand_query", action="store_true", help="Enable query expansion")
    parser.add_argument("--no_rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--output", type=Path, help="Output CSV file")
    parser.add_argument("--model_name", type=str, help="Override model name")
    parser.add_argument("--pretrained", type=str, help="Override pretrained weights")
    
    args = parser.parse_args()
    
    # Initialize query pipeline
    query_pipeline = QueryProcessingPipeline(
        artifact_dir=args.artifact_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        enable_reranking=not args.no_rerank
    )
    
    # Search
    results = query_pipeline.search(
        query=args.query,
        search_mode=args.search_mode,
        k=args.k,
        expand_query=args.expand_query
    )
    
    # Display results
    print(f"\nTop {min(20, len(results))} results for: '{args.query}'")
    print("-" * 80)
    for i, result in enumerate(results[:20]):
        search_type = result.metadata.get('search_type', 'unknown')
        print(f"{i+1:2d}. {result.video_id:<15} frame {result.frame_idx:>6d} "
              f"score: {result.score:6.3f} [{search_type}]")
    
    # Export if requested
    if args.output:
        query_pipeline.export_results(results, args.output)


if __name__ == "__main__":
    main()
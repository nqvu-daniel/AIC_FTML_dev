#!/usr/bin/env python3
"""
Create training data for reranker from AIC dataset metadata.
This script can be called from colab pipeline to generate training data automatically.
"""

import json
import argparse
import random
from pathlib import Path
import pandas as pd
from typing import List, Dict
import os
import sys

def create_training_data_from_metadata(dataset_root: str, output_file: str = "data/train.jsonl", num_examples: int = 50):
    """
    Generate training data from downloaded AIC metadata.
    This function is designed to be called from the colab pipeline.
    """
    dataset_root = Path(dataset_root)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if we have the metadata we downloaded earlier
    media_info_dir = dataset_root / "media_info" 
    map_keyframes_dir = dataset_root / "map_keyframes"
    
    # If we don't have the metadata locally, try to find it in tmp_download
    if not media_info_dir.exists():
        tmp_media_info = Path("tmp_download/media-info")
        tmp_map_keyframes = Path("tmp_download/map-keyframes") 
        
        if tmp_media_info.exists() and tmp_map_keyframes.exists():
            media_info_dir = tmp_media_info
            map_keyframes_dir = tmp_map_keyframes
        else:
            print("ERROR: No metadata found. Please run dataset download first.")
            return False
    
    print(f"Using metadata from:")
    print(f"  Media info: {media_info_dir}")
    print(f"  Keyframes: {map_keyframes_dir}")
    
    # Find available videos
    video_ids = []
    if media_info_dir.exists():
        for json_file in media_info_dir.glob("*.json"):
            video_id = json_file.stem
            # Check if we also have keyframe mapping
            if (map_keyframes_dir / f"{video_id}.csv").exists():
                video_ids.append(video_id)
    
    if not video_ids:
        print("ERROR: No videos with complete metadata found")
        return False
    
    print(f"Found {len(video_ids)} videos with complete metadata")
    
    # Sample videos if we have too many
    if len(video_ids) > 20:
        video_ids = random.sample(video_ids, 20)
        print(f"Sampling {len(video_ids)} videos for training data generation")
    
    # Query templates based on the Vietnamese news content we found
    query_templates = [
        # Vietnamese news queries
        "tin tức mới nhất",
        "bản tin hôm nay", 
        "thời sự việt nam",
        "tin tức sáng nay",
        "chương trình 60 giây",
        "HTV tin tức",
        "báo cáo thông tin",
        "cập nhật tin tức",
        
        # English news/media queries  
        "news anchor speaking",
        "television broadcast",
        "reporter on camera", 
        "news program segment",
        "live news show",
        "studio presentation",
        "media interview",
        "news update show",
        "presenter talking",
        "broadcast journalism",
        
        # General content queries
        "person presenting",
        "people talking",
        "television studio",
        "media content",
        "video presentation",
        "professional broadcast",
        "communication show",
        "information program"
    ]
    
    training_data = []
    
    for video_id in video_ids:
        try:
            # Load media info
            media_file = media_info_dir / f"{video_id}.json"
            with open(media_file, 'r', encoding='utf-8') as f:
                media_info = json.load(f)
            
            # Load keyframe mapping  
            map_file = map_keyframes_dir / f"{video_id}.csv"
            df = pd.read_csv(map_file)
            
            if df.empty:
                continue
                
            # Get video metadata for query generation
            title = media_info.get("title", "")
            keywords = media_info.get("keywords", [])
            description = media_info.get("description", "")
            
            # Select representative frames (every 20% of video)
            total_frames = len(df)
            if total_frames < 3:
                continue
                
            # Select 3-5 frames spread across the video
            num_frames = min(5, max(3, total_frames // 10))
            frame_indices = []
            
            for i in range(num_frames):
                idx = int((i * total_frames) // num_frames) 
                if idx < len(df):
                    frame_indices.append(int(df.iloc[idx]['frame_idx']))
            
            # Generate 2-3 queries for this video
            selected_queries = random.sample(query_templates, min(3, len(query_templates)))
            
            # Add keyword-based queries if available
            if keywords:
                # Use Vietnamese keywords if available
                vn_keywords = [kw for kw in keywords if any(c for c in kw if ord(c) > 127)]
                if vn_keywords:
                    selected_queries.append(random.choice(vn_keywords[:3]))
            
            for query in selected_queries:
                # Create training example
                positives = [{"video_id": video_id, "frame_idx": frame_idx} for frame_idx in frame_indices]
                
                training_data.append({
                    "query": query.strip(),
                    "positives": positives
                })
                
                if len(training_data) >= num_examples:
                    break
            
            if len(training_data) >= num_examples:
                break
                
        except Exception as e:
            print(f"Warning: Error processing {video_id}: {e}")
            continue
    
    if not training_data:
        print("ERROR: No training data generated")
        return False
    
    # Save training data
    print(f"Generated {len(training_data)} training examples")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Training data saved to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create training data from AIC metadata")
    parser.add_argument("--dataset_root", type=str, default="/content/aic2025", 
                       help="Dataset root directory")
    parser.add_argument("--output", type=str, default="data/train.jsonl",
                       help="Output training file")
    parser.add_argument("--num_examples", type=int, default=50,
                       help="Number of training examples to generate")
    
    args = parser.parse_args()
    
    success = create_training_data_from_metadata(
        args.dataset_root, 
        args.output, 
        args.num_examples
    )
    
    if success:
        print("\n✅ Training data creation successful!")
        print(f"Next step: Train the reranker with:")
        print(f"python src/training/train_reranker.py --index_dir ./artifacts --train_jsonl {args.output}")
        return 0
    else:
        print("\n❌ Training data creation failed!")
        return 1

if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Smart end-to-end pipeline with CLIP-guided frame sampling and indexing.
"""

import argparse
import os
import json
import sys
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Conservative dependency handling - only install if missing and explicitly requested
FORCE_INSTALL = os.getenv('AIC_FORCE_INSTALL', '').lower() in ('1', 'true', 'yes')

def try_import_with_fallback():
    """Try to import dependencies, only install if FORCE_INSTALL is set"""
    
    # First try - check what's missing
    try:
        import numpy as np
        import pandas as pd  
        import torch
        from tqdm import tqdm
        from PIL import Image
        import faiss
        import open_clip
        return True, None  # All imports successful
    except ImportError as e:
        missing_dep = str(e).split("'")[1] if "'" in str(e) else str(e)
        
        if not FORCE_INSTALL:
            print(f"Missing dependency: {missing_dep}")
            print("To auto-install dependencies, set environment variable: AIC_FORCE_INSTALL=1")
            print("Or install manually with: pip install -r requirements.txt")
            return False, [missing_dep]
        
        # Auto-install mode - install all requirements at once
        print(f"Missing dependency detected: {missing_dep}. Installing all requirements...")
        
        import subprocess
        try:
            # Install from requirements.txt if available
            req_file = Path(__file__).parent.parent / "requirements.txt"
            if req_file.exists():
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", str(req_file)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("Successfully installed requirements from requirements.txt")
            else:
                # Fallback to individual packages
                packages = [
                    "torch>=2.1", "torchvision>=0.16.0", "torchaudio>=2.1.0",
                    "open-clip-torch>=2.24.0", "faiss-cpu>=1.7.4",
                    "opencv-python-headless>=4.8.0", "Pillow>=10.0",
                    "numpy>=1.24", "pandas>=2.0", "scipy>=1.11.0",
                    "scikit-learn>=1.4", "tqdm>=4.66", "pyarrow>=14.0.0"
                ]
                for package in packages:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("Successfully installed individual packages")
            
        except subprocess.CalledProcessError as install_error:
            print(f"Failed to install dependencies: {install_error}")
            return False, [missing_dep]
        
        # Try imports again after installation
        try:
            import numpy as np
            import pandas as pd
            import torch  
            from tqdm import tqdm
            from PIL import Image
            import faiss
            import open_clip
            print("All dependencies successfully imported after installation!")
            return True, None
        except ImportError as e:
            return False, [str(e)]

# Try importing with conservative fallback
success, errors = try_import_with_fallback()
if not success:
    print("Failed to import required dependencies:")
    for error in errors or []:
        print(f"  - {error}")
    sys.exit(1)

# Import successful, continue with regular imports
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
import faiss
import open_clip

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import ensure_dir, load_image, normalize_rows, save_faiss, to_parquet, as_type
import config
from src.sampling.frames_auto import CLIPGuidedFrameSampler


def extract_keyframes_with_sampling(video_paths, output_dir, sampler, target_frames=50):
    """Extract keyframes using CLIP-guided sampling"""
    
    output_dir = Path(output_dir)
    keyframes_dir = output_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    
    mapping_data = []
    
    for video_path in tqdm(video_paths, desc="Sampling keyframes"):
        video_path = Path(video_path)
        video_id = video_path.stem
        
        try:
            # Sample frames intelligently
            selected_frames = sampler.sample_frames_intelligent(
                video_path, target_frames=target_frames
            )
            
            if not selected_frames:
                print(f"Warning: No frames selected for {video_id}")
                continue
            
            # Extract and save keyframes
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            
            for i, (frame_idx, timestamp, scores) in enumerate(selected_frames):
                # Read specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Save keyframe
                    frame_filename = f"{video_id}_frame_{frame_idx:06d}.jpg"
                    frame_path = keyframes_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Add to mapping
                    mapping_data.append({
                        'video_id': video_id,
                        'frame_path': str(frame_path),
                        'frame_index': frame_idx,
                        'timestamp': timestamp,
                        'relevance_score': scores['relevance'],
                        'is_scene_boundary': scores['scene_boundary']
                    })
            
            cap.release()
            print(f"Extracted {len(selected_frames)} keyframes from {video_id}")
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    # Save mapping data
    mapping_df = pd.DataFrame(mapping_data)
    mapping_path = output_dir / "keyframe_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    
    print(f"Extracted {len(mapping_data)} total keyframes")
    print(f"Keyframes saved to: {keyframes_dir}")
    print(f"Mapping saved to: {mapping_path}")
    
    return mapping_data


def embed_keyframes(model, preprocess, device, keyframe_paths, batch_size=64):
    """Embed keyframes using the model"""
    embeddings = []
    
    for i in tqdm(range(0, len(keyframe_paths), batch_size), desc="Computing embeddings"):
        batch_paths = keyframe_paths[i:i+batch_size]
        
        # Load and preprocess images
        images = []
        for path in batch_paths:
            try:
                img = load_image(path)
                img_tensor = preprocess(img).unsqueeze(0)
                images.append(img_tensor)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Create a zero tensor as placeholder
                images.append(torch.zeros(1, 3, 224, 224))
        
        if images:
            batch_tensor = torch.cat(images).to(device)
            
            with torch.no_grad():
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        batch_embeddings = model.encode_image(batch_tensor)
                else:
                    batch_embeddings = model.encode_image(batch_tensor)
                    
            embeddings.append(batch_embeddings.cpu().numpy())
    
    if embeddings:
        return np.concatenate(embeddings, axis=0)
    else:
        return np.array([])


def build_faiss_index(embeddings, use_flat=False):
    """Build FAISS index from embeddings"""
    
    if len(embeddings) == 0:
        raise ValueError("No embeddings to index")
    
    # Normalize embeddings for cosine similarity
    embeddings = normalize_rows(embeddings)
    
    dim = embeddings.shape[1]
    print(f"Building FAISS index for {len(embeddings)} embeddings of dim {dim}")
    
    if use_flat:
        # Exact search
        index = faiss.IndexFlatIP(dim)
        print("Using exact IndexFlatIP")
    else:
        # HNSW for faster approximate search
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 100
        print("Using HNSW index")
    
    index.add(embeddings.astype('float32'))
    print(f"Index built with {index.ntotal} vectors")
    
    return index


def create_mapping_dataframe(keyframe_data, embeddings):
    """Create mapping dataframe for the index"""
    
    mapping_rows = []
    
    for i, item in enumerate(keyframe_data):
        mapping_rows.append({
            'global_id': i,
            'video_id': item['video_id'],
            'frame_path': item['frame_path'],
            'frame_index': item['frame_index'],
            'timestamp': item['timestamp'],
            'relevance_score': item['relevance_score'],
            'is_scene_boundary': item['is_scene_boundary']
        })
    
    return pd.DataFrame(mapping_rows)


def build_text_corpus(mapping_df, keyframes_dir):
    """Build text corpus from frame metadata for BM25"""
    
    corpus_entries = []
    
    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="Building text corpus"):
        # Create text description for each frame
        video_id = row['video_id']
        timestamp = row['timestamp']
        relevance = row['relevance_score']
        is_boundary = row['is_scene_boundary']
        
        # Extract collection and video number
        collection_match = re.match(r'(L\d+)_V(\d+)', video_id)
        if collection_match:
            collection = collection_match.group(1)
            video_num = collection_match.group(2)
        else:
            collection = "unknown"
            video_num = "unknown"
        
        # Create descriptive text
        text_parts = [
            f"video {video_id}",
            f"collection {collection}",
            f"video number {video_num}",
            f"timestamp {timestamp:.1f} seconds"
        ]
        
        if relevance > 0.7:
            text_parts.append("high relevance frame")
        elif relevance > 0.5:
            text_parts.append("medium relevance frame")
        
        if is_boundary:
            text_parts.append("scene boundary frame")
        
        # Combine into raw text
        raw_text = " ".join(text_parts)
        
        # Simple tokenization
        tokens = raw_text.lower().replace(",", "").split()
        
        corpus_entries.append({
            "raw": raw_text,
            "tokens": tokens
        })
    
    return corpus_entries


def main():
    parser = argparse.ArgumentParser(description="Smart pipeline with CLIP-guided sampling and indexing")
    parser.add_argument("--video_dir", type=Path, required=True, help="Directory containing video files")
    parser.add_argument("--video_pattern", type=str, default="*.mp4", help="Video file pattern")
    parser.add_argument("--output_dir", type=Path, default="./pipeline_output", help="Output directory")
    parser.add_argument("--artifact_dir", type=Path, default=None, help="Artifact output directory (default: output_dir/artifacts)")
    parser.add_argument("--target_frames", type=int, default=50, help="Target frames per video")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--use_flat", action="store_true", help="Use exact FAISS index instead of HNSW")
    parser.add_argument("--sampling_only", action="store_true", help="Only do frame sampling, skip indexing")
    parser.add_argument("--indexing_only", action="store_true", help="Only do indexing, skip sampling (requires existing keyframes)")
    parser.add_argument("--experimental", action="store_true", help="Use experimental model")
    parser.add_argument("--exp_model", type=str, default=None, help="Experimental model preset")
    
    args = parser.parse_args()
    
    # Setup directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.artifact_dir is None:
        args.artifact_dir = args.output_dir / "artifacts"
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find videos
    if not args.indexing_only:
        video_files = list(args.video_dir.rglob(args.video_pattern))
        print(f"Found {len(video_files)} video files")
        
        if not video_files:
            print(f"No videos found matching {args.video_pattern}")
            return
    
    # Model selection
    if args.experimental and args.exp_model:
        if args.exp_model in config.EXPERIMENTAL_PRESETS:
            model_name, pretrained = config.EXPERIMENTAL_PRESETS[args.exp_model]
            print(f"Using experimental preset '{args.exp_model}': {model_name} with {pretrained}")
        else:
            model_name = args.exp_model
            pretrained = config.MODEL_PRETRAINED
            print(f"Using experimental model: {model_name} with {pretrained}")
    else:
        model_name = config.MODEL_NAME
        pretrained = config.MODEL_PRETRAINED
        print(f"Using configured model: {model_name} with {pretrained}")
    
    # Load model
    print(f"Loading {model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    
    # Step 1: Frame Sampling
    if not args.indexing_only:
        print("\n=== STEP 1: CLIP-Guided Frame Sampling ===")
        
        sampler = CLIPGuidedFrameSampler(
            dataset_root=args.video_dir,
            batch_size=args.batch_size,
            model_name=model_name
        )
        
        keyframe_data = extract_keyframes_with_sampling(
            video_files, 
            args.output_dir,
            sampler,
            target_frames=args.target_frames
        )
        
        if args.sampling_only:
            print("Sampling complete! Use --indexing_only to build the index.")
            return
    
    # Step 2: Load existing keyframes if indexing only
    if args.indexing_only:
        print("\n=== Loading Existing Keyframes ===")
        mapping_path = args.output_dir / "keyframe_mapping.csv"
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
        
        mapping_df = pd.read_csv(mapping_path)
        keyframe_data = mapping_df.to_dict('records')
        print(f"Loaded {len(keyframe_data)} keyframes from mapping")
    
    # Step 3: Indexing
    print("\n=== STEP 2: Building Search Index ===")
    
    # Get keyframe paths
    keyframe_paths = [item['frame_path'] for item in keyframe_data]
    print(f"Embedding {len(keyframe_paths)} keyframes...")
    
    # Compute embeddings
    embeddings = embed_keyframes(model, preprocess, device, keyframe_paths, args.batch_size)
    
    if len(embeddings) == 0:
        raise ValueError("No embeddings computed!")
    
    print(f"Computed {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Build FAISS index
    index = build_faiss_index(embeddings, use_flat=args.use_flat)
    
    # Create mapping dataframe
    mapping_df = create_mapping_dataframe(keyframe_data, embeddings)
    
    # Build text corpus
    print("Building text corpus for BM25...")
    corpus_entries = build_text_corpus(mapping_df, args.output_dir / "keyframes")
    
    # Save artifacts
    print("\n=== STEP 3: Saving Artifacts ===")
    
    # Save FAISS index
    index_path = args.artifact_dir / "index.faiss"
    save_faiss(index, index_path)
    print(f"FAISS index saved: {index_path}")
    
    # Save mapping
    mapping_path = args.artifact_dir / "mapping.parquet"
    to_parquet(mapping_df, mapping_path)
    print(f"Mapping saved: {mapping_path}")
    
    # Save text corpus
    corpus_path = args.artifact_dir / "text_corpus.jsonl"
    with open(corpus_path, 'w') as f:
        for entry in corpus_entries:
            f.write(json.dumps(entry) + '\n')
    print(f"Text corpus saved: {corpus_path}")
    
    # Save pipeline info
    info = {
        "model_name": model_name,
        "model_pretrained": pretrained,
        "embedding_dim": int(embeddings.shape[1]),
        "total_keyframes": len(keyframe_data),
        "total_videos": len(set(item['video_id'] for item in keyframe_data)),
        "target_frames_per_video": args.target_frames,
        "index_type": "flat" if args.use_flat else "hnsw",
        "sampling_method": "clip_guided"
    }
    
    info_path = args.artifact_dir / "pipeline_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Pipeline info saved: {info_path}")
    
    print(f"\n✅ Pipeline complete! Artifacts saved to: {args.artifact_dir}")
    print(f"   - Index: {len(keyframe_data)} keyframes from {info['total_videos']} videos")
    print(f"   - Model: {model_name} ({pretrained})")
    print(f"   - Embedding dim: {info['embedding_dim']}")


if __name__ == "__main__":
    main()
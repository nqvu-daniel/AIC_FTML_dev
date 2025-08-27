#!/usr/bin/env python3
"""
Example: Using the new near-SOTA encoders (FastSAM, EasyOCR, BLIP-2)
Shows how to use the implemented alternatives to complete your architecture.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.encoders.sam_encoder import FastSAMEncoder
from src.encoders.ocr_encoder import EasyOCREncoder  
from src.encoders.blip_encoder import BLIPCaptioner
from src.pipeline.data_pipeline import DataPreprocessingPipeline
from src.pipeline.unified_pipeline import UnifiedVideoPipeline


def demo_individual_encoders():
    """Demo each encoder individually"""
    
    print("🚀 Testing Individual Encoders")
    print("=" * 50)
    
    # Test image paths (replace with your actual images)
    test_images = [
        "example_image1.jpg",  # Replace with actual paths
        "example_image2.jpg"
    ]
    
    print("\n1. 🎯 FastSAM (50x faster than SAM2)")
    print("-" * 30)
    try:
        fastsam = FastSAMEncoder()
        if fastsam.available:
            results = fastsam.process(test_images[:1])  # Test with one image
            print(f"✅ FastSAM: {len(results)} results")
            print(f"   Masks: {results[0]['num_masks']} detected")
            print(f"   Feature dim: {len(results[0]['features'])}")
        else:
            print("⚠️  FastSAM not available (need: pip install ultralytics)")
    except Exception as e:
        print(f"❌ FastSAM error: {e}")
    
    print("\n2. 📝 EasyOCR (70+ languages)")  
    print("-" * 30)
    try:
        ocr = EasyOCREncoder(languages=['en', 'vi'])
        if ocr.available:
            texts = ocr.encode(test_images[:1])  # Test with one image
            print(f"✅ EasyOCR: {len(texts)} texts extracted")
            if texts[0]:
                print(f"   Sample: '{texts[0][:50]}...'")
            else:
                print("   No text detected in test image")
        else:
            print("⚠️  EasyOCR not available (need: pip install easyocr)")
    except Exception as e:
        print(f"❌ EasyOCR error: {e}")
        
    print("\n3. 🖼️ BLIP-2 (SOTA captioning)")
    print("-" * 30) 
    try:
        blip = BLIPCaptioner()
        if blip.available:
            captions = blip.encode(test_images[:1])  # Test with one image
            print(f"✅ BLIP-2: {len(captions)} captions generated")
            if captions[0]:
                print(f"   Caption: '{captions[0]}'")
            else:
                print("   No caption generated")
        else:
            print("⚠️  BLIP-2 not available (need: pip install transformers)")
    except Exception as e:
        print(f"❌ BLIP-2 error: {e}")


def demo_integrated_pipeline():
    """Demo the full integrated pipeline with new features"""
    
    print("\n\n🎯 Full Pipeline with New Features")
    print("=" * 50)
    
    # Example video directory (replace with actual path)
    video_dir = Path("./example_videos")
    output_dir = Path("./pipeline_output_new")
    
    if not video_dir.exists():
        print(f"⚠️  Video directory not found: {video_dir}")
        print("   Create example_videos/ with some .mp4 files to test")
        return
    
    try:
        # Create pipeline with new features enabled
        pipeline = DataPreprocessingPipeline(
            output_dir=output_dir,
            artifact_dir=output_dir / "artifacts",
            target_frames=20,  # Smaller for demo
            batch_size=16,
            enable_ocr=True,      # 📝 Enable EasyOCR
            enable_captions=True, # 🖼️ Enable BLIP-2  
            enable_segmentation=False  # 🎯 FastSAM (optional)
        )
        
        # Find videos
        video_files = list(video_dir.rglob("*.mp4"))
        if not video_files:
            print(f"No .mp4 files found in {video_dir}")
            return
            
        print(f"Found {len(video_files)} videos")
        
        # Process with new features
        summary = pipeline.process_videos(video_files[:2])  # Process first 2 videos
        
        print("\n✅ Pipeline Results:")
        print(f"   Videos processed: {summary['total_videos']}")
        print(f"   Keyframes extracted: {summary['total_keyframes']}")
        print(f"   Text corpus size: {summary['corpus_size']}")
        print(f"   Embedding dimension: {summary['embedding_dimension']}")
        
        # Check what features were extracted
        corpus_path = output_dir / "artifacts" / "text_corpus.jsonl"
        if corpus_path.exists():
            import json
            with open(corpus_path, 'r') as f:
                sample_entry = json.loads(f.readline())
                
            print(f"\n📊 Sample corpus entry:")
            print(f"   Raw text length: {len(sample_entry['raw'])} chars")
            print(f"   Has OCR: {sample_entry.get('has_ocr', False)}")
            print(f"   Has caption: {sample_entry.get('has_caption', False)}")
            print(f"   Sample text: '{sample_entry['raw'][:100]}...'")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")


def demo_unified_interface():
    """Demo the unified pipeline interface"""
    
    print("\n\n🎯 Unified Pipeline Interface")
    print("=" * 50)
    
    video_dir = Path("./example_videos") 
    
    if not video_dir.exists():
        print(f"⚠️  Video directory not found: {video_dir}")
        return
        
    try:
        # Use the clean unified interface
        pipeline = UnifiedVideoPipeline(
            output_dir=Path("./unified_output"),
            model_name="ViT-L-16-SigLIP-256"  # Your current CLIP model
        )
        
        # Find videos
        video_paths = list(video_dir.rglob("*.mp4"))[:1]  # Just 1 for demo
        
        if video_paths:
            print(f"Processing {len(video_paths)} videos...")
            
            # Build index with new features
            summary = pipeline.build_index(
                video_paths=video_paths,
                target_frames=10,  # Small for demo
                batch_size=8
            )
            
            print(f"✅ Index built: {summary['total_keyframes']} keyframes")
            
            # Test search
            results = pipeline.search(
                query="person walking outdoor scene",
                search_mode="hybrid",
                k=5
            )
            
            print(f"✅ Search results: {len(results)} matches")
            for i, result in enumerate(results[:3]):
                print(f"   {i+1}. {result.video_id} frame {result.frame_idx} (score: {result.score:.3f})")
                
    except Exception as e:
        print(f"❌ Unified pipeline error: {e}")


if __name__ == "__main__":
    print("🎬 AIC FTML - New Near-SOTA Features Demo")
    print("Testing FastSAM + EasyOCR + BLIP-2 integration")
    print("=" * 60)
    
    # Test individual encoders
    demo_individual_encoders()
    
    # Test integrated pipeline 
    demo_integrated_pipeline()
    
    # Test unified interface
    demo_unified_interface()
    
    print("\n" + "=" * 60)
    print("🎉 Demo complete!")
    print("\nTo use these features:")
    print("1. Install dependencies: pip install ultralytics easyocr transformers accelerate")
    print("2. Run: python pipeline.py build --video_dir /data --enable_ocr --enable_captions")
    print("3. Search: python search.py --query 'your search text'")
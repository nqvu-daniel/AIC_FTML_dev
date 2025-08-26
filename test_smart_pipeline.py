#!/usr/bin/env python3
"""
Test script for the updated smart pipeline with CLIP-guided sampling.
"""

import sys
from pathlib import Path
import subprocess
import tempfile
import shutil

def test_smart_pipeline():
    """Test the smart pipeline components"""
    
    print("🧪 Testing Smart Pipeline Components...")
    
    # Test 1: Check if imports work
    print("\n1. Testing imports...")
    try:
        from src.sampling.frames_auto import CLIPGuidedFrameSampler
        import config
        print("   ✅ All imports successful")
        print(f"   📊 Current model: {config.MODEL_NAME} ({config.MODEL_PRETRAINED})")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Check if model loads
    print("\n2. Testing model loading...")
    try:
        sampler = CLIPGuidedFrameSampler(
            dataset_root=Path("."),
            batch_size=8,  # Small batch for testing
            use_gpu=False  # CPU for testing
        )
        print("   ✅ CLIP model loaded successfully")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False
    
    # Test 3: Check if smart_pipeline.py script exists and is callable
    print("\n3. Testing smart pipeline script...")
    script_path = Path("scripts/smart_pipeline.py")
    if script_path.exists():
        print("   ✅ Smart pipeline script found")
        
        # Try to get help output
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and "--target_frames" in result.stdout:
                print("   ✅ Smart pipeline script is callable")
            else:
                print(f"   ❌ Script help failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"   ❌ Script test failed: {e}")
            return False
    else:
        print(f"   ❌ Smart pipeline script not found: {script_path}")
        return False
    
    # Test 4: Check query templates and embeddings
    print("\n4. Testing query processing...")
    try:
        query_embeddings = sampler._encode_text_batch(["test query", "another test"])
        print(f"   ✅ Query embeddings computed: {query_embeddings.shape}")
    except Exception as e:
        print(f"   ❌ Query processing failed: {e}")
        return False
    
    # Test 5: Check prepare_pipeline_dir includes new files
    print("\n5. Testing pipeline preparation...")
    try:
        from scripts.prepare_pipeline_dir import NEEDED_FILES
        needed_paths = [str(f) for f in NEEDED_FILES]
        if "src/sampling/frames_auto.py" in needed_paths:
            print("   ✅ Pipeline preparation includes frames_auto.py")
        else:
            print("   ❌ frames_auto.py not in NEEDED_FILES")
            print(f"   📋 Current files: {needed_paths}")
            return False
    except Exception as e:
        print(f"   ❌ Pipeline preparation test failed: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    print("\n📚 Usage Examples:")
    print("   # Full pipeline with SigLIP2:")
    print("   python scripts/smart_pipeline.py --video_dir /path/to/videos --experimental --exp_model siglip2-l16-256")
    print("\n   # Just sampling:")
    print("   python scripts/smart_pipeline.py --video_dir /path/to/videos --sampling_only --target_frames 30")
    print("\n   # Traditional pipeline with new sampling:")
    print("   python smart_pipeline.py /path/to/dataset")
    
    return True

def test_config_models():
    """Test available model configurations"""
    print("\n🔧 Testing Model Configurations...")
    
    try:
        import config
        
        print(f"   🎯 Default model: {config.MODEL_NAME} ({config.MODEL_PRETRAINED})")
        print(f"   🔄 Fallback model: {config.DEFAULT_CLIP_MODEL} ({config.DEFAULT_CLIP_PRETRAINED})")
        
        if hasattr(config, 'EXPERIMENTAL_PRESETS'):
            print(f"   🧪 Available experimental presets:")
            for key, (model, pretrained) in config.EXPERIMENTAL_PRESETS.items():
                print(f"      - {key}: {model} ({pretrained})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Config test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Smart Pipeline Test Suite")
    print("=" * 50)
    
    success = test_smart_pipeline()
    if success:
        test_config_models()
        print("\n✅ Pipeline is ready to use!")
    else:
        print("\n❌ Pipeline has issues - check the errors above")
        sys.exit(1)
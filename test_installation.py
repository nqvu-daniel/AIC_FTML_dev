#!/usr/bin/env python3
"""
Test installation script for AIC FTML academic pipeline.
Validates that all critical dependencies are properly installed and working.
"""

import sys
import importlib.util
from pathlib import Path


def test_import(module_name: str, friendly_name: str, critical: bool = True) -> bool:
    """Test if a module can be imported successfully."""
    try:
        if module_name == "transnetv2_pytorch":
            # Special case for TransNet-V2
            import transnetv2_pytorch
            # Try to access core functionality
            _ = transnetv2_pytorch.TransNetV2
            print(f"✅ {friendly_name}")
            return True
        elif module_name == "faiss":
            import faiss
            # Test basic FAISS functionality
            index = faiss.IndexFlatIP(128)
            print(f"✅ {friendly_name} - Basic functionality OK")
            return True
        elif module_name == "cv2":
            import cv2
            # Test basic OpenCV functionality
            _ = cv2.__version__
            print(f"✅ {friendly_name} - Version: {cv2.__version__}")
            return True
        elif module_name == "torch":
            import torch
            cuda_status = "CUDA available" if torch.cuda.is_available() else "CPU only"
            print(f"✅ {friendly_name} - Version: {torch.__version__} ({cuda_status})")
            return True
        else:
            __import__(module_name)
            print(f"✅ {friendly_name}")
            return True
            
    except ImportError as e:
        status = "❌" if critical else "⚠️"
        print(f"{status} {friendly_name}: {e}")
        return False
    except Exception as e:
        status = "❌" if critical else "⚠️"
        print(f"{status} {friendly_name}: Unexpected error - {e}")
        return False


def test_academic_pipeline():
    """Test academic pipeline components specifically."""
    print("\n🏆 Testing Academic Pipeline Components...")
    
    # Test TransNet-V2 shot boundary detection
    try:
        import transnetv2_pytorch
        import torch
        
        # Create a simple test
        model = transnetv2_pytorch.TransNetV2()
        print("✅ TransNet-V2 model initialization")
        
        # Test with dummy data
        dummy_frames = torch.randn(1, 100, 3, 224, 224)  # 100 frames batch
        with torch.no_grad():
            predictions = model(dummy_frames)
        print("✅ TransNet-V2 shot boundary detection")
        
    except Exception as e:
        print(f"❌ TransNet-V2 functionality: {e}")
    
    # Test CLIP integration
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        print("✅ OpenCLIP model loading")
    except Exception as e:
        print(f"❌ OpenCLIP functionality: {e}")
    
    # Test FAISS GPU vs CPU
    try:
        import faiss
        import torch
        
        if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
            print("✅ GPU FAISS functionality available")
        else:
            print("✅ CPU FAISS functionality (GPU not available)")
    except Exception as e:
        print(f"❌ FAISS functionality: {e}")


def main():
    print("🧪 AIC FTML Installation Test")
    print("=" * 50)
    
    # Critical dependencies that must work
    critical_deps = [
        ("numpy", "NumPy", True),
        ("torch", "PyTorch", True),
        ("torchvision", "TorchVision", True),
        ("cv2", "OpenCV", True),
        ("PIL", "Pillow", True),
        ("pandas", "Pandas", True),
        ("faiss", "FAISS", True),
        ("transnetv2_pytorch", "TransNet-V2", True),
        ("open_clip", "OpenCLIP", True),
        ("ffmpeg", "ffmpeg-python", True),
    ]
    
    # Enhanced academic dependencies
    enhanced_deps = [
        ("ultralytics", "Ultralytics (FastSAM)", False),
        ("easyocr", "EasyOCR", False),
        ("transformers", "Transformers (BLIP-2)", False),
        ("accelerate", "Accelerate", False),
        ("rank_bm25", "BM25", False),
        ("sklearn", "Scikit-learn", False),
        ("scipy", "SciPy", False),
        ("decord", "Decord", False),
        ("pyarrow", "PyArrow", False),
    ]
    
    print("🔧 Testing Critical Dependencies:")
    critical_passed = 0
    for module, name, critical in critical_deps:
        if test_import(module, name, critical):
            critical_passed += 1
    
    print(f"\n🏆 Testing Enhanced Academic Dependencies:")
    enhanced_passed = 0
    for module, name, critical in enhanced_deps:
        if test_import(module, name, critical):
            enhanced_passed += 1
    
    # Test academic pipeline functionality
    test_academic_pipeline()
    
    # Summary
    print(f"\n📊 Installation Test Results:")
    print(f"  Critical dependencies: {critical_passed}/{len(critical_deps)} ✅")
    print(f"  Enhanced dependencies: {enhanced_passed}/{len(enhanced_deps)} ✅")
    
    if critical_passed == len(critical_deps):
        print(f"\n🎉 Installation test PASSED!")
        print(f"  ✅ All critical dependencies working")
        print(f"  ✅ Academic pipeline ready for TransNet-V2 processing")
        print(f"  📚 Ready for academic competition deployment")
        return 0
    else:
        print(f"\n❌ Installation test FAILED!")
        print(f"  Missing {len(critical_deps) - critical_passed} critical dependencies")
        print(f"  📝 Please resolve dependency conflicts and re-run test")
        return 1


if __name__ == "__main__":
    sys.exit(main())
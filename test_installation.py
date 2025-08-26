#!/usr/bin/env python3
"""
Test script to verify all dependencies are properly installed.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_core_imports():
    """Test core library imports"""
    print("Testing core library imports...")

    try:
        import torch

        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not available")
        return False

    try:
        import torchvision

        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError:
        print("✗ torchvision not available")
        return False

    try:
        import open_clip

        print("✓ open_clip available")
    except ImportError:
        print("✗ open_clip not available")
        return False

    try:
        import faiss

        print("✓ faiss available")
    except ImportError:
        print("✗ faiss not available")
        return False

    return True


def test_cv_imports():
    """Test computer vision imports"""
    print("\nTesting computer vision imports...")

    try:
        import cv2

        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not available")
        return False

    try:
        from PIL import Image

        print("✓ Pillow available")
    except ImportError:
        print("✗ Pillow not available")
        return False

    try:
        import decord

        print("✓ decord available")
    except ImportError:
        print("✗ decord not available")
        return False

    return True


def test_data_imports():
    """Test data processing imports"""
    print("\nTesting data processing imports...")

    try:
        import pandas as pd

        print(f"✓ pandas {pd.__version__}")
    except ImportError:
        print("✗ pandas not available")
        return False

    try:
        import numpy as np

        print(f"✓ numpy {np.__version__}")
    except ImportError:
        print("✗ numpy not available")
        return False

    try:
        import scipy

        print(f"✓ scipy {scipy.__version__}")
    except ImportError:
        print("✗ scipy not available")
        return False

    try:
        import pyarrow

        print(f"✓ pyarrow {pyarrow.__version__}")
    except ImportError:
        print("✗ pyarrow not available")
        return False

    return True


def test_ml_imports():
    """Test machine learning imports"""
    print("\nTesting machine learning imports...")

    try:
        import sklearn

        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        print("✗ scikit-learn not available")
        return False

    try:
        import joblib

        print(f"✓ joblib {joblib.__version__}")
    except ImportError:
        print("✗ joblib not available")
        return False

    try:
        import rank_bm25

        print("✓ rank_bm25 available")
    except ImportError:
        print("✗ rank_bm25 not available")
        return False

    return True


def test_utility_imports():
    """Test utility imports"""
    print("\nTesting utility imports...")

    try:
        import tqdm

        print(f"✓ tqdm {tqdm.__version__}")
    except ImportError:
        print("✗ tqdm not available")
        return False

    try:
        import yaml

        print("✓ pyyaml available")
    except ImportError:
        print("✗ pyyaml not available")
        return False

    return True


def test_optional_imports():
    """Test optional imports"""
    print("\nTesting optional imports...")

    try:
        import lightgbm

        print(f"✓ lightgbm {lightgbm.__version__} (optional)")
    except ImportError:
        print("○ lightgbm not available (optional)")

    try:
        import fiftyone

        print("✓ fiftyone available (optional)")
    except ImportError:
        print("○ fiftyone not available (optional)")


def test_gpu_availability():
    """Test GPU availability"""
    print("\nTesting GPU availability...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
        else:
            print("○ CUDA not available (CPU-only mode)")
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")


def test_project_imports():
    """Test project-specific imports"""
    print("\nTesting project imports...")

    try:
        import utils

        print("✓ utils module available")
    except ImportError:
        print("✗ utils module not available")
        return False

    try:
        import config

        print("✓ config module available")
    except ImportError:
        print("✗ config module not available")
        return False

    return True


def main():
    """Run all tests"""
    print("AIC_FTML_dev Dependency Test Suite")
    print("=" * 50)

    all_tests_passed = True

    # Core tests
    all_tests_passed &= test_core_imports()
    all_tests_passed &= test_cv_imports()
    all_tests_passed &= test_data_imports()
    all_tests_passed &= test_ml_imports()
    all_tests_passed &= test_utility_imports()
    all_tests_passed &= test_project_imports()

    # Optional tests (don't affect pass/fail)
    test_optional_imports()
    test_gpu_availability()

    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All required dependencies are available!")
        print("The project should run successfully.")
    else:
        print("✗ Some required dependencies are missing!")
        print("Please run: python install_utils.py --all")

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())

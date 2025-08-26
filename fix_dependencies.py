#!/usr/bin/env python3
"""
Fix dependency conflicts in the environment.
"""

import subprocess
import sys

def run_command(cmd):
    """Run a shell command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False

def fix_numpy_conflict():
    """Fix numpy version conflicts."""
    print("Fixing numpy version conflicts...")
    
    # Uninstall conflicting packages
    print("\n1. Uninstalling conflicting OpenCV packages...")
    run_command([sys.executable, "-m", "pip", "uninstall", "-y", 
                 "opencv-contrib-python", "opencv-python", "opencv-python-headless"])
    
    # Install compatible numpy version
    print("\n2. Installing compatible numpy version...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "numpy>=1.24,<2.0"])
    
    # Reinstall OpenCV with compatible version
    print("\n3. Reinstalling OpenCV-headless (compatible version)...")
    run_command([sys.executable, "-m", "pip", "install", "opencv-python-headless>=4.8.0,<4.10"])
    
    # Fix open_clip if needed
    print("\n4. Reinstalling open-clip-torch...")
    run_command([sys.executable, "-m", "pip", "uninstall", "-y", "open-clip-torch"])
    run_command([sys.executable, "-m", "pip", "install", "open-clip-torch>=2.24.0"])
    
    print("\n✅ Dependencies fixed!")

def verify_installation():
    """Verify that key imports work."""
    print("\n5. Verifying installation...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except Exception as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except Exception as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import open_clip
        print("✓ open-clip-torch available")
    except Exception as e:
        print(f"✗ open-clip import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA: {torch.version.cuda}")
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import faiss
        print("✓ FAISS available")
        if hasattr(faiss, 'StandardGpuResources'):
            print("  GPU support: Available")
    except Exception as e:
        print(f"✗ FAISS import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("AIC_FTML Dependency Fixer")
    print("=" * 50)
    
    fix_numpy_conflict()
    
    if verify_installation():
        print("\n✅ All dependencies successfully fixed!")
        print("You can now run the pipeline without conflicts.")
    else:
        print("\n⚠️ Some issues remain. You may need to:")
        print("1. Restart your Python kernel/runtime")
        print("2. Run: pip install -r requirements.txt")
        print("3. Check for other conflicting packages")
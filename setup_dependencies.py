#!/usr/bin/env python3
"""
Enhanced setup script for AIC_FTML_dev project.
Handles both pip and conda installations with robust error handling.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def check_conda():
    """Check if conda is available"""
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_pip():
    """Check if pip is available"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def install_with_conda(gpu=False, force=False):
    """Install using conda environment"""
    print("Setting up conda environment...")
    
    env_file = "environment-gpu.yml" if gpu else "environment.yml"
    env_name = "aic-ftml-gpu" if gpu else "aic-ftml"
    
    if not Path(env_file).exists():
        print(f"ERROR: {env_file} not found!")
        return False
    
    # Check if environment exists
    try:
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        env_exists = env_name in result.stdout
    except Exception:
        env_exists = False
    
    if env_exists and not force:
        print(f"Environment '{env_name}' already exists.")
        response = input("Remove and recreate? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Using existing environment.")
            return True
    
    if env_exists:
        print(f"Removing existing environment: {env_name}")
        try:
            subprocess.run(['conda', 'env', 'remove', '-n', env_name, '-y'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove environment: {e}")
            return False
    
    # Create new environment
    print(f"Creating conda environment from {env_file}...")
    try:
        subprocess.run(['conda', 'env', 'create', '-f', env_file], check=True)
        print(f"Successfully created conda environment: {env_name}")
        print(f"\nTo activate: conda activate {env_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create conda environment: {e}")
        return False

def install_with_pip(gpu=False):
    """Install using pip"""
    print("Installing dependencies with pip...")
    
    # Ensure pip is up to date
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Failed to upgrade pip")
    
    # PyTorch installation (different for GPU/CPU)
    if gpu:
        pytorch_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch>=2.1', 'torchvision>=0.16.0', 'torchaudio>=2.1.0',
            '--index-url', 'https://download.pytorch.org/whl/cu121'
        ]
        print("Installing PyTorch with CUDA support...")
    else:
        pytorch_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch>=2.1', 'torchvision>=0.16.0', 'torchaudio>=2.1.0',
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ]
        print("Installing PyTorch for CPU...")
    
    try:
        subprocess.run(pytorch_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install PyTorch: {e}")
        return False
    
    # Install other dependencies
    if not Path('requirements.txt').exists():
        print("ERROR: requirements.txt not found!")
        return False
    
    print("Installing other dependencies from requirements.txt...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        return False
    
    print("Successfully installed all dependencies with pip!")
    return True

def install_with_install_utils(gpu=False):
    """Install using our custom install_utils"""
    print("Using install_utils for dependency management...")
    
    try:
        # Import and use install_utils
        from install_utils import install_all_dependencies
        success = install_all_dependencies(force_gpu=gpu)
        if success:
            print("Successfully installed all dependencies!")
        else:
            print("Failed to install some dependencies.")
        return success
    except ImportError:
        print("install_utils not available, falling back to pip...")
        return install_with_pip(gpu)

def main():
    parser = argparse.ArgumentParser(description="Setup AIC_FTML_dev dependencies")
    parser.add_argument("--method", choices=['conda', 'pip', 'auto'], default='auto',
                       help="Installation method")
    parser.add_argument("--gpu", action="store_true", 
                       help="Install GPU versions (CUDA support)")
    parser.add_argument("--cpu", action="store_true",
                       help="Install CPU-only versions")
    parser.add_argument("--force", action="store_true",
                       help="Force reinstall even if environment exists")
    parser.add_argument("--test", action="store_true",
                       help="Run dependency tests after installation")
    
    args = parser.parse_args()
    
    # Determine GPU/CPU preference
    if args.cpu:
        gpu_mode = False
    elif args.gpu:
        gpu_mode = True
    else:
        # Auto-detect GPU availability
        try:
            import torch
            gpu_mode = torch.cuda.is_available()
            if gpu_mode:
                print("CUDA detected, installing GPU versions")
            else:
                print("No CUDA detected, installing CPU versions")
        except ImportError:
            gpu_mode = False
            print("PyTorch not available, defaulting to CPU installation")
    
    # Choose installation method
    success = False
    
    if args.method == 'conda' or (args.method == 'auto' and check_conda()):
        print("Using conda for installation...")
        success = install_with_conda(gpu_mode, args.force)
    elif args.method == 'pip' or check_pip():
        print("Using pip for installation...")
        success = install_with_install_utils(gpu_mode)
    else:
        print("ERROR: Neither conda nor pip is available!")
        return 1
    
    if not success:
        print("Installation failed!")
        return 1
    
    # Run tests if requested
    if args.test:
        print("\nRunning dependency tests...")
        try:
            result = subprocess.run([sys.executable, 'test_installation.py'], check=True)
        except subprocess.CalledProcessError:
            print("Some tests failed, but installation completed.")
        except FileNotFoundError:
            print("test_installation.py not found, skipping tests.")
    
    print("\nSetup complete!")
    print("You can now run the smart pipeline:")
    print("  python scripts/smart_pipeline.py --help")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
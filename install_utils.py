#!/usr/bin/env python3
"""
Robust installation utilities for AIC_FTML_dev project.
Automatically installs missing dependencies with proper error handling.
"""

import importlib
import logging
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Dependency mapping: import_name -> (package_name, version_constraint)
DEPENDENCY_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    # Core ML/AI frameworks
    "torch": ("torch", ">=2.1"),
    "torchvision": ("torchvision", ">=0.16.0"),
    "torchaudio": ("torchaudio", ">=2.1.0"),
    "open_clip": ("open-clip-torch", ">=2.24.0"),
    # Computer Vision & Video Processing
    "cv2": ("opencv-python-headless", ">=4.8.0"),
    "PIL": ("Pillow", ">=10.0"),
    "decord": ("decord", ">=0.6.0"),
    # Fast similarity search
    "faiss": ("faiss-cpu", ">=1.7.4"),
    # Data Processing & Storage
    "pandas": ("pandas", ">=2.0"),
    "numpy": ("numpy", ">=1.24"),
    "scipy": ("scipy", ">=1.11.0"),
    "pyarrow": ("pyarrow", ">=14.0.0"),
    # Machine Learning
    "sklearn": ("scikit-learn", ">=1.4"),
    "joblib": ("joblib", ">=1.3"),
    # Text Processing & Search
    "rank_bm25": ("rank_bm25", ">=0.2.2"),
    # Utility & Infrastructure
    "tqdm": ("tqdm", ">=4.66"),
    "yaml": ("pyyaml", ">=6.0"),
    # Optional Dependencies
    "lightgbm": ("lightgbm", "==4.5.0"),
}

# Special case packages that might be installed differently
SPECIAL_PACKAGES = {
    "torch": {
        "install_cmd": [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch>=2.1",
            "torchvision>=0.16.0",
            "torchaudio>=2.1.0",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        "gpu_cmd": [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch>=2.1",
            "torchvision>=0.16.0",
            "torchaudio>=2.1.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
    }
}


def check_import(module_name: str, package_name: Optional[str] = None) -> bool:
    """
    Check if a module can be imported.

    Args:
        module_name: Name of the module to import
        package_name: Optional package name (for error reporting)

    Returns:
        True if import succeeds, False otherwise
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        pkg = package_name or module_name
        logger.debug(f"Module '{module_name}' not found (package: {pkg})")
        return False


def install_package(package_name: str, version: Optional[str] = None, force_gpu: bool = False) -> bool:
    """
    Install a package using pip.

    Args:
        package_name: Name of the package to install
        version: Version constraint (e.g., '>=2.0')
        force_gpu: Force GPU version if available

    Returns:
        True if installation succeeds, False otherwise
    """
    try:
        # Handle special packages
        if package_name == "torch" and package_name in SPECIAL_PACKAGES:
            if force_gpu:
                cmd = SPECIAL_PACKAGES[package_name]["gpu_cmd"]
            else:
                cmd = SPECIAL_PACKAGES[package_name]["install_cmd"]
            logger.info("Installing PyTorch ecosystem...")
        else:
            # Standard package installation
            package_spec = f"{package_name}{version}" if version else package_name
            cmd = [sys.executable, "-m", "pip", "install", package_spec]
            logger.info(f"Installing {package_spec}...")

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully installed {package_name}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error installing {package_name}: {e}")
        return False


def check_and_install(import_names: List[str], force_gpu: bool = False, optional_packages: List[str] = None) -> bool:
    """
    Check and install missing dependencies.

    Args:
        import_names: List of import names to check
        force_gpu: Force GPU versions where available
        optional_packages: List of optional packages (won't fail if can't install)

    Returns:
        True if all required dependencies are available, False otherwise
    """
    optional_packages = optional_packages or []
    missing_packages = []
    failed_installs = []

    # Check what's missing
    for import_name in import_names:
        if not check_import(import_name):
            if import_name in DEPENDENCY_MAP:
                package_name, version = DEPENDENCY_MAP[import_name]
                missing_packages.append((import_name, package_name, version))
            else:
                logger.warning(f"Unknown dependency: {import_name}")
                missing_packages.append((import_name, import_name, None))

    if not missing_packages:
        logger.info("All dependencies are already installed!")
        return True

    logger.info(f"Found {len(missing_packages)} missing dependencies")

    # Install missing packages
    installed_packages = set()
    for import_name, package_name, version in missing_packages:
        # Skip if we already installed this package (e.g., torch ecosystem)
        if package_name in installed_packages:
            continue

        is_optional = import_name in optional_packages
        success = install_package(package_name, version, force_gpu)

        if success:
            installed_packages.add(package_name)
            # Verify installation
            if not check_import(import_name):
                logger.warning(f"Package {package_name} installed but import {import_name} still fails")
                if not is_optional:
                    failed_installs.append(import_name)
        elif is_optional:
            logger.warning(f"Optional package {package_name} failed to install - continuing")
        else:
            failed_installs.append(import_name)

    if failed_installs:
        logger.error(f"Failed to install required dependencies: {failed_installs}")
        return False

    logger.info("All required dependencies are now available!")
    return True


def smart_install_for_script(script_name: str, force_gpu: bool = False) -> bool:
    """
    Install dependencies for specific scripts based on their known requirements.

    Args:
        script_name: Name of the script (e.g., 'smart_pipeline.py')
        force_gpu: Force GPU versions where available

    Returns:
        True if all dependencies are available, False otherwise
    """
    # Define script-specific dependencies
    script_dependencies = {
        "smart_pipeline.py": [
            "torch",
            "open_clip",
            "faiss",
            "cv2",
            "numpy",
            "pandas",
            "PIL",
            "tqdm",
            "sklearn",
            "scipy",
            "pyarrow",
        ],
        "frames_auto.py": ["torch", "open_clip", "cv2", "numpy", "sklearn", "scipy", "PIL", "tqdm", "decord"],
        "search.py": ["torch", "open_clip", "faiss", "rank_bm25", "numpy", "pandas", "joblib"],
        "train_reranker.py": ["torch", "open_clip", "faiss", "sklearn", "joblib"],
        "train_reranker_gbm.py": ["torch", "open_clip", "faiss", "sklearn", "joblib", "lightgbm"],
    }

    # Get base script name
    base_name = script_name.split("/")[-1]

    if base_name in script_dependencies:
        dependencies = script_dependencies[base_name]
        optional = ["lightgbm"] if base_name == "train_reranker_gbm.py" else []
        return check_and_install(dependencies, force_gpu, optional)
    else:
        # Default fallback - install core dependencies
        core_deps = ["torch", "open_clip", "faiss", "cv2", "numpy", "pandas", "PIL", "tqdm", "sklearn"]
        return check_and_install(core_deps, force_gpu)


def install_all_dependencies(force_gpu: bool = False) -> bool:
    """
    Install all project dependencies from requirements.txt equivalent.

    Args:
        force_gpu: Force GPU versions where available

    Returns:
        True if all dependencies are available, False otherwise
    """
    all_deps = list(DEPENDENCY_MAP.keys())
    optional = ["lightgbm"]
    return check_and_install(all_deps, force_gpu, optional)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Install AIC_FTML_dev dependencies")
    parser.add_argument("--gpu", action="store_true", help="Install GPU versions where available")
    parser.add_argument("--script", type=str, help="Install dependencies for specific script")
    parser.add_argument("--all", action="store_true", help="Install all dependencies")

    args = parser.parse_args()

    if args.script:
        success = smart_install_for_script(args.script, args.gpu)
    elif args.all:
        success = install_all_dependencies(args.gpu)
    else:
        # Default: install core dependencies
        core_deps = ["torch", "open_clip", "faiss", "cv2", "numpy", "pandas", "PIL", "tqdm"]
        success = check_and_install(core_deps, args.gpu)

    sys.exit(0 if success else 1)

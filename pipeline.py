#!/usr/bin/env python3
"""
New Unified Video Retrieval Pipeline
Replaces smart_pipeline.py with clean modular architecture
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline.unified_pipeline import main

if __name__ == "__main__":
    main()
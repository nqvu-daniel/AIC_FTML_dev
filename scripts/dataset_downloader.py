#!/usr/bin/env python3
"""
Wrapper entrypoint to keep notebooks/docs stable.
Delegates to utils/dataset_downloader.py
"""
import runpy
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("utils.dataset_downloader", run_name="__main__")


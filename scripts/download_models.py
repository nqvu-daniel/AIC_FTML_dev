#!/usr/bin/env python3
"""Wrapper entrypoint for utils/download_models.py"""
import runpy
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("utils.download_models", run_name="__main__")


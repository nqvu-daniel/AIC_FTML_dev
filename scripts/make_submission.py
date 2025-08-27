#!/usr/bin/env python3
"""Wrapper entrypoint for utils/make_submission.py"""
import runpy
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("utils.make_submission", run_name="__main__")


#!/usr/bin/env python3
import argparse
from pathlib import Path

def clean_features(dataset_root: Path, dry_run: bool = False) -> int:
    features_dir = dataset_root / "features"
    if not features_dir.exists():
        print(f"[INFO] No features directory at {features_dir}")
        return 0
    npys = list(features_dir.glob("*.npy"))
    if not npys:
        print(f"[INFO] No .npy feature files in {features_dir}")
        return 0
    print(f"[INFO] Found {len(npys)} precomputed feature files in {features_dir}")
    if dry_run:
        for p in npys[:10]:
            print(f"  would delete: {p.name}")
        if len(npys) > 10:
            print("  ...")
        return len(npys)
    deleted = 0
    for p in npys:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"[WARN] Failed to delete {p}: {e}")
    print(f"[OK] Deleted {deleted} .npy feature files")
    return deleted

def main():
    ap = argparse.ArgumentParser(description="Delete precomputed features to force recomputation with new model")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="List files without deleting")
    args = ap.parse_args()
    clean_features(args.dataset_root, dry_run=args.dry_run)

if __name__ == "__main__":
    main()


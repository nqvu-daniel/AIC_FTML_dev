#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


NEEDED_FILES = [
    Path("config.py"),
    Path("utils.py"),
    Path("src/retrieval/use.py"),
]


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="Assemble a minimal runnable pipeline folder with artifacts")
    ap.add_argument("--outdir", type=Path, default=Path("my_pipeline"))
    ap.add_argument("--artifact_dir", type=Path, default=Path("./artifacts"))
    ap.add_argument("--include_model", action="store_true", help="Include reranker*.joblib if present")
    ap.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    args = ap.parse_args()

    out = args.outdir
    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Copy minimal code files
    for f in NEEDED_FILES:
        if not f.exists():
            raise SystemExit(f"Missing required file: {f}")
        dst = out / f
        copy_file(f, dst)

    # Copy artifacts
    art = args.artifact_dir
    required = [art / "index.faiss", art / "mapping.parquet", art / "text_corpus.jsonl"]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")
    (out / "artifacts").mkdir(exist_ok=True)
    for p in required:
        copy_file(p, out / "artifacts" / p.name)

    if args.include_model:
        for p in sorted(art.glob("reranker*.joblib")):
            copy_file(p, out / "artifacts" / p.name)

    # Convenience: create submissions/ dir and a small README
    (out / "submissions").mkdir(exist_ok=True)
    readme = out / "README_RUN.md"
    readme.write_text(
        (
            "Minimal pipeline folder\n\n"
            "Usage (from this folder):\n\n"
            "  # Activate your environment first\n"
            "  # conda activate aic-ftml-gpu  (or aic-ftml)\n\n"
            "  # Run a query (Top-100 CSV in submissions/)\n"
            "  python src/retrieval/use.py --query \"your query\"\n\n"
            "Notes:\n"
            "- Artifacts are under ./artifacts.\n"
            "- use.py auto-loads reranker.joblib if present.\n"
        ),
        encoding="utf-8",
    )

    print(f"[OK] Assembled pipeline directory → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


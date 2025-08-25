#!/usr/bin/env python3
import argparse
from pathlib import Path
import tarfile


def main():
    ap = argparse.ArgumentParser(description="Package artifacts (index, mapping, corpus, model) into a tar.gz bundle")
    ap.add_argument("--artifact_dir", type=Path, default=Path("./artifacts"))
    ap.add_argument("--output", type=Path, default=Path("./artifacts_bundle.tar.gz"))
    ap.add_argument("--include_model", action="store_true", help="Include reranker*.joblib if found")
    args = ap.parse_args()

    art = args.artifact_dir
    required = [art / "index.faiss", art / "mapping.parquet", art / "text_corpus.jsonl"]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as tf:
        for p in required:
            tf.add(p, arcname=p.name)
        if args.include_model:
            for p in art.glob("reranker*.joblib"):
                tf.add(p, arcname=p.name)

    print(f"[OK] Wrote bundle → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


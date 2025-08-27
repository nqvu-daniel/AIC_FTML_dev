#!/usr/bin/env python3
"""
Train a lightweight learning-to-rank (reranker) model using existing artifacts.

Inputs:
- An artifacts directory containing:
  - `index.faiss` and `mapping.parquet` (vector index + metadata)
  - `bm25_index.json` (text corpus index)
- A training JSONL file with entries like:
  {"query": "text", "positives": [{"video_id":"L21_V001","frame_idx":2412}, ...]}

The trainer generates candidates via vector and text search, builds simple
features per candidate, fits a LogisticRegression classifier, and saves it to
`artifacts/reranker.joblib` by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd

# Local imports
from ..encoders.clip_encoder import CLIPTextEncoder
from ..indexing.vector_index import FAISSIndex
from ..indexing.text_index import BM25Index


def load_indexes(artifact_dir: Path) -> Tuple[FAISSIndex, BM25Index, pd.DataFrame, int]:
    artifact_dir = Path(artifact_dir)
    index_path = artifact_dir / "index.faiss"
    mapping_path = artifact_dir / "mapping.parquet"
    bm25_path = artifact_dir / "bm25_index.json"

    if not index_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(f"Missing index files: {index_path}, {mapping_path}")

    mapping_df = pd.read_parquet(mapping_path)

    # Determine embedding dim from pipeline_info.json if present
    emb_dim = 512
    info_path = artifact_dir / "pipeline_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            emb_dim = int(info.get("embedding_dim", emb_dim))
        except Exception:
            pass

    v_index = FAISSIndex(emb_dim)
    v_index.load(index_path, mapping_path)

    t_index = BM25Index()
    if bm25_path.exists():
        t_index.load(bm25_path)
    else:
        # empty fallback so training still runs
        print(f"Warning: text index not found at {bm25_path}; proceeding with vector-only features")

    return v_index, t_index, mapping_df, emb_dim


def parse_training_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def build_feature_row(
    video_id: str,
    frame_idx: int,
    vector_rank: int | None,
    vector_score: float | None,
    text_rank: int | None,
    text_score: float | None,
) -> List[float]:
    # Use large rank value for missing to keep features bounded
    vr = float(vector_rank if vector_rank is not None else 1e6)
    tr = float(text_rank if text_rank is not None else 1e6)
    vs = float(vector_score if vector_score is not None else 0.0)
    ts = float(text_score if text_score is not None else 0.0)
    both = 1.0 if (vector_rank is not None and text_rank is not None) else 0.0
    # Reciprocal rank fusion proxy using ranks only (k=60)
    rrf = (1.0 / (60.0 + (vector_rank if vector_rank is not None else 1e6))) + \
          (1.0 / (60.0 + (text_rank if text_rank is not None else 1e6)))
    return [vs, ts, vr, tr, both, rrf]


def train_reranker(
    artifact_dir: Path,
    train_jsonl: Path,
    topk: int = 200,
    max_queries: int | None = None,
    C: float = 1.0,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> joblib:
    # Load indexes and encoders
    v_index, t_index, mapping_df, _ = load_indexes(artifact_dir)
    text_encoder = CLIPTextEncoder()

    # Load training data
    rows = parse_training_jsonl(train_jsonl)
    if max_queries is not None:
        rows = rows[:max_queries]
    if not rows:
        raise ValueError("No training rows found in JSONL")

    X: List[List[float]] = []
    y: List[int] = []

    # Build quick lookup for positives per query
    for qi, row in enumerate(rows, 1):
        query = row.get("query", "")
        positives = {(p.get("video_id", ""), int(p.get("frame_idx", -1))) for p in row.get("positives", [])}
        if not query:
            continue

        # Encode text and search
        q_vec = text_encoder.process(query)
        v_dists, v_inds = v_index.search(q_vec, topk)

        # Gather vector candidates and their metadata
        vec_candidates: Dict[Tuple[str, int], Tuple[int, float]] = {}
        for rank, (dist, idx) in enumerate(zip(v_dists[0].tolist(), v_inds[0].tolist())):
            if idx < 0 or idx >= len(v_index.metadata):
                continue
            md = v_index.metadata[idx]
            key = (md.get("video_id", ""), int(md.get("frame_idx", 0)))
            # FAISS returns inner product; use as is (normalized vectors)
            vec_candidates[key] = (rank, float(dist))

        # Text candidates via BM25
        text_candidates: Dict[Tuple[str, int], Tuple[int, float]] = {}
        try:
            t_scores, t_inds = t_index.search(query, topk)
            for rank, (score, idx) in enumerate(zip(t_scores, t_inds)):
                if idx < 0 or idx >= len(t_index.documents):
                    continue
                doc = t_index.documents[idx]
                key = (doc.get("video_id", ""), int(doc.get("frame_idx", 0)))
                text_candidates[key] = (rank, float(score))
        except Exception:
            # If text index empty, skip
            pass

        # Union of candidates
        all_keys = set(vec_candidates.keys()) | set(text_candidates.keys())
        if not all_keys:
            continue

        for key in all_keys:
            vrank, vscore = vec_candidates.get(key, (None, None))
            trank, tscore = text_candidates.get(key, (None, None))
            feats = build_feature_row(key[0], key[1], vrank, vscore, trank, tscore)
            X.append(feats)
            y.append(1 if key in positives else 0)

    if not X:
        raise ValueError("No training features built; ensure indexes and training JSONL are valid")

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int32)

    # Train model
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        C=C,
        max_iter=200,
        n_jobs=-1,
        class_weight=class_weight,
        random_state=random_state,
    )
    clf.fit(X_arr, y_arr)

    print(f"[OK] trained reranker on {len(y_arr)} samples; pos={int(y_arr.sum())}, neg={int((1-y_arr).sum())}")
    return clf


def main():
    ap = argparse.ArgumentParser(description="Train reranker model from artifacts and training JSONL")
    ap.add_argument("--index_dir", type=Path, default=Path("./artifacts"))
    ap.add_argument("--train_jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="Output joblib path (default: <index_dir>/reranker.joblib)")
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--no_balanced", action="store_true", help="Disable class_weight=balanced")

    args = ap.parse_args()
    model = train_reranker(
        artifact_dir=args.index_dir,
        train_jsonl=args.train_jsonl,
        topk=args.topk,
        max_queries=args.max_queries,
        C=args.C,
        class_weight=None if args.no_balanced else "balanced",
    )

    out = args.out or (args.index_dir / "reranker.joblib")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"[OK] saved → {out}")


if __name__ == "__main__":
    raise SystemExit(main())


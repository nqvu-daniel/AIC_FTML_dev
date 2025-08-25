import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi

import config
from utils import from_parquet, load_faiss


def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z\u00C0-\u1EF9]+", " ", text)
    toks = [t for t in text.split() if len(t) > 1]
    return toks


def encode_text(model, tokenizer, device, text: str):
    tok = tokenizer([text]).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                t = model.encode_text(tok)
        else:
            t = model.encode_text(tok)
    t = t.float().cpu().numpy()
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    return t


def collect_candidates(query, mapping, index, bm25, tokens_list, raw_docs, top_dense=400, top_bm25=400):
    import open_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(
        config.MODEL_NAME, pretrained=config.MODEL_PRETRAINED, device=device
    )
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    qv = encode_text(model, tokenizer, device, query)
    D, I = index.search(qv, top_dense)
    dense_idx, dense_scores = I[0], D[0]

    q_tokens = simple_tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_idx = np.argsort(-bm25_scores)[:top_bm25]

    pool = list(set(dense_idx.tolist()) | set(bm25_idx.tolist()))
    rows = []
    for gid in pool:
        r = mapping.iloc[int(gid)]
        try:
            r_dense = np.where(dense_idx == gid)[0][0] + 1
            s_dense = float(dense_scores[r_dense - 1])
        except Exception:
            r_dense, s_dense = 10_000, 0.0
        try:
            r_bm25 = np.where(bm25_idx == gid)[0][0] + 1
            s_bm25 = float(bm25_scores[gid])
        except Exception:
            r_bm25, s_bm25 = 10_000, 0.0

        qset = set(q_tokens)
        dset = set(tokens_list[gid])
        overlap = len(qset & dset)

        rows.append(
            {
                "global_idx": int(gid),
                "video_id": r["video_id"],
                "frame_idx": int(r["frame_idx"]),
                "n": int(r["n"]),
                "dense_score": s_dense,
                "bm25_score": s_bm25,
                "rank_dense": r_dense,
                "rank_bm25": r_bm25,
                "token_overlap": overlap,
                "importance_score": float(r.get("importance_score", 1.0)),
            }
        )
    df = pd.DataFrame(rows)
    # neighbor consensus (local window)
    df = df.sort_values(["video_id", "n"]).reset_index(drop=True)
    consensus = []
    for _, row in df.iterrows():
        cnt = (
            df[(df["video_id"] == row["video_id"]) & (df["n"].between(row["n"] - 1, row["n"] + 1))].shape[0]
            - 1
        )
        consensus.append(cnt)
    df["neighbor_consensus"] = consensus
    return df


def dedup_temporal(df, radius=1):
    kept = []
    last = {}
    for _, row in df.sort_values("score", ascending=False).iterrows():
        key = row["video_id"]
        n = row["n"]
        if key in last and abs(n - last[key]) <= radius:
            continue
        kept.append(row)
        last[key] = n
    return pd.DataFrame(kept)


def slugify(text: str, maxlen: int = 48) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^0-9a-zA-Z\u00C0-\u1EF9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if len(text) > maxlen:
        text = text[:maxlen]
    return text or "query"


def maybe_download_model(url: str, outfile: Path):
    if not url:
        return False
    try:
        from urllib.request import urlopen, Request

        outfile.parent.mkdir(parents=True, exist_ok=True)
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as r:  # nosec - URL provided by user
            data = r.read()
        with open(outfile, "wb") as f:
            f.write(data)
        print(f"[OK] Downloaded model → {outfile}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to download model: {e}")
        return False


def _artifacts_present(index_dir: Path) -> bool:
    return (
        (index_dir / "mapping.parquet").exists()
        and (index_dir / "index.faiss").exists()
        and (index_dir / "text_corpus.jsonl").exists()
    )


def _maybe_download_and_unpack_bundle(bundle_url: str, index_dir: Path) -> bool:
    if not bundle_url:
        return False
    try:
        from urllib.request import urlopen, Request
        import tarfile
        import zipfile

        index_dir.mkdir(parents=True, exist_ok=True)
        tmp = index_dir / "bundle.tmp"
        req = Request(bundle_url, headers={"User-Agent": "Mozilla/5.0"})
        print(f"[INFO] Downloading artifacts bundle from {bundle_url} ...")
        with urlopen(req) as r:  # nosec - URL provided by user
            data = r.read()
        with open(tmp, "wb") as f:
            f.write(data)
        # Try to guess format and extract
        extracted = False
        try:
            if tarfile.is_tarfile(tmp):
                with tarfile.open(tmp) as tf:
                    tf.extractall(index_dir)
                extracted = True
            elif zipfile.is_zipfile(tmp):
                with zipfile.ZipFile(tmp) as zf:
                    zf.extractall(index_dir)
                extracted = True
        finally:
            tmp.unlink(missing_ok=True)
        if not extracted:
            print("[WARN] Unknown bundle format; expected .zip or .tar(.gz)")
            return False
        print(f"[OK] Extracted bundle to {index_dir}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to download/extract bundle: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="One-shot search: outputs Top-100 CSV for KIS")
    ap.add_argument("--query", required=True, help="Text query")
    ap.add_argument("--index_dir", type=Path, default=config.ARTIFACT_DIR)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--dedup_radius", type=int, default=1)
    ap.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to submissions/kis_<slug>.csv",
    )
    ap.add_argument("--model_path", type=Path, default=Path("./artifacts/reranker.joblib"))
    ap.add_argument("--model_url", type=str, default=None, help="Optional URL to download reranker if missing")
    ap.add_argument(
        "--bundle_url",
        type=str,
        default=None,
        help="Optional URL to a zip/tar bundle containing index.faiss, mapping.parquet, text_corpus.jsonl, and optionally reranker.joblib",
    )
    args = ap.parse_args()

    # Resolve outfile
    if args.outfile is None:
        slug = slugify(args.query)
        args.outfile = Path("submissions") / f"kis_{slug}.csv"

    # Check required artifacts
    mapping_path = args.index_dir / "mapping.parquet"
    index_path = args.index_dir / "index.faiss"
    corpus_path = args.index_dir / "text_corpus.jsonl"
    if not _artifacts_present(args.index_dir):
        # Try to fetch artifacts bundle if provided
        if args.bundle_url:
            _maybe_download_and_unpack_bundle(args.bundle_url, args.index_dir)
        if not _artifacts_present(args.index_dir):
            raise FileNotFoundError(
                f"Artifacts missing in {args.index_dir}. Expected: mapping.parquet, index.faiss, text_corpus.jsonl.\n"
                "Provide --bundle_url to auto-download a prepared bundle for this competition,"
                " or run the pipeline locally (dev use)."
            )

    # Load artifacts
    mapping = from_parquet(mapping_path).reset_index(drop=True)
    index = load_faiss(index_path)
    raw_docs, tokens_list = [], []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            raw_docs.append(j["raw"])
            tokens_list.append(j["tokens"])
    bm25 = BM25Okapi(tokens_list)

    # Collect features
    feats = collect_candidates(
        args.query, mapping, index, bm25, tokens_list, raw_docs, top_dense=400, top_bm25=400
    )

    # Load or download reranker
    model = None
    if not args.model_path.exists() and args.model_url:
        maybe_download_model(args.model_url, args.model_path)
    if args.model_path.exists():
        bundle = joblib.load(args.model_path)
        model = bundle["model"]
        feat_names = bundle.get(
            "feature_names",
            ["dense_score", "bm25_score", "rank_dense", "rank_bm25", "token_overlap", "neighbor_consensus", "importance_score"],
        )
        # Ensure features present
        for feat in feat_names:
            if feat not in feats.columns:
                feats[feat] = 1.0 if feat == "importance_score" else 0.0
        X = feats[feat_names].values
        probs = model.predict_proba(X)[:, 1]
        feats["score"] = probs
    else:
        # Fall back to RRF-like using ranks
        k = 60
        feats["score"] = 1.0 / (k + feats["rank_dense"]) + 1.0 / (k + feats["rank_bm25"])

    # Rank, dedup, select
    feats = feats.sort_values("score", ascending=False)
    feats = dedup_temporal(feats, radius=args.dedup_radius).head(args.topk)

    # Write CSV
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    feats[["video_id", "frame_idx"]].to_csv(args.outfile, header=False, index=False)
    print(f"[OK] wrote {len(feats)} lines → {args.outfile}")


if __name__ == "__main__":
    main()

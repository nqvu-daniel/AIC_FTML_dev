
import argparse, json, re, numpy as np, pandas as pd, torch, faiss, open_clip, joblib
from pathlib import Path
from rank_bm25 import BM25Okapi

from utils import load_faiss, from_parquet
import config

def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z\u00C0-\u1EF9]+", " ", text)
    toks = [t for t in text.split() if len(t) > 1]
    return toks

def encode_text(model, tokenizer, device, text: str):
    tok = tokenizer([text]).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
        t = model.encode_text(tok)
    t = t.float().cpu().numpy()
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    return t

def rrf_fuse(rank_lists, k=60):
    scores = {}
    for lst in rank_lists:
        for r, gid in enumerate(lst):
            scores[gid] = scores.get(gid, 0.0) + 1.0 / (k + r + 1)
    return scores

def collect_candidates(query, mapping, index, bm25, tokens_list, raw_docs, top_dense=400, top_bm25=400):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained=config.MODEL_PRETRAINED, device=device)
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
            s_dense = float(dense_scores[r_dense-1])
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

        rows.append({
            "global_idx": int(gid),
            "video_id": r["video_id"],
            "frame_idx": int(r["frame_idx"]),
            "n": int(r["n"]),
            "dense_score": s_dense,
            "bm25_score": s_bm25,
            "rank_dense": r_dense,
            "rank_bm25": r_bm25,
            "token_overlap": overlap,
        })
    df = pd.DataFrame(rows)
    # neighbor consensus
    df = df.sort_values(["video_id","n"])
    consensus = []
    for _, row in df.iterrows():
        cnt = df[(df["video_id"]==row["video_id"]) & (df["n"].between(row["n"]-1, row["n"]+1))].shape[0] - 1
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=Path, default=config.ARTIFACT_DIR)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--dense_top", type=int, default=400)
    ap.add_argument("--bm25_top", type=int, default=400)
    ap.add_argument("--dedup_radius", type=int, default=1)
    ap.add_argument("--model_path", type=Path, default=Path("./artifacts/reranker.joblib"))
    args = ap.parse_args()

    mapping = from_parquet(args.index_dir / "mapping.parquet").reset_index(drop=True)
    index = load_faiss(args.index_dir / "index.faiss")
    corpus_path = args.index_dir / "text_corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} not found; run build_text.py first.")

    raw_docs, tokens_list = [], []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            raw_docs.append(j["raw"])
            tokens_list.append(j["tokens"])

    bm25 = BM25Okapi(tokens_list)

    # gather candidates & features
    feats = collect_candidates(args.query, mapping, index, bm25, tokens_list, raw_docs, top_dense=args.dense_top, top_bm25=args.bm25_top)
    # try learned model
    model = None
    if args.model_path.exists():
        bundle = joblib.load(args.model_path)
        model = bundle["model"]
        feat_names = bundle["feature_names"]
        X = feats[feat_names].values
        probs = model.predict_proba(X)[:,1]
        feats["score"] = probs
    else:
        # fallback to RRF with k=60 using the ranks we already have
        k = 60
        feats["score"] = 1.0/(k + feats["rank_dense"]) + 1.0/(k + feats["rank_bm25"])

    feats = feats.sort_values("score", ascending=False)
    feats = dedup_temporal(feats, radius=args.dedup_radius).head(args.topk)
    cols = ["video_id","frame_idx","n","score"]
    print(feats[cols].to_string(index=False))

if __name__ == "__main__":
    main()

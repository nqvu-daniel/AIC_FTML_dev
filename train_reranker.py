
import argparse, json, re, numpy as np, pandas as pd, torch, faiss, open_clip, joblib
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.linear_model import LogisticRegression

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

def build_bm25_and_docs(corpus_path: Path):
    raw_docs, tokens_list = [], []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            raw_docs.append(j["raw"])
            tokens_list.append(j["tokens"])
    bm25 = BM25Okapi(tokens_list)
    return bm25, raw_docs, tokens_list

def neighbor_consensus(doc_ids, window=1):
    # reward if neighbors with close n also appear among top candidates
    s = set(doc_ids)
    # approximate by counting same-video adjacency via doc order; refined using mapping outside
    return None  # computed later with mapping

def collect_features_for_query(query, mapping, index, bm25, tokens_list, raw_docs, top_dense=400, top_bm25=400):
    # dense
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained=config.MODEL_PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    qv = encode_text(model, tokenizer, device, query)
    D, I = index.search(qv, top_dense)
    dense_idx, dense_scores = I[0], D[0]

    # bm25
    q_tokens = simple_tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_idx = np.argsort(-bm25_scores)[:top_bm25]

    # candidate pool
    pool = list(set(dense_idx.tolist()) | set(bm25_idx.tolist()))
    features = []
    for gid in pool:
        r = mapping.iloc[int(gid)]
        # positions (if absent, set large rank)
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

        # token overlap
        qset = set(q_tokens)
        dset = set(tokens_list[gid])
        overlap = len(qset & dset)

        features.append({
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
    # neighbor consensus (need mapping by (video_id,n))
    df = pd.DataFrame(features)
    df = df.sort_values(["video_id","n"])
    consensus = []
    for i, row in df.iterrows():
        # count neighbors within ±1 n that appear in pool for same video
        cnt = df[(df["video_id"]==row["video_id"]) & (df["n"].between(row["n"]-1, row["n"]+1))].shape[0] - 1
        consensus.append(cnt)
    df["neighbor_consensus"] = consensus
    return df

def label_candidates(df_feats, positives):
    # positives: list of dicts with either (video_id, frame_idx) OR (video_id, u, v)
    pos_exact = {(p["video_id"], int(p["frame_idx"])) for p in positives if "frame_idx" in p}
    pos_ranges = [(p["video_id"], int(p["u"]), int(p["v"])) for p in positives if "u" in p and "v" in p]

    labels = []
    for _, r in df_feats.iterrows():
        key = (r["video_id"], int(r["frame_idx"]))
        y = 1 if key in pos_exact else 0
        if not y:
            for vid, u, v in pos_ranges:
                if r["video_id"] == vid and u <= int(r["frame_idx"]) <= v:
                    y = 1; break
        labels.append(y)
    df_feats["label"] = labels
    return df_feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=Path, default=config.ARTIFACT_DIR)
    ap.add_argument("--train_jsonl", type=Path, required=True, help="JSONL with fields: query, positives=[{video_id, frame_idx}|{video_id,u,v}]")
    ap.add_argument("--dense_top", type=int, default=400)
    ap.add_argument("--bm25_top", type=int, default=400)
    ap.add_argument("--outfile", type=Path, default=Path("./artifacts/reranker.joblib"))
    args = ap.parse_args()

    mapping = from_parquet(args.index_dir / "mapping.parquet").reset_index(drop=True)
    index = load_faiss(args.index_dir / "index.faiss")

    corpus_path = args.index_dir / "text_corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} not found; run build_text.py first.")
    bm25, raw_docs, tokens_list = build_bm25_and_docs(corpus_path)

    X, y = [], []
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            q = j["query"]
            pos = j.get("positives", [])
            feats = collect_features_for_query(q, mapping, index, bm25, tokens_list, raw_docs, top_dense=args.dense_top, top_bm25=args.bm25_top)
            feats = label_candidates(feats, pos)
            # keep positives and hard negatives (top 200 by dense/bm25)
            feats = feats.sort_values(["dense_score","bm25_score"], ascending=False).head(600)
            # feature matrix
            cols = ["dense_score","bm25_score","rank_dense","rank_bm25","token_overlap","neighbor_consensus"]
            X.extend(feats[cols].values.tolist())
            y.extend(feats["label"].values.tolist())

    X = np.array(X); y = np.array(y)
    # simple, robust model
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(X, y)
    from joblib import dump
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": clf, "feature_names": ["dense_score","bm25_score","rank_dense","rank_bm25","token_overlap","neighbor_consensus"]}, args.outfile)
    print(f"[OK] trained reranker on {len(y)} samples; saved → {args.outfile}")

if __name__ == "__main__":
    main()

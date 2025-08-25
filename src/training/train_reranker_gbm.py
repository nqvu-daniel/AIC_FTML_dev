
import argparse, json, re, numpy as np, pandas as pd, torch, open_clip, joblib
from pathlib import Path
from rank_bm25 import BM25Okapi
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from utils import load_faiss, from_parquet
import config

def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z\u00C0-\u1EF9]+", " ", text)
    toks = [t for t in text.split() if len(t) > 1]
    return toks

def encode_text(model, tokenizer, device, text: str):
    tok = tokenizer([text]).to(device)
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                t = model.encode_text(tok)
        else:
            t = model.encode_text(tok)
    t = t.float().cpu().numpy()
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    return t

def collect_features(query, mapping, index, bm25, tokens_list, top_dense=600, top_bm25=600):
    # dense
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(getattr(config, "ADV_MODEL_DEFAULT", "ViT-L-14"), pretrained="laion2b_s32b_b82k", device=device)
    tokenizer = open_clip.get_tokenizer(getattr(config, "ADV_MODEL_DEFAULT", "ViT-L-14"))
    qv = encode_text(model, tokenizer, device, query)
    import faiss
    D, I = index.search(qv, top_dense)
    dense_idx, dense_scores = I[0], D[0]
    # bm25
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
        qset = set(q_tokens); dset = set(tokens_list[gid])
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
            "importance_score": float(r.get("importance_score", 1.0)),  # Add importance if available
        })
    df = pd.DataFrame(rows).sort_values(["video_id","n"])
    # neighbor consensus
    df["neighbor_consensus"] = 0
    for vid, grp in df.groupby("video_id"):
        ns = grp["n"].values
        idxs = grp.index.values
        for i, n in zip(idxs, ns):
            cnt = np.sum((ns >= n-1) & (ns <= n+1)) - 1
            df.loc[i, "neighbor_consensus"] = cnt
    return df

def label(df, positives):
    pos_exact = {(p["video_id"], int(p["frame_idx"])) for p in positives if "frame_idx" in p}
    pos_ranges = [(p["video_id"], int(p["u"]), int(p["v"])) for p in positives if "u" in p and "v" in p]
    ys = []
    for _, r in df.iterrows():
        y = 1 if (r["video_id"], int(r["frame_idx"])) in pos_exact else 0
        if not y:
            for vid,u,v in pos_ranges:
                if r["video_id"]==vid and u<=int(r["frame_idx"])<=v:
                    y=1; break
        ys.append(y)
    df["label"]=ys
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=Path, required=True)
    ap.add_argument("--train_jsonl", type=Path, required=True)
    ap.add_argument("--outfile", type=Path, default=Path("./artifacts/reranker_gbm.joblib"))
    args = ap.parse_args()

    mapping = from_parquet(args.index_dir / "mapping.parquet").reset_index(drop=True)
    import faiss
    index = faiss.read_index(str(args.index_dir / "index.faiss"))
    # bm25 corpus
    raw_docs, tokens_list = [], []
    with open(args.index_dir / "text_corpus.jsonl","r",encoding="utf-8") as f:
        for line in f:
            j=json.loads(line); raw_docs.append(j["raw"]); tokens_list.append(j["tokens"])
    bm25 = BM25Okapi(tokens_list)

    X, y = [], []
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            feats = collect_features(j["query"], mapping, index, bm25, tokens_list)
            feats = label(feats, j.get("positives", []))
            feats = feats.sort_values(["dense_score","bm25_score"], ascending=False).head(800)
            cols = ["dense_score","bm25_score","rank_dense","rank_bm25","token_overlap","neighbor_consensus","importance_score"]
            X.extend(feats[cols].values.tolist()); y.extend(feats["label"].values.tolist())
    import numpy as np
    X = np.array(X); y = np.array(y)
    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=400, class_weight="balanced")
    clf.fit(X, y)
    from joblib import dump
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": clf, "feature_names": ["dense_score","bm25_score","rank_dense","rank_bm25","token_overlap","neighbor_consensus","importance_score"]}, args.outfile)
    print(f"[OK] trained GBM reranker on {len(y)} samples; saved -> {args.outfile}")

if __name__=="__main__":
    main()

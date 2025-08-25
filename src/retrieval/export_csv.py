import argparse, json, numpy as np, pandas as pd, torch, faiss, open_clip, re
from pathlib import Path
from rank_bm25 import BM25Okapi

from utils import load_faiss, from_parquet
import config

def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z\u00C0-\u1EF9]+", " ", text)
    toks = [t for t in text.split() if len(t) > 1]
    return toks

def rrf_fuse(rank_lists, k=60):
    scores = {}
    for lst in rank_lists:
        for r, gid in enumerate(lst):
            scores[gid] = scores.get(gid, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

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

def dedup_temporal(df, radius=1):
    kept = []
    last = {}
    for _, row in df.iterrows():
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
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--answer", type=str, default=None)
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--bm25_top", type=int, default=400)
    ap.add_argument("--dense_top", type=int, default=400)
    ap.add_argument("--dedup_radius", type=int, default=1)
    args = ap.parse_args()

    # load mapping & dense index
    mapping = from_parquet(args.index_dir / "mapping.parquet").reset_index(drop=True)
    index = load_faiss(args.index_dir / "index.faiss")

    # load text corpus
    corpus_path = args.index_dir / "text_corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} not found. Run build_text.py first.")
    raw_docs, tokens_list = [], []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            raw_docs.append(j["raw"])
            tokens_list.append(j["tokens"])

    bm25 = BM25Okapi(tokens_list)
    q_tokens = simple_tokenize(args.query)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top_idx = np.argsort(-bm25_scores)[:args.bm25_top]

    # dense
    import open_clip, torch, numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained=config.MODEL_PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    qv = encode_text(model, tokenizer, device, args.query)
    D, I = index.search(qv, args.dense_top)
    dense_top_idx = I[0]

    # fuse
    fused = rrf_fuse([dense_top_idx, bm25_top_idx], k=args.rrf_k)
    # build list and dedup
    import pandas as pd
    rows = []
    for gid, score in fused[:args.dense_top + args.bm25_top]:
        r = mapping.iloc[int(gid)].to_dict()
        r["fused_score"] = float(score)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("fused_score", ascending=False)
    df = dedup_temporal(df, radius=args.dedup_radius).head(100)

    # write CSV
    if args.answer is None:
        outdf = df[["video_id","frame_idx"]]
    else:
        outdf = df[["video_id","frame_idx"]].copy()
        outdf["answer"] = args.answer

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(args.outfile, header=False, index=False)
    print(f"[OK] wrote {len(outdf)} lines → {args.outfile}")

if __name__ == "__main__":
    main()

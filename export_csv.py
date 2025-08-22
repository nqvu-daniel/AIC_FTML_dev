import argparse, pandas as pd
from pathlib import Path
import subprocess, sys, numpy as np, torch, faiss, open_clip

from utils import load_faiss, from_parquet
import config

def run_search(index_dir: Path, query: str, topk: int = 100):
    index = load_faiss(index_dir / "index.faiss")
    mapping = from_parquet(index_dir / "mapping.parquet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained=config.MODEL_PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)

    # text emb
    tok = tokenizer([query]).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
        t = model.encode_text(tok)
    t = t.float().cpu().numpy()
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)

    D, I = index.search(t, topk)
    hits = mapping.iloc[I[0]].copy()
    hits["score"] = D[0]
    hits = hits.sort_values("score", ascending=False)
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=Path, default=config.ARTIFACT_DIR)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--answer", type=str, default=None, help="Optional Q&A digit string to include")
    args = ap.parse_args()

    hits = run_search(args.index_dir, args.query, topk=100)

    # Build CSV columns per challenge spec
    if args.answer is None:
        df = hits[["video_id","frame_idx"]].copy()
    else:
        df = hits[["video_id","frame_idx"]].copy()
        df["answer"] = args.answer

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outfile, header=False, index=False)
    print(f"[OK] wrote {len(df)} lines → {args.outfile}")

if __name__ == "__main__":
    main()

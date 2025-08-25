import argparse, numpy as np, pandas as pd, torch, faiss
from pathlib import Path
from tqdm import tqdm
import open_clip

from utils import load_faiss, from_parquet
import config

def encode_text(model, tokenizer, device, text: str):
    tok = tokenizer([text]).to(device)
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                t = model.encode_text(tok)
        else:
            t = model.encode_text(tok)
    t = t.float().cpu().numpy()
    # normalize
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    return t

def dedup_temporal(df, radius=1):
    # Deduplicate neighbors: keep the highest score within same video around nearby n
    kept = []
    last = {}
    for _, row in df.iterrows():
        key = row["video_id"]
        n = row["n"]
        if key in last and abs(n - last[key]) <= radius:
            # already placed a neighbor nearby
            continue
        kept.append(row)
        last[key] = n
    return pd.DataFrame(kept)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=Path, default=config.ARTIFACT_DIR)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--default-clip", action="store_true", help="Use default ViT-B-32 CLIP (512D) to match 512D indexes")
    ap.add_argument("--experimental", action="store_true", help="Enable experimental model selection (advanced backbones)")
    ap.add_argument("--exp-model", type=str, default=None, help="Experimental model name or preset key (see config.EXPERIMENTAL_PRESETS)")
    ap.add_argument("--exp-pretrained", type=str, default=None, help="Override pretrained tag for experimental model")
    ap.add_argument("--dedup_radius", type=int, default=1, help="suppress close-by keyframes in the same video")
    args = ap.parse_args()

    index = load_faiss(args.index_dir / "index.faiss")
    mapping = from_parquet(args.index_dir / "mapping.parquet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.default_clip:
        model_name = getattr(config, "DEFAULT_CLIP_MODEL", "ViT-B-32")
        pretrained = getattr(config, "DEFAULT_CLIP_PRETRAINED", "openai")
    elif args.experimental:
        preset_key = (args.exp_model or '').lower() if args.exp_model else None
        picked = None
        if preset_key and preset_key in getattr(config, 'EXPERIMENTAL_PRESETS', {}):
            picked = config.EXPERIMENTAL_PRESETS[preset_key]
        elif preset_key:
            model_name = args.exp_model
            pretrained = args.exp_pretrained or getattr(config, 'MODEL_PRETRAINED', 'openai')
        else:
            for key in getattr(config, 'EXPERIMENTAL_FALLBACK_ORDER', []):
                if key in config.EXPERIMENTAL_PRESETS:
                    picked = config.EXPERIMENTAL_PRESETS[key]
                    break
        if picked:
            model_name, pretrained = picked
        if args.exp_pretrained:
            pretrained = args.exp_pretrained
    else:
        model_name = config.MODEL_NAME
        pretrained = config.MODEL_PRETRAINED
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)

    qv = encode_text(model, tokenizer, device, args.query)
    D, I = index.search(qv, args.topk * 3)  # over-fetch for dedup
    I = I[0]; D = D[0]

    hits = mapping.iloc[I].copy()
    hits["score"] = D
    hits = hits.sort_values("score", ascending=False)
    if args.dedup_radius > 0:
        hits = dedup_temporal(hits, radius=args.dedup_radius)
        hits = hits.head(args.topk)
    else:
        hits = hits.head(args.topk)

    cols = ["video_id","frame_idx","n","pts_time","keyframe_path","score"]
    print(hits[cols].to_string(index=False))

if __name__ == "__main__":
    main()

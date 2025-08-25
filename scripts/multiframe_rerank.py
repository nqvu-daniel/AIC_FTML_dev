
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, open_clip, cv2
from PIL import Image
from utils import from_parquet
import config

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def encode_imgs(model, preprocess, device, imgs):
    # imgs: list of RGB np arrays
    tensors = []
    for im in imgs:
        if im is None:
            continue
        pil = Image.fromarray(im)
        tensors.append(preprocess(pil))
    if not tensors:
        return None
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feats = model.encode_image(batch)
        else:
            feats = model.encode_image(batch)
    feats = feats.float()
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)
    return feats.cpu().numpy()

def encode_text(model, tokenizer, device, text):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=Path, required=True)
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--cand_csv", type=Path, required=True, help="CSV with columns [video_id,frame_idx,n,score] from search_hybrid(_rerank)")
    ap.add_argument("--window", type=int, default=2, help="frames on each side of n to aggregate")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--model_name", type=str, default=getattr(config, "ADV_MODEL_DEFAULT", "ViT-L-14"))
    args = ap.parse_args()

    mapping = from_parquet(args.index_dir / "mapping.parquet")
    mapping = mapping.set_index(["video_id","n"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained="laion2b_s32b_b82k", device=device)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    qv = encode_text(model, tokenizer, device, args.query)  # [1,d]
    qv_t = torch.from_numpy(qv)

    df = pd.read_csv(args.cand_csv)
    out = []
    for _, row in df.iterrows():
        vid = row["video_id"]; n = int(row["n"])
        imgs = []
        for dn in range(n-args.window, n+args.window+1):
            key = (vid, dn)
            if key not in mapping.index:
                continue
            m = mapping.loc[key]
            # Try intelligent keyframes first, then competition keyframes
            intelligent_path = args.dataset_root / "keyframes_intelligent" / vid / f"{int(m['n']):03d}.png"
            competition_path = args.dataset_root / "keyframes" / vid / f"{int(m['n']):03d}.png"
            
            if intelligent_path.exists():
                img_path = intelligent_path
            elif competition_path.exists():
                img_path = competition_path
            else:
                continue
            
            imgs.append(load_image(str(img_path)))
        feats = encode_imgs(model, preprocess, device, imgs)
        if feats is None: 
            agg = row["score"]
        else:
            # cosine similarities
            sims = (torch.from_numpy(feats) @ qv_t.T).squeeze(1).numpy()
            # robust aggregate: max + 0.5 * mean
            agg = float(np.max(sims) + 0.5 * np.mean(sims))
        out.append({**row.to_dict(), "mf_score": agg})

    out = pd.DataFrame(out).sort_values("mf_score", ascending=False).head(args.topk)
    cols = ["video_id","frame_idx","n","mf_score"]
    print(out[cols].to_string(index=False))

if __name__ == "__main__":
    main()

import argparse, os, json, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
import open_clip
from PIL import Image
import faiss

from utils import ensure_dir, load_image, normalize_rows, save_faiss, to_parquet, as_type
import config

def embed_keyframes(model, preprocess, device, kf_paths):
    embs = []
    bs = 64
    for i in tqdm(range(0, len(kf_paths), bs), desc="Embedding keyframes"):
        batch_paths = kf_paths[i:i+bs]
        imgs = [preprocess(load_image(p)).unsqueeze(0) for p in batch_paths]
        imgs = torch.cat(imgs).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
            img_feats = model.encode_image(imgs)
        img_feats = img_feats.float().cpu().numpy()
        embs.append(img_feats)
    return np.concatenate(embs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--videos", nargs="+", required=True, help="Video IDs to index, e.g., L21_V001 L22_V003")
    ap.add_argument("--use_precomputed", action="store_true", help="Use features/*.npy if present (clip-level), else compute from keyframes")
    ap.add_argument("--flat", action="store_true", help="Use exact IndexFlatIP instead of HNSW")
    args = ap.parse_args()

    root = args.dataset_root
    ART = config.ARTIFACT_DIR
    ensure_dir(ART)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained=config.MODEL_PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)

    all_vecs = []
    rows = []

    for vid in args.videos:
        map_csv = root / "meta" / f"{vid}.map_keyframe.csv"
        if not map_csv.exists():
            raise FileNotFoundError(map_csv)
        df = pd.read_csv(map_csv)
        # Expect columns: n, pts_time, fps, frame_idx
        kf_dir = root / "keyframes" / vid
        kf_paths = [(kf_dir / f"{int(n):03d}.png") for n in df["n"]]
        for p in kf_paths:
            if not p.exists():
                raise FileNotFoundError(p)

        if args.use_precomputed:
            feat_file = root / "features" / f"{vid}.npy"
            if not feat_file.exists():
                print(f"[WARN] {feat_file} not found; falling back to image embeddings")
                vecs = embed_keyframes(model, preprocess, device, kf_paths)
            else:
                vecs = np.load(feat_file)  # shape [T, D]; we assume T==len(kf_paths)
                if vecs.shape[0] != len(kf_paths):
                    print(f"[WARN] feature count {vecs.shape[0]} != keyframes {len(kf_paths)}; truncating to min")
                    m = min(vecs.shape[0], len(kf_paths))
                    vecs = vecs[:m]
                    df = df.iloc[:m]
                    kf_paths = kf_paths[:m]
        else:
            vecs = embed_keyframes(model, preprocess, device, kf_paths)

        vecs = normalize_rows(vecs)
        start_idx = sum(x.shape[0] for x in all_vecs)
        all_vecs.append(vecs)

        # mapping rows
        for i, (n, pts, fps, fidx, p) in enumerate(zip(df["n"], df["pts_time"], df["fps"], df["frame_idx"], kf_paths)):
            rows.append({
                "global_idx": start_idx + i,
                "video_id": vid,
                "n": int(n),
                "pts_time": float(pts),
                "fps": float(fps),
                "frame_idx": int(fidx),
                "keyframe_path": str(p)
            })

    X = np.concatenate(all_vecs, axis=0).astype("float32")
    # choose index
    if args.flat:
        index = faiss.IndexFlatIP(X.shape[1])
    else:
        index = faiss.IndexHNSWFlat(X.shape[1], 32)
        index.hnsw.efConstruction = 200
    index.add(X)

    # save
    faiss_path = config.ARTIFACT_DIR / "index.faiss"
    map_path = config.ARTIFACT_DIR / "mapping.parquet"
    save_faiss(index, faiss_path)
    to_parquet(pd.DataFrame(rows), map_path)
    print(f"[OK] Saved index → {faiss_path}")
    print(f"[OK] Saved mapping → {map_path}")
    print(f"[INFO] Total vectors: {X.shape[0]}")

if __name__ == "__main__":
    main()

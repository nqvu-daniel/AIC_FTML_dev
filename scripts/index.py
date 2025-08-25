import argparse, os, json, numpy as np, pandas as pd, torch, sys
from pathlib import Path
from tqdm import tqdm
import open_clip
from PIL import Image
import faiss

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils import ensure_dir, load_image, normalize_rows, save_faiss, to_parquet, as_type
import config

def embed_keyframes(model, preprocess, device, kf_paths):
    embs = []
    bs = 64
    for i in tqdm(range(0, len(kf_paths), bs), desc="Embedding keyframes"):
        batch_paths = kf_paths[i:i+bs]
        imgs = [preprocess(load_image(p)).unsqueeze(0) for p in batch_paths]
        imgs = torch.cat(imgs).to(device)
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    img_feats = model.encode_image(imgs)
            else:
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

    all_vecs = []
    rows = []

    for vid in args.videos:
        map_csv = root / "meta" / f"{vid}.map_keyframe.csv"
        if not map_csv.exists():
            raise FileNotFoundError(map_csv)
        df = pd.read_csv(map_csv)
        # Expect columns: n, pts_time, fps, frame_idx, [importance_score]
        
        # Collect keyframes from both directories (competition + intelligent)
        kf_paths = []
        importance_scores = []
        
        for _, row in df.iterrows():
            n = int(row["n"])
            
            # Try intelligent keyframes first, then competition keyframes
            intelligent_path = root / "keyframes_intelligent" / vid / f"{n:03d}.png"
            competition_path = root / "keyframes" / vid / f"{n:03d}.png"
            
            if intelligent_path.exists():
                kf_paths.append(intelligent_path)
            elif competition_path.exists():
                kf_paths.append(competition_path)
            else:
                raise FileNotFoundError(f"Keyframe {n:03d}.png not found for {vid}")
            
            # Get importance score if available
            importance = row.get("importance_score", 1.0)
            importance_scores.append(float(importance))

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

        # mapping rows with importance scores
        for i, (idx, row) in enumerate(df.iterrows()):
            rows.append({
                "global_idx": start_idx + i,
                "video_id": vid,
                "n": int(row["n"]),
                "pts_time": float(row["pts_time"]),
                "fps": float(row["fps"]),
                "frame_idx": int(row["frame_idx"]),
                "keyframe_path": str(kf_paths[i]),
                "importance_score": importance_scores[i]
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

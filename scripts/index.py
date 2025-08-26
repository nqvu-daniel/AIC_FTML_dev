import argparse
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

import config
from utils import ensure_dir, load_image, normalize_rows, save_faiss, to_parquet


def embed_keyframes(model, preprocess, device, kf_paths):
    embs = []
    bs = 64
    for i in tqdm(range(0, len(kf_paths), bs), desc="Embedding keyframes"):
        batch_paths = kf_paths[i : i + bs]
        imgs = [preprocess(load_image(p)).unsqueeze(0) for p in batch_paths]
        imgs = torch.cat(imgs).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    img_feats = model.encode_image(imgs)
            else:
                img_feats = model.encode_image(imgs)
        img_feats = img_feats.float().cpu().numpy()
        embs.append(img_feats)
    return np.concatenate(embs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Video collections to index, e.g., L21 L22, or specific videos L21_V001 L22_V003",
    )
    ap.add_argument(
        "--use_precomputed", action="store_true", default=True, help="Use features/*.npy if present (default: True)"
    )
    ap.add_argument("--no_precomputed", action="store_true", help="Force live computation, ignore precomputed features")
    ap.add_argument("--flat", action="store_true", help="Use exact IndexFlatIP instead of HNSW")
    ap.add_argument(
        "--default-clip",
        action="store_true",
        help="Use default ViT-B-32 CLIP model (512D, compatible with precomputed features)",
    )
    ap.add_argument(
        "--experimental", action="store_true", help="Enable experimental model selection (advanced backbones)"
    )
    ap.add_argument(
        "--exp-model",
        type=str,
        default=None,
        help="Experimental model name or preset key (see config.EXPERIMENTAL_PRESETS)",
    )
    ap.add_argument("--exp-pretrained", type=str, default=None, help="Override pretrained tag for experimental model")
    ap.add_argument(
        "--segments",
        type=Path,
        default=None,
        help="Optional segments.parquet to index representative frames per segment",
    )
    args = ap.parse_args()

    root = args.dataset_root
    ART = config.ARTIFACT_DIR
    ensure_dir(ART)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose model based on flags
    if hasattr(args, "default_clip") and args.default_clip:
        model_name = config.DEFAULT_CLIP_MODEL
        model_pretrained = config.DEFAULT_CLIP_PRETRAINED
        print(f"Using default CLIP model: {model_name} ({model_pretrained})")
    elif getattr(args, "experimental", False):
        # Experimental path: pick from preset or fallback order
        preset_key = (args.exp_model or "").lower() if args.exp_model else None
        picked = None
        if preset_key and preset_key in getattr(config, "EXPERIMENTAL_PRESETS", {}):
            picked = config.EXPERIMENTAL_PRESETS[preset_key]
        elif preset_key and preset_key not in getattr(config, "EXPERIMENTAL_PRESETS", {}):
            # Treat exp-model as raw model name; use provided or default pretrained
            model_name = args.exp_model
            model_pretrained = args.exp_pretrained or getattr(config, "MODEL_PRETRAINED", "openai")
            print(f"[EXPERIMENTAL] Using custom model: {model_name} ({model_pretrained})")
        else:
            # Try fallbacks in order
            for key in getattr(config, "EXPERIMENTAL_FALLBACK_ORDER", []):
                if key in config.EXPERIMENTAL_PRESETS:
                    picked = config.EXPERIMENTAL_PRESETS[key]
                    print(f"[EXPERIMENTAL] Selected preset '{key}': {picked[0]} ({picked[1]})")
                    break
        if picked:
            model_name, model_pretrained = picked
        # Override pretrained if user supplied
        if args.exp_pretrained and "model_pretrained" in locals():
            model_pretrained = args.exp_pretrained
    else:
        model_name = config.MODEL_NAME
        model_pretrained = config.MODEL_PRETRAINED
        print(f"Using configured model: {model_name} ({model_pretrained})")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_pretrained, device=device)

    # Get model embedding dimension for compatibility checks
    model_dim = model.visual.output_dim
    print(f"Model: {model_name} ({model_pretrained}) | embedding dim: {model_dim}")

    all_vecs = []
    rows = []

    # Expand video collections (L21 -> all L21_V* videos)
    all_video_ids = []
    for vid_arg in args.videos:
        if re.match(r"^L\d{2}$", vid_arg):  # Collection ID like L21
            # Find all videos in this collection
            map_keyframes_dir = root / "map_keyframes"
            if map_keyframes_dir.exists():
                collection_videos = sorted([f.stem for f in map_keyframes_dir.glob(f"{vid_arg}_V*.csv")])
                if collection_videos:
                    all_video_ids.extend(collection_videos)
                    print(f"Found {len(collection_videos)} videos in collection {vid_arg}")
                else:
                    print(f"Warning: No videos found for collection {vid_arg}")
            else:
                raise FileNotFoundError(f"map_keyframes directory not found: {map_keyframes_dir}")
        else:  # Specific video ID like L21_V001
            all_video_ids.append(vid_arg)

    if not all_video_ids:
        raise ValueError("No video IDs found to process")

    print(f"Processing {len(all_video_ids)} videos: {all_video_ids[:5]}{'...' if len(all_video_ids) > 5 else ''}")

    # Load segments if provided
    seg_df_all = None
    if args.segments and Path(args.segments).exists():
        try:
            seg_df_all = pd.read_parquet(args.segments)
        except Exception as e:
            print(f"[WARN] Failed to read segments file {args.segments}: {e}")
            seg_df_all = None

    for vid in all_video_ids:
        map_csv = root / "map_keyframes" / f"{vid}.csv"
        if not map_csv.exists():
            raise FileNotFoundError(map_csv)
        df = pd.read_csv(map_csv)
        # Expect columns: n, pts_time, fps, frame_idx, [importance_score]

        # If segments provided, select representative keyframes per segment by nearest frame_idx
        selected_rows = None
        seg_rows = []
        if seg_df_all is not None:
            seg_df = seg_df_all[seg_df_all["video_id"] == vid].copy()
            if not seg_df.empty and {"rep_frames", "seg_id"}.issubset(seg_df.columns):
                frame_list = df["frame_idx"].to_numpy()
                idx_by_frame = {int(f): i for i, f in enumerate(frame_list)}
                take_indices = []
                n_to_seg = {}
                for _, srow in seg_df.iterrows():
                    reps = srow["rep_frames"]
                    if isinstance(reps, str):
                        try:
                            import ast

                            reps = ast.literal_eval(reps)
                        except Exception:
                            reps = []
                    for rf in reps or []:
                        # find nearest keyframe by frame_idx
                        if int(rf) in idx_by_frame:
                            i = idx_by_frame[int(rf)]
                        else:
                            # nearest by absolute diff
                            i = int(np.argmin(np.abs(frame_list - int(rf))))
                        take_indices.append(i)
                        n_val = int(df.iloc[i]["n"])
                        n_to_seg[n_val] = int(srow["seg_id"])
                if take_indices:
                    take_indices = sorted(set(int(i) for i in take_indices))
                    selected_rows = df.iloc[take_indices].reset_index(drop=True)
                    # attach seg_id mapping later per row
                    seg_rows = [n_to_seg.get(int(r["n"]), -1) for _, r in selected_rows.iterrows()]

        # Collect keyframes from both directories (competition + intelligent)
        kf_paths = []
        importance_scores = []

        iter_df = selected_rows if selected_rows is not None else df
        for _, row in iter_df.iterrows():
            n = int(row["n"])

            # Try intelligent keyframes first, then competition keyframes (support multiple formats)
            intelligent_path = None
            competition_path = None

            for ext in [".png", ".jpg", ".jpeg"]:
                intelligent_candidate = root / "keyframes_intelligent" / vid / f"{n:03d}{ext}"
                competition_candidate = root / "keyframes" / vid / f"{n:03d}{ext}"

                if intelligent_candidate.exists():
                    intelligent_path = intelligent_candidate
                    break
                elif competition_candidate.exists():
                    competition_path = competition_candidate

            if intelligent_path:
                kf_paths.append(intelligent_path)
            elif competition_path:
                kf_paths.append(competition_path)
            else:
                raise FileNotFoundError(f"Keyframe {n:03d}.[png|jpg|jpeg] not found for {vid}")

            # Get importance score if available
            importance = row.get("importance_score", 1.0)
            importance_scores.append(float(importance))

        if args.use_precomputed and not args.no_precomputed:
            feat_file = root / "features" / f"{vid}.npy"
            if not feat_file.exists():
                print(f"[WARN] {feat_file} not found; falling back to image embeddings")
                vecs = embed_keyframes(model, preprocess, device, kf_paths)
            else:
                vecs = np.load(feat_file)  # shape [T, D]; we assume T==len(kf_paths)

                # Model compatibility check
                precomputed_dim = vecs.shape[1]
                if precomputed_dim != model_dim:
                    print("[ERROR] Precomputed feature dimension mismatch!")
                    print(f"  Precomputed: {precomputed_dim}D (from {feat_file})")
                    print(f"  Model: {model_dim}D ({model_name})")
                    print(f"  Falling back to live computation for {vid}")
                    vecs = embed_keyframes(model, preprocess, device, kf_paths)
                else:
                    print(f"[OK] Using precomputed features for {vid}: {vecs.shape} ({vecs.dtype})")
                    # Handle keyframe count mismatch
                    if vecs.shape[0] != len(kf_paths):
                        print(f"[WARN] Feature count {vecs.shape[0]} != keyframes {len(kf_paths)}; truncating to min")
                        m = min(vecs.shape[0], len(kf_paths))
                        vecs = vecs[:m]
                        iter_df = iter_df.iloc[:m]
                        kf_paths = kf_paths[:m]
        else:
            print(f"[COMPUTE] Generating embeddings for {vid} using {config.MODEL_NAME}")
            vecs = embed_keyframes(model, preprocess, device, kf_paths)

        vecs = normalize_rows(vecs)
        start_idx = sum(x.shape[0] for x in all_vecs)
        all_vecs.append(vecs)

        # mapping rows with importance scores
        for i, (idx, row) in enumerate(iter_df.iterrows()):
            mrow = {
                "global_idx": start_idx + i,
                "video_id": vid,
                "n": int(row["n"]),
                "pts_time": float(row["pts_time"]),
                "fps": float(row["fps"]),
                "frame_idx": int(row["frame_idx"]),
                "keyframe_path": str(kf_paths[i]),
                "importance_score": importance_scores[i],
            }
            # Attach seg_id if we used segments
            if selected_rows is not None and seg_rows:
                try:
                    mrow["seg_id"] = int(seg_rows[i])
                except Exception:
                    mrow["seg_id"] = -1
            rows.append(mrow)

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

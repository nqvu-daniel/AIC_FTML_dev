import argparse, json, re, pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

def simple_tokenize(text: str):
    # very lightweight tokenizer suitable for VI/EN mix
    text = text.lower()
    # keep letters/numbers, replace others with space
    text = re.sub(r"[^0-9a-zA-Z\u00C0-\u1EF9]+", " ", text)
    toks = [t for t in text.split() if len(t) > 1]
    return toks

def load_media_info(dataset_root: Path, vid: str):
    f = dataset_root / "media_info" / f"{vid}.json"
    title = desc = ""
    keywords = []
    if f.exists():
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
            title = j.get("title","") or ""
            desc = j.get("description","") or ""
            keywords = j.get("keywords",[]) or []
        except Exception:
            pass
    return title, desc, keywords

def collect_objects(objects_dir: Path, vid: str, n: int):
    # Check both possible object directories (for competition and intelligent keyframes)
    jf = objects_dir / vid / f"{n:03d}.json"
    if not jf.exists():
        # Try alternative naming or location if needed
        return []
    try:
        j = json.loads(jf.read_text(encoding="utf-8"))
        entities = j.get("detection_class_entities", []) or []
        scores = j.get("detection_scores", []) or []
        # keep top-10 by score (cast to float), with a small threshold
        pairs = sorted([(float(s), e) for s, e in zip(scores, entities)], reverse=True)
        ents = [e for s, e in pairs[:10] if s >= 0.15]
        return ents
    except Exception:
        return []

def _load_transcripts(transcripts_path: Path):
    if not transcripts_path or not transcripts_path.exists():
        return {}
    vid2texts: dict[str, list[str]] = {}
    with open(transcripts_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            vid = (j.get("video_id") or "").strip()
            if not vid:
                continue
            # Accept either segment-level entries with 'text' or raw 'transcript'
            txt = (j.get("text") or j.get("transcript") or "").strip()
            if not txt:
                continue
            vid2texts.setdefault(vid, []).append(txt)
    # Join per-video
    return {k: "\n".join(v) for k, v in vid2texts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--videos", nargs="+", required=True, help="Video collections, e.g., L21 L22, or specific videos L21_V001 L22_V003")
    ap.add_argument("--artifact_dir", type=Path, default=Path("./artifacts"))
    ap.add_argument("--transcripts", type=Path, default=None, help="Optional transcripts JSONL (WhisperX or similar); merged per-video into corpus")
    ap.add_argument("--segments", type=Path, default=None, help="Optional segments.parquet to include seg_id in corpus entries")
    args = ap.parse_args()

    obj_dir = args.dataset_root / "objects"
    out_jsonl = args.artifact_dir / "text_corpus.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Expand video collections (L21 -> all L21_V* videos)
    all_video_ids = []
    for vid_arg in args.videos:
        if re.match(r'^L\d{2}$', vid_arg):  # Collection ID like L21
            # Find all videos in this collection
            map_keyframes_dir = args.dataset_root / "map_keyframes"
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

    rows = []
    # Optional ASR transcripts merged per video
    transcripts_text = _load_transcripts(args.transcripts) if args.transcripts else {}
    # Optional segments
    seg_df_all = None
    if args.segments and args.segments.exists():
        try:
            seg_df_all = pd.read_parquet(args.segments)
        except Exception:
            seg_df_all = None

    for vid in tqdm(all_video_ids, desc="Building text corpus"):
        title, desc, kw = load_media_info(args.dataset_root, vid)
        # try to load mapping parquet created by index.py
        # if not present yet, fall back to CSV map to get list of n
        map_parquet = args.artifact_dir / "mapping.parquet"
        if map_parquet.exists():
            import pandas as pd
            df = pd.read_parquet(map_parquet)
            df = df[df["video_id"] == vid].copy()
            key_seq = list(zip(df["global_idx"], df["n"]))
        else:
            map_csv = args.dataset_root / "map_keyframes" / f"{vid}.csv"
            if not map_csv.exists():
                raise FileNotFoundError(map_csv)
            import pandas as pd
            df = pd.read_csv(map_csv)
            key_seq = list(zip(range(len(df)), df["n"]))

        extra_txt = transcripts_text.get(vid, "")
        # Optional seg_id mapping from mapping.parquet
        seg_map = {}
        if map_parquet.exists():
            if "seg_id" in df.columns:
                for gi, n_val, sid in zip(df["global_idx"], df["n"], df["seg_id"]):
                    try:
                        seg_map[int(gi)] = int(sid)
                    except Exception:
                        seg_map[int(gi)] = -1
        for global_idx, n in key_seq:
            ents = collect_objects(obj_dir, vid, int(n))
            raw_parts = [title, desc, " ".join(kw), " ".join(ents)]
            if extra_txt:
                raw_parts.append(extra_txt)
            raw = " ".join([p for p in raw_parts if p]).strip()
            tokens = simple_tokenize(raw)
            rec = {"global_idx": int(global_idx), "video_id": vid, "n": int(n), "raw": raw, "tokens": tokens}
            if seg_map:
                rec["seg_id"] = int(seg_map.get(int(global_idx), -1))
            rows.append(rec)

    # write jsonl
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            # tokens as list; json will store it as array
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(rows)} docs → {out_jsonl}")

if __name__ == "__main__":
    main()

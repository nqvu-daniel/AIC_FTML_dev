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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--artifact_dir", type=Path, default=Path("./artifacts"))
    args = ap.parse_args()

    obj_dir = args.dataset_root / "objects"
    out_jsonl = args.artifact_dir / "text_corpus.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for vid in tqdm(args.videos, desc="Building text corpus"):
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

        for global_idx, n in key_seq:
            ents = collect_objects(obj_dir, vid, int(n))
            raw = " ".join([title, desc, " ".join(kw), " ".join(ents)]).strip()
            tokens = simple_tokenize(raw)
            rows.append({"global_idx": int(global_idx), "video_id": vid, "n": int(n), "raw": raw, "tokens": tokens})

    # write jsonl
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            # tokens as list; json will store it as array
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(rows)} docs → {out_jsonl}")

if __name__ == "__main__":
    main()

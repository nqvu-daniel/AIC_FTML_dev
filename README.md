# AI-Challenge 2024 — Minimal Keyframe Retrieval (Dense + CSV Export)

This is a **competition-oriented**, single-GPU friendly toolkit to:
- Index keyframes with a CLIP-family model (OpenCLIP).
- Search top-100 frames for a **text query**.
- Export predictions as `video_id,frame_idx[,answer]` CSV for CodaLab.
- (Optional) Use precomputed features per video (e.g., `L21_V001.npy`) instead of re-embedding.

It assumes a dataset layout like:

```
dataset_root/
  videos/
    L21_V001.mp4
  keyframes/
    L21_V001/
      001.png 002.png ...   # keyframe images
  meta/
    L21_V001.map_keyframe.csv     # with columns: n, pts_time, fps, frame_idx
    L21_V001.media_info.json
    objects/
      L21_V001/
        001.json 002.json ...
  features/
    L21_V001.npy     # (optional) precomputed clip-level features (np.float32 [T, D])
```

> **Note**: This code is intentionally **lean** and not built for cluster-scale indexing. It avoids databases and uses FAISS files directly.

## Quickstart

```bash
# 1) Install (ideally in a fresh venv)
pip install -r requirements.txt

# 2) Build an index over keyframes (uses OpenCLIP ViT-B/32 by default)
python index.py --dataset_root /path/to/dataset_root --videos L21_V001 L22_V003

# 3) Search
python search.py --index_dir ./artifacts --query "đám cháy ở quận 8" --topk 100

# 4) Export CSV for a query bundle (single query demo)
python export_csv.py --index_dir ./artifacts --query "..." --outfile submission/query-1-kis.csv
```

To switch models or paths, edit `config.py`.

---

## What’s inside?

- `config.py` — central paths & model choice
- `utils.py` — helpers (image loading, FAISS I/O, tqdm wrappers)
- `index.py` — loads keyframes, computes embeddings (**or** uses `features/*.npy` if provided), builds FAISS, and writes a mapping parquet
- `search.py` — encodes a text query, retrieves top-k from FAISS, does light temporal de-dup
- `export_csv.py` — runs search and writes CodaLab-style CSV

You can later plug this into FiftyOne by loading the mapping parquet and image paths into a `fiftyone.Dataset`.


## New: Hybrid search (BM25 + Dense) with RRF
We now build a lightweight **text corpus** per keyframe using:
- `meta/{VID}.media_info.json` → title, description, keywords
- `meta/objects/{VID}/{n:03}.json` → top object labels

### Build the text corpus
```bash
python build_text.py --dataset_root /path/to/dataset_root --videos L21_V001
```

### Run hybrid search (prints fused results)
```bash
python search_hybrid.py --index_dir ./artifacts --query "cháy chung cư quận 8" --topk 100
```

### Export hybrid CSV for CodaLab
```bash
python export_csv_hybrid.py --index_dir ./artifacts --query "..." --outfile submission/query-1-kis.csv
```



## New: Learned re-ranker (logistic regression)
Train a compact model to combine dense & BM25 signals with simple context features.

**Training data JSONL format (one per line):**
```json
{"query": "mô tả sự kiện", "positives": [{"video_id":"L21_V001","frame_idx":2412}]}
{"query": "...", "positives": [{"video_id":"L21_V001","u":2400,"v":2450}]}
```

**Train:**
```bash
python train_reranker.py --index_dir ./artifacts --train_jsonl data/train_dev.jsonl
```

**Search with learned re-ranker:**
```bash
python search_hybrid_rerank.py --index_dir ./artifacts --query "..." --topk 100
```
If no `artifacts/reranker.joblib` is found, the script falls back to **RRF-only** fusion.



### Advanced options (3-month track)
- **Encoders**: switch to `EVA02-L-14` / `ViT-L-14-336` / SigLIP via `config.ADV_MODEL_DEFAULT`.
- **Multi-frame rerank**: `multiframe_rerank.py` to re-score each candidate using a ±W frame window.
- **Pseudo‑Relevance Feedback (PRF)**: `prf_expand.py` to expand the text query before retrieval.
- **GBM reranker**: `train_reranker_gbm.py` to learn a stronger re-ranker than logistic regression.
- **FiftyOne operator**: `plugins/aic_tools` adds a button to run hybrid+rerank and tag top matches.

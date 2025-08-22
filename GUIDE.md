
# Developer Guide — AI-Challenge Intelligent Retrieval System

Welcome! This guide is for a **first-timer** to get productive in under an hour.

---

## 0) TL;DR
- `frames_intelligent*.py` extracts **intelligent keyframes** to augment competition data (~70-90% compression).
- `index.py` builds a **dense** FAISS index over ALL keyframes (competition + intelligent).
- `build_text.py` builds a **light text corpus** per keyframe (media title/description/keywords + top object labels).
- `search_hybrid.py` runs **Dense + BM25** with **RRF fusion**.
- `search_hybrid_rerank.py` applies a **learned re-ranker** over the fused shortlist.
- `export_csv*.py` writes the **CodaLab CSV** per query.

---

## 1) Repo layout
```
ai_challenge_retriever/
  artifacts/                      # indices, mapping, text corpus, trained reranker
  frames_auto.py                  # basic uniform/shot sampling (baseline)
  frames_intelligent.py           # intelligent temporal window sampling (NEW)
  frames_intelligent_fast.py      # optimized version for large datasets (NEW)
  index.py                        # dense index builder
  build_text.py                   # text corpus builder
  search.py                       # dense-only search (baseline)
  search_hybrid.py                # dense + BM25 + RRF (no training)
  search_hybrid_rerank.py         # hybrid + learned re-ranker (training optional)
  export_csv.py                   # dense-only CSV export
  export_csv_hybrid.py            # hybrid CSV export
  multiframe_rerank.py            # temporal context re-scoring
  train_reranker_gbm.py           # GBM training
  prf_expand.py                   # query expansion
  config.py
  utils.py
  README.md
  GUIDE.md                        # ← you are here
```

---

## 2) Dataset layout (what the code expects)
```
dataset_root/
  videos/{VID}.mp4                     # original videos
  keyframes/{VID}/{n:03}.png           # competition-provided keyframes
  keyframes_intelligent/{VID}/{n:03}.png # our intelligent samples (NEW)
  meta/{VID}.map_keyframe.csv          # columns: n, pts_time, fps, frame_idx, [importance_score]
  meta/{VID}.media_info.json           # title, description, keywords (optional)
  meta/objects/{VID}/{n:03}.json       # detected objects (optional)
  features/{VID}.npy                   # optional precomputed features [T,D]
```

> The code maps **keyframe index `n` → frame_idx** via `map_keyframe.csv`.  
> CSV submission lines are **(video_id, frame_idx[, answer])** — no header.  
> **NEW:** Intelligent sampling adds importance scores to help prioritize frames.

---

## 3) Quickstart (intelligent sampling + dense + hybrid + reranker)
```bash
pip install -r requirements.txt

# Extract intelligent keyframes (NEW - do this first!)
python frames_intelligent.py --dataset_root /path/to/dataset --videos L21_V001 --mode intelligent
# Or for speed on large datasets:
# python frames_intelligent_fast.py --dataset_root /path/to/dataset --videos L21_V001 --mode ultra_fast

# Dense index (now indexes BOTH competition + intelligent keyframes)
python index.py --dataset_root /path/to/dataset_root --videos L21_V001

# Text corpus (media_info + objects)
python build_text.py --dataset_root /path/to/dataset_root --videos L21_V001

# Hybrid retrieval (prints fused result)
python search_hybrid.py --index_dir ./artifacts --query "mô tả sự kiện" --topk 100

# (Optional) Train learned re-ranker on your dev queries
# Provide a JSONL with one object per line:
# { "query": "text...", "positives": [{"video_id":"L21_V001","frame_idx":2412}, ...] }
python train_reranker.py --index_dir ./artifacts --train_jsonl data/train_dev.jsonl

# Hybrid + learned re-ranking
python search_hybrid_rerank.py --index_dir ./artifacts --query "mô tả sự kiện" --topk 100
```

---

## 4) What is the learned re-ranker?
We train a **small logistic regression** on a compact feature vector for each candidate frame:
- `dense_score` (CLIP similarity)
- `bm25_score` (lexical score)
- `rank_dense`, `rank_bm25` (reciprocal rank cues)
- `token_overlap` (query ∩ frame tokens)
- `neighbor_consensus` (how many near frames from same video appear in the shortlist)

This yields a probability that a frame matches the query. It’s **fast**, **data-light**, and beats naive fusion on most dev sets.

> If no trained model is found, the pipeline falls back to **RRF-only**.

---

## 5) SOTA knobs (safe defaults)
- **Model:** default is `ViT-L-14` (OpenCLIP/CLIP). If VRAM is tight, drop to `ViT-B-32` in `config.py`.
- **Index:** HNSW (M=32) works well for mid-scale; use FlatIP for exact testing.
- **Fusion:** RRF with `k=60` is strong without training. Keep it even with a learned re-ranker as a fallback.

---

## 6) Developer workflow
1. **Add a new video**: 
   - Place video in `videos/{VID}.mp4`
   - Run intelligent sampling: `frames_intelligent.py --videos {VID}`
   - Competition keyframes go in `keyframes/{VID}/`
   - Intelligent frames saved to `keyframes_intelligent/{VID}/`
   - Mapping saved to `meta/{VID}.map_keyframe.csv`
2. **Rebuild** index and text corpus: `index.py` + `build_text.py` (indexes ALL frames).
3. **Test** queries with `search_hybrid.py`.
4. **Collect dev labels** (JSONL) and run `train_reranker.py`.
5. **Use** `search_hybrid_rerank.py` for submissions and `export_csv_hybrid.py` to produce CSV.
6. **Review in FiftyOne**: load `artifacts/mapping.parquet` to browse results visually.

---

## 7) Coding style & tips
- Keep scripts single-purpose and small; add flags rather than branches.
- Avoid heavyweight dependencies; this is a **contest** repo.
- Log every assumption (e.g., truncation when feature count ≠ keyframe count).
- Prefer functional helpers over classes unless you maintain state.

Happy hunting.


---

## 8) Leveling up (GPU-friendly, 4070 Ti)

**Try better encoders** in `config.py`:
- `EVA02-L-14` (fast & strong), or `ViT-L-14-336` for larger receptive field.
- SigLIP is supported if your open_clip install has a SigLIP checkpoint.

**Multi-frame rerank**:
```bash
python multiframe_rerank.py --index_dir ./artifacts   --dataset_root /path/to/dataset_root   --query "mô tả sự kiện"   --cand_csv artifacts/last_candidates.csv   --window 2 --topk 100
```

**PRF (pseudo‑relevance feedback)**:
```bash
python prf_expand.py --corpus ./artifacts/text_corpus.jsonl   --query "mô tả sự kiện" --fb_docs 10 --fb_terms 10
# take the expanded query string and pass it to search_hybrid_rerank.py
```

**Train a stronger re-ranker (GBM)**:
```bash
python train_reranker_gbm.py --index_dir ./artifacts --train_jsonl data/train_dev.jsonl
python search_hybrid_rerank.py --index_dir ./artifacts --query "..." --topk 100 --model_path artifacts/reranker_gbm.joblib
```

**FiftyOne Operator**:
- Install the plugin by pointing FiftyOne to `plugins/aic_tools` and enable it in the App.
- Look for the operator **aic_tools/hybrid_search** in the Operator Browser.


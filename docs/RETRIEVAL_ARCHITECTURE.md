# Retrieval Architecture (Clean V2)

This document describes exactly what the system runs now. It is segment‑first, hybrid (dense + sparse), supports optional cross‑encoder re‑ranking, and exports submission CSVs for KIS, VQA, and TRAKE. The competition host performs all scoring.

## High‑Level Flow

```
Ingest  ──▶  Segmentation  ──▶  Indexing  ──▶  Retrieval  ──▶  (optional) CE Re‑rank  ──▶  CSV Output
videos/keyframes      shots       FAISS+BM25          RRF fusion            top‑K only           KIS/VQA/TRAKE
```

## Components

- Segmentation
  - Script: `scripts/segment_videos.py` (OpenCV HSV‑delta by default; TransNetV2 TorchScript optional via `--model_path`).
  - Output: `artifacts/segments.parquet` with `video_id, seg_id, start_frame, end_frame, start_sec, end_sec, rep_frames` (1–3 reps/segment).

- Dense Index (FAISS)
  - Encoder: SigLIP2/CLIP via `open_clip_torch` (unchanged from baseline; still the core of dense retrieval).
  - Indexed items: representative keyframes only (segment‑first). Mapped back to original frames via `map_keyframes`.
  - Output: `artifacts/index.faiss` + `artifacts/mapping.parquet` (includes `seg_id`, `keyframe_path`).

- Sparse Corpus (BM25)
  - Sources: media_info (title/description/keywords), objects (entities), optional transcripts JSONL (`--transcripts`).
  - Output: `artifacts/text_corpus.jsonl` (one doc per indexed keyframe; includes tokens and optional `seg_id`).

- Retrieval & Fusion
  - Dense: text → SigLIP2/CLIP → FAISS.
  - Sparse: BM25 over `text_corpus.jsonl` tokens.
  - Fusion: Reciprocal Rank Fusion (RRF) combines dense/sparse ranks.

- Re‑ranking (optional)
  - Logistic reranker: uses `artifacts/reranker.joblib` if present.
  - Cross‑encoder (`--rerank ce`): pairwise text↔image scoring on top candidates using `keyframe_path` images (CLIP/SigLIP family). Improves precision@K.

- Tasks & Outputs (CSV only)
  - KIS: `video_id,frame_idx`.
  - VQA: `video_id,frame_idx,answer` (answer provided via CLI; we do not generate it).
  - TRAKE (current): pick top video; for each event from `--events_json`, select best frame in that video; output `video_id,frame1,..,frameN`.
  - Planned: monotonic ε‑alignment (DP/Viterbi) for TRAKE.

## What Did Not Change

- Dense embeddings remain CLIP family (SigLIP2/CLIP via `open_clip_torch`). We changed the sampling strategy (frame sampling → segment representatives), not the dense backbone.
- Intelligent keyframe sampling (keyframe‑only path) is still available; running `index.py` without `--segments` behaves like before.

## Artifacts

```
artifacts/
  segments.parquet   # segments + representative frames
  index.faiss        # dense index of representative keyframes
  mapping.parquet    # metadata: video_id, n, frame_idx, seg_id, keyframe_path, ...
  text_corpus.jsonl  # BM25 corpus (raw + tokens)
  reranker.joblib    # optional logistic reranker bundle

submissions/
  <query_id>.csv     # KIS/VQA/TRAKE CSVs (host scores)
```

## Minimal Commands

```
# Segment
python scripts/segment_videos.py --dataset_root /data/aic2025 --videos L21 L22 --artifact_dir ./artifacts

# Index segment reps
python scripts/index.py --dataset_root /data/aic2025 --videos L21 L22 --segments ./artifacts/segments.parquet

# Build corpus (+ASR optional)
python scripts/build_text.py --dataset_root /data/aic2025 --videos L21 L22 \
  --segments ./artifacts/segments.parquet --artifact_dir ./artifacts \
  --transcripts data/transcripts.jsonl

# Search
python src/retrieval/use.py --query "..." --query_id q1 --rerank ce
python src/retrieval/use.py --task vqa --query "..." --answer "màu xanh" --query_id q2 --rerank ce
python src/retrieval/use.py --task trake --query "high jump" --events_json data/events.json --query_id q3 --rerank ce
```

## Notes

- The competition host computes all scores. This system only exports CSV predictions.
- FAISS auto‑uses GPU when compatible; HNSW stays on CPU. CE uses autocast on CUDA.
- ASR JSONL is optional; pass your precomputed transcripts to `build_text.py --transcripts` to improve recall.


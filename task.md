Retrieval V2 — Tasks, Deliverables, and Plan

This task plan defines the concrete work to upgrade the project to a competition-ready V2: segment-first indexing, ASR-powered hybrid retrieval, cross-encoder re-ranking, and explicit TRAKE alignment. Scoring is performed by the host; this system exports prediction CSVs only.

**Scope & Goals**
- Improve Top-1 precision and recall for Textual KIS and Q&A by adding audio (ASR) and a cross-encoder re-ranker.
- Meet TRAKE’s requirement: pick the correct video first, then output exactly one semantic keyframe per event within ε tolerance.
- Keep the current strengths (SigLIP2, FAISS, BM25, RRF) while adding segment timelines and temporal logic.

**Status Snapshot**
- Done:
  - Segment videos (OpenCV fallback; TransNetV2-ready interface) → `scripts/segment_videos.py`, `src/segmentation/transnetv2.py`, outputs `artifacts/segments.parquet`.
  - Segment-aware indexing → `scripts/index.py --segments` indexes representative frames; `mapping.parquet` includes `seg_id` and `keyframe_path`.
  - Corpus enrichment with transcripts → `scripts/build_text.py --transcripts` merges ASR text (per video) into BM25 docs; supports `--segments`.
  - Retrieval CLI enhancements → `src/retrieval/use.py` supports `--rerank ce` and `--task trake` with `--events_json`.
- Next:
  - TRAKE ε-alignment: add monotonic DP/Viterbi over per‑event scores, `--epsilon` flag (default 10 frames).
  - OCR enrichment: `scripts/ocr_frames.py` + merge into corpus; boosts KIS/VQA on screen text.
  - Cross‑encoder polish: add `--ce-model/--ce-pretrained`, cache preprocessed tensors, limit CE to top‑50.
  - Optional modality boosters: InternVideo2 (segment video features) and CLAP (audio) behind flags.
  - Docs: add minimal `data/events.json` and `data/transcripts.jsonl` stubs and update examples.

**Deliverables**
- New components:
  - Segmentation: `src/segmentation/transnetv2.py` + `scripts/segment_videos.py` → `artifacts/segments.parquet`.
  - ASR: `scripts/asr_whisperx.py` → `artifacts/transcripts.jsonl`.
  - OCR: `scripts/ocr_frames.py` → `artifacts/ocr.jsonl`.
  - Optional captions/tags: `scripts/caption_frames.py`, `scripts/detector_tags.py` → `artifacts/captions.jsonl`, `artifacts/tags.jsonl`.
  - Cross-encoder re-ranker: `src/retrieval/rerank_ce.py`.
  - VQA answerer: `src/retrieval/answer_vqa.py`.
  - TRAKE alignment: `src/retrieval/trake_align.py`.
- Updated build and retrieval:
  - Segment-level corpus build: extend `scripts/build_text.py`.
  - Segment-aware indexing: extend `scripts/index.py` to select 1–3 representative frames per segment.
  - Retrieval CLI: extend `src/retrieval/use.py` to run segment-first hybrid search, optional re-rankers, and task-specific outputs.
- Documentation:
  - Update `docs/RETRIEVAL_ARCHITECTURE.md` with V2 diagram and flow.
  - Keep `docs/COMPETITION_DATA_AND_SUBMISSIONS.md` as the host-only spec.

**Data Schemas**
- `artifacts/segments.parquet`
  - Columns: `video_id` (str), `seg_id` (int), `start_sec` (float), `end_sec` (float), `start_frame` (int), `end_frame` (int), `rep_frames` (list[int])
- `artifacts/transcripts.jsonl`
  - One JSON per segment: `{ "video_id": str, "seg_id": int, "text": str, "words": [{"text": str, "start_sec": float, "end_sec": float}] }`
- `artifacts/ocr.jsonl`
  - Per keyframe or rep-frame: `{ "video_id": str, "seg_id": int, "n": int, "text": str }`
- `artifacts/captions.jsonl` (optional)
  - `{ "video_id": str, "seg_id": int, "n": int, "caption": str }`
- `artifacts/tags.jsonl` (optional)
  - `{ "video_id": str, "seg_id": int, "n": int, "tags": [str] }`
- `artifacts/text_corpus.jsonl` (segment-level)
  - `{ "global_seg_idx": int, "video_id": str, "seg_id": int, "raw": str, "tokens": [str] }`
- `artifacts/index_img.faiss` and `artifacts/seg_mapping.parquet`
  - Mapping columns: `global_seg_idx` (int), `video_id` (str), `seg_id` (int), `rep_frames` (list[int])

**Components & Responsibilities**
- Segmentation (TransNetV2):
  - Detect shot/scene boundaries; write `segments.parquet` and choose `rep_frames` (first/middle/last or energy-based).
- ASR (WhisperX):
  - Transcribe audio; produce segment text with word timestamps; robust to Vietnamese/English.
- OCR:
  - Extract on-screen text for `rep_frames`; multilingual OCR (PP-OCRv3/docTR).
- Corpus build:
  - Merge ASR + OCR + optional captions/tags into a segment-level BM25 corpus.
- Indexing:
  - Compute SigLIP2 embeddings for `rep_frames`; aggregate per segment (e.g., mean/MaxIP across reps) and build `index_img.faiss`.
- Retrieval & Fusion:
  - For a text query: run BM25 over segments and FAISS over segment visuals; fuse via RRF; select top-K segments; map back to frames.
- Cross-encoder re-ranking:
  - Re-score top-K candidates using a pairwise text↔image model (SigLIP pair or LMM for top-20); improve precision@K and correct-video gating.
- Q&A:
  - After retrieval, answer with a VQA model on re-ranked frames; numeric post-processing (digits) with OCR fallback.
- TRAKE alignment:
  - Given an event list, score frames within the selected video; apply temporal smoothness (DP/Viterbi) to pick one semantic keyframe per event within ε.

**CLI & Flags (current + planned)**
- `src/retrieval/use.py`
  - `--mode kis|vqa|trake` (alias of `--task`) [current]
  - `--rerank none|lr|ce` (logistic baseline or cross-encoder) [current]
  - `--events_json <file>` for TRAKE events [current]
  - `--segments segments.parquet` (auto-default `artifacts/segments.parquet`)
  - `--transcripts transcripts.jsonl` (optional)
  - `--ocr ocr.jsonl` (optional)
  - `--topk 100` and `--dedup_radius 1` (existing)
  - `--experimental`, `--default-clip` (existing)
  - `--epsilon <frames>` TRAKE ε tolerance [planned]
  - `--ce-model`, `--ce-pretrained` (override CE backbone) [planned]

**Milestones & Order of Work**
- Sprint 1 (DONE): Baseline upgrades
  - Segment videos → `segments.parquet`; index segment reps; merge transcripts; `use.py` supports CE and TRAKE basic.
- Sprint 2 (NOW): Precision & full coverage
  - TRAKE ε-alignment (DP/Viterbi) with `--epsilon` flag.
  - OCR extraction and corpus merge; expose `--ocr` path.
  - CE polish: `--ce-model/--ce-pretrained`, caching, rerank top‑50.
  - Optional: InternVideo2 pooled features and CLAP audio embeddings.

**Default Models (fast, open)**
- Embeddings: `SigLIP2-L/16-256` (image/text).
- Segmentation: TransNetV2.
- ASR: WhisperX.
- OCR: PP-OCRv3 or docTR.
- Cross-encoder re-rank: SigLIP pairwise as default; optionally Qwen2-VL/LLaVA-OneVision for top-20 heavy re-rank.
- Optional: InternVideo2 small (video), CLAP (audio) when queries mention sounds.

**Acceptance Criteria**
- KIS/VQA: Pipeline exports valid CSVs with correct formatting; improves top-1/top-5 accuracy on an internal dev set versus current baseline.
- TRAKE: For a provided multi-event query spec, outputs one keyframe per event and passes ε-window checks on synthetic tests.
- Performance: End-to-end retrieval latency remains practical (target <1s/query without heavy CE; <3s with CE limited to top-20).
- Robustness: Works with Vietnamese/English queries; degrades gracefully if ASR/OCR not available (corpus still builds).

**Risks & Mitigations**
- Model VRAM limits: provide `--default-clip` and batch sizes; restrict CE to top-20.
- ASR latency: cache transcripts; process once per video; fall back to captions/OCR if ASR missing.
- Schema drift: centralize schemas in this file; validate at load time and fail with clear errors.
- Over-optimization: keep feature flags to disable optional modules.

**Out of Scope**
- Official scoring is performed by the host; we do not implement or claim competition scores here.
- Heavy DB deployment by default; start with FAISS + Parquet/JSONL. Postgres/pgvector, Elasticsearch, or Tantivy are optional later.

**Next Steps**
- Approve this plan.
- I’ll scaffold modules, extend build/retrieval scripts, and add minimal unit tests for schema IO and alignment logic.

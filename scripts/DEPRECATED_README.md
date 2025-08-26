# Deprecated Files

This directory contains deprecated files that should not be used in the modern pipeline.

## Deprecated:
- `smart_pipeline.py.deprecated` - Old monolithic pipeline approach
  - **DO NOT USE**: This has been replaced by the segment-first approach
  - **Use instead**: 
    1. `scripts/segment_videos.py --use_transnetv2`
    2. `scripts/index.py`
    3. `scripts/build_text.py`

## Modern Pipeline:
The current pipeline follows a segment-first architecture:

1. **Video Segmentation**: `scripts/segment_videos.py`
   - Uses TransNetV2 for deep learning shot detection
   - Falls back to OpenCV if TransNetV2 unavailable
   - Outputs: `artifacts/segments.parquet`

2. **Visual Indexing**: `scripts/index.py`
   - Builds FAISS index from segment representative frames
   - GPU-accelerated when available
   - Outputs: `artifacts/index.faiss`, `artifacts/mapping.parquet`

3. **Text Corpus**: `scripts/build_text.py`
   - Creates text search corpus
   - Merges ASR transcripts if available
   - Outputs: `artifacts/text_corpus.jsonl`

4. **Retrieval**: `src/retrieval/use.py`
   - Hybrid visual + text search
   - Cross-encoder reranking
   - Outputs: `submissions/{query_id}.csv`

This replaces the old "smart pipeline" with a more modular, maintainable approach.
# Migration Guide: From Smart Pipeline to Segment-First Architecture

## ⚠️ Important: Pipeline Architecture Changed

The old monolithic `smart_pipeline.py` has been **deprecated** and replaced with a modern segment-first architecture.

## 🚫 Old Approach (Deprecated)
```bash
# DON'T USE THIS ANYMORE:
python smart_pipeline.py /data/aic2024
```

## ✅ New Approach (Current)

The pipeline is now modular with specialized scripts for each step:

### Step 1: Video Segmentation with TransNetV2
```bash
python scripts/segment_videos.py \
  --dataset_root /path/to/dataset \
  --videos L21 L22 L23 \
  --artifact_dir ./artifacts \
  --use_transnetv2  # Enable deep learning shot detection
```
- **Output**: `artifacts/segments.parquet`
- **Features**: TransNetV2 deep learning shot detection with OpenCV fallback

### Step 2: Build Visual Search Index
```bash
python scripts/index.py \
  --dataset_root /path/to/dataset \
  --videos L21 L22 L23 \
  --segments ./artifacts/segments.parquet
```
- **Output**: `artifacts/index.faiss`, `artifacts/mapping.parquet`
- **Features**: GPU-accelerated FAISS indexing

### Step 3: Build Text Corpus
```bash
python scripts/build_text.py \
  --dataset_root /path/to/dataset \
  --videos L21 L22 L23 \
  --artifact_dir ./artifacts \
  --segments ./artifacts/segments.parquet \
  --transcripts data/transcripts.jsonl  # Optional ASR
```
- **Output**: `artifacts/text_corpus.jsonl`
- **Features**: BM25 text search with optional ASR integration

### Step 4: Run Retrieval
```bash
python src/retrieval/use.py \
  --query "your search query" \
  --query_id q1 \
  --rerank ce
```
- **Output**: `submissions/{query_id}.csv`
- **Features**: Hybrid search with cross-encoder reranking

## 🔄 Key Differences

| Aspect | Old (smart_pipeline) | New (segment-first) |
|--------|---------------------|-------------------|
| **Architecture** | Monolithic script | Modular pipeline |
| **Shot Detection** | Basic OpenCV | TransNetV2 deep learning |
| **FAISS** | CPU-only | GPU-accelerated |
| **Maintainability** | Single large file | Specialized modules |
| **Error Recovery** | Restart entire pipeline | Resume from any step |
| **ASR Integration** | Limited | Full corpus merge |
| **Dependencies** | open_clip_torch==2.24.0 | open-clip-torch>=2.24.0 |

## 📦 Updated Dependencies

Install the modern stack:
```bash
pip install -r requirements.txt
pip install transnetv2-pytorch>=1.0.0  # For TransNetV2
pip install faiss-gpu-cu12  # For GPU acceleration (if CUDA available)
```

## 🚀 Benefits of Migration

1. **Better Quality**: TransNetV2 provides superior shot boundary detection
2. **Faster Processing**: GPU acceleration throughout the pipeline
3. **Modular Design**: Can re-run individual steps without redoing everything
4. **Error Recovery**: Resume from failure points
5. **Maintainability**: Easier to debug and extend
6. **ASR Ready**: Proper text corpus integration

## ⚠️ Deprecated Files

These files have been deprecated and should not be used:
- `smart_pipeline.py` → Removed
- `test_smart_pipeline.py` → Removed  
- `SETUP_SMART_PIPELINE.md` → Removed
- `scripts/smart_pipeline.py` → Renamed to `smart_pipeline.py.deprecated`

## 💡 Quick Migration Checklist

- [ ] Update dependencies: `pip install -r requirements.txt`
- [ ] Install TransNetV2: `pip install transnetv2-pytorch`
- [ ] Install GPU FAISS (if GPU available): `pip install faiss-gpu-cu12`
- [ ] Switch to segment-first pipeline scripts
- [ ] Update any automation scripts to use new pipeline
- [ ] Remove references to `smart_pipeline.py`

## 📝 Example: Full Pipeline Run

```bash
# Set up environment
conda activate aic-ftml-gpu

# Run complete pipeline
DATASET=/path/to/aic2025
VIDEOS="L21 L22 L23"

# 1. Segment videos
python scripts/segment_videos.py \
  --dataset_root $DATASET \
  --videos $VIDEOS \
  --artifact_dir ./artifacts \
  --use_transnetv2

# 2. Build index
python scripts/index.py \
  --dataset_root $DATASET \
  --videos $VIDEOS \
  --segments ./artifacts/segments.parquet

# 3. Build text corpus  
python scripts/build_text.py \
  --dataset_root $DATASET \
  --videos $VIDEOS \
  --artifact_dir ./artifacts \
  --segments ./artifacts/segments.parquet

# 4. Run search
python src/retrieval/use.py \
  --query "person opening laptop" \
  --query_id q1 \
  --rerank ce
```

## 🆘 Getting Help

If you encounter issues during migration:
1. Check that all dependencies are updated
2. Ensure TransNetV2 is installed: `pip install transnetv2-pytorch`
3. Verify FAISS GPU if using GPU: `python -c "import faiss; print(hasattr(faiss, 'StandardGpuResources'))"`
4. Review the error messages - they now provide better guidance

The new pipeline is more robust, faster, and produces better results!
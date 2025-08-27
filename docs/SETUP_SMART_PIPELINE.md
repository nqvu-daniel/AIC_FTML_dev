# Academic-Grade Pipeline Setup Guide 🏆

The pipeline has been upgraded with **academic-grade TransNet-V2 shot boundary detection** + **intelligent sampling enhancements** + **SigLIP2** support for superior retrieval performance.

## 🔧 Requirements

Install dependencies first:

```bash
# Install academic-grade packages
pip install -r requirements.txt

# Key academic dependencies:
pip install transnetv2-pytorch>=1.0.5 ffmpeg-python ultralytics transformers easyocr

# Or for Colab:
!pip install transnetv2-pytorch open_clip_torch faiss-cpu pandas numpy Pillow tqdm scikit-learn rank_bm25 scipy pyarrow ultralytics transformers easyocr ffmpeg-python
```

## 🎯 Quick Start

### 1. Full Academic-Grade Pipeline 
```bash
# Academic-grade TransNet-V2 + SigLIP2 (recommended)
python pipeline.py build \
    --video_dir /path/to/videos \
    --target_frames 50 \
    --use_transnet \
    --enable_ocr --enable_captions

# Pure intelligent sampling (disable TransNet-V2)  
python pipeline.py build \
    --video_dir /path/to/videos \
    --target_frames 50 \
    --disable_transnet

# Full academic stack with all features
python pipeline.py build \
    --video_dir /path/to/videos \
    --target_frames 50 \
    --use_transnet \
    --enable_ocr --enable_captions --enable_segmentation \
    --use_flat
```

### 2. Academic Search & Evaluation

**Search with Academic-Grade System**
```bash
# Hybrid search with TransNet-V2 processed keyframes
python search.py --query "news anchor speaking" --search_mode hybrid --k 100

# Vector-only search
python search.py --query "outdoor scene" --search_mode vector --k 100 --output results.csv

# Text search with BM25
python search.py --query "weather forecast" --search_mode text --k 100
```

**Contest Submission Generation**
```bash
# Generate CSV submissions for AIC 2025
python utils/make_submission.py --spec queries.json --index_dir ./artifacts
```

### 3. Traditional Pipeline (now with CLIP sampling)
```bash
python smart_pipeline.py /path/to/dataset
```

## 🧪 Testing

Test if everything works:
```bash
python test_smart_pipeline.py
```

## 📊 What's New vs Old Pipeline

| Feature | Old Pipeline | New Pipeline |
|---------|-------------|--------------|
| **Frame Selection** | Visual complexity only | CLIP semantic understanding |
| **Model** | ViT-B-32 | SigLIP2-L/16 (82.5% ImageNet) |
| **Scene Detection** | Basic histogram | CLIP embedding similarity |
| **Query Relevance** | None | 13 common query types |
| **Diversity** | Temporal gaps only | Semantic diversity + temporal |
| **Multilingual** | No | Yes (SigLIP2) |

## 🔍 Available Models

Configure in `config.py` or use experimental flags:

- `siglip2-l16-256` - SigLIP2 L/16 (82.5% ImageNet, multilingual)
- `siglip2-so400m` - SigLIP2 So400m (83.1% ImageNet) 
- `bigg` - OpenCLIP ViT-bigG/14 (80.1% ImageNet)
- `h14` - ViT-H-14 (78.0% ImageNet)

## 🎛️ Key Parameters

- `--target_frames N` - Frames per video (default: 30-50)
- `--diversity_weight 0.4` - Semantic diversity importance  
- `--relevance_weight 0.4` - Query relevance importance
- `--temporal_weight 0.2` - Temporal spread importance
- `--min_gap_seconds 1.0` - Minimum time between frames

## 📈 Expected Improvements

- **Better retrieval quality**: Semantic frame selection
- **Multilingual queries**: SigLIP2 multilingual tokenizer  
- **Scene awareness**: Better coverage of video content
- **Query-aware**: Frames selected for common search patterns

## 🐛 Troubleshooting

**Import errors**: Install requirements: `pip install -r requirements.txt`

**CUDA memory**: Use smaller `--batch_size` or `--target_frames`

**No videos found**: Check `--video_pattern` (default: `*.mp4`)

**Old artifacts**: Delete `artifacts/` folder to rebuild with new model

## 💡 Colab Usage

```python
# Install
!pip install open_clip_torch faiss-cpu scikit-learn rank_bm25

# Full pipeline  
!python scripts/smart_pipeline.py --video_dir /content/videos --experimental --exp_model siglip2-l16-256

# Test query
!python src/retrieval/use.py --query "a person opening a laptop"
```
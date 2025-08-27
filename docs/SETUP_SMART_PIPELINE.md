# Smart Pipeline Setup Guide 🚀

The pipeline has been upgraded with **CLIP-guided frame sampling** and **SigLIP2** support for better retrieval performance.

## 🔧 Requirements

Install dependencies first:

```bash
# Install Python packages
pip install -r requirements.txt

# Or for Colab:
!pip install open_clip_torch faiss-cpu pandas numpy Pillow tqdm scikit-learn rank_bm25 scipy pyarrow
```

## 🎯 Quick Start

### 1. Full Pipeline (Sampling + Indexing)
```bash
# With SigLIP2 (recommended for T4)
python scripts/smart_pipeline.py \
    --video_dir /path/to/videos \
    --target_frames 50 \
    --experimental --exp_model siglip2-l16-256

# Default model
python scripts/smart_pipeline.py \
    --video_dir /path/to/videos \
    --target_frames 30
```

### 2. Step-by-Step Pipeline

**Step 1: Frame Sampling Only**
```bash
python scripts/smart_pipeline.py \
    --video_dir /path/to/videos \
    --target_frames 50 \
    --sampling_only
```

**Step 2: Indexing Only** (after sampling)
```bash
python scripts/smart_pipeline.py \
    --video_dir /path/to/videos \
    --artifact_dir ./artifacts \
    --indexing_only
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
# AI Challenge 2024 — Intelligent Video Retrieval System

An automated, intelligent video processing system for the AIC 2024 Event Retrieval challenge. Features one-command automation, advanced frame sampling, and hybrid search capabilities.

## Key Features

### Smart Automation
- **One-command processing**: Automatically discovers, validates, and processes your entire dataset
- **Intelligent video discovery**: Finds all videos regardless of naming convention or archive format
- **Dataset validation**: Automatically extracts archives, fixes structure issues, validates files
- **Progress tracking**: Real-time logging with timing and error handling
- **Resume capability**: Skips already processed components to save time

### Advanced Intelligent Sampling
Multiple sophisticated algorithms replace basic uniform sampling:
- **Visual complexity scoring**: Edge density, color diversity, texture analysis
- **Scene change detection**: Multi-method scene boundary identification
- **Motion analysis**: Optical flow-based movement tracking
- **Semantic importance**: Context-aware frame selection with temporal weighting
- **Smart deduplication**: Ensures minimum gaps while maximizing coverage

### Hybrid Search Architecture
- **Dense retrieval**: CLIP-based visual similarity matching
- **Lexical search**: BM25-based text matching with metadata
- **Fusion**: RRF (Reciprocal Rank Fusion) combining both approaches
- **Re-ranking**: Learned models for improved precision
- **Multi-frame context**: Temporal re-scoring for better accuracy

## Quick Start

### Competition Host Quickstart (1–2 commands)
Evaluating on the fixed competition dataset. No frame extraction or indexing required for hosts.

```bash
# 1) Create and use the environment (GPU or CPU)
./setup_env.sh --gpu   # or: ./setup_env.sh --cpu
conda activate aic-ftml-gpu  # or: aic-ftml

# 2) Run a query (writes Top-100 CSV → submissions/)
python src/retrieval/use.py --query "your search"

# Optional for first run: auto-fetch a prebuilt artifacts bundle (index + mapping + corpus + model)
python src/retrieval/use.py --query "your search" \
  --bundle_url <ARTIFACTS_BUNDLE_URL>
```

### Setup Environment
```bash
# Default: create GPU environment (use --cpu for CPU-only)
./setup_env.sh --gpu   # or: ./setup_env.sh --cpu
conda activate aic-ftml-gpu  # or: conda activate aic-ftml
```

### Process Your Dataset
```bash
# One command processes everything automatically
python smart_pipeline.py /path/to/your/dataset

# With options
python smart_pipeline.py /path/to/your/dataset --workers 8 --no-gpu
```

### Linux CLI Deployment
- Prereqs: Linux with conda (Miniconda/Anaconda). GPU optional (CUDA 11.8+/driver if using GPU env).
- Install env: `./setup_env.sh --gpu` (or `--cpu`), then `conda activate aic-ftml-gpu` (or `aic-ftml`).
- Run end‑to‑end via CLI: `python smart_pipeline.py /data/aic2024`.
- No web service is included by default; this project is CLI‑first. A thin API can be added if needed.

### GPU Support & Requirements
- **Recommended**: NVIDIA GPU with 8GB+ VRAM for faster processing
- **CPU fallback**: Works on CPU-only systems (slower embedding/search)
- **FAISS**: Automatically detects GPU and uses it when available
- **Index types**: Flat index for GPU acceleration, HNSW for CPU-only
- **Colab**: Both notebooks auto-detect GPU and install appropriate FAISS version

### Search and Export
```bash
# One-shot search (recommended for users). Writes Top-100 CSV to submissions/
python src/retrieval/use.py --query "your search description"

# Developer alternatives:
# Inspect reranked results (prints table)
python src/retrieval/search_hybrid_rerank.py --index_dir ./artifacts \
  --query "your search description" --topk 100

# Official submission naming (per-query files): write directly to submissions/{query_id}.csv
python src/retrieval/use.py --query "your search" --query_id q123

# VQA format (3 columns): video_id,frame_idx,answer
python src/retrieval/use.py --query "question text" --task vqa --answer "màu xanh" --query_id q_vqa_01


# Export baseline fusion (RRF) CSV explicitly (also supports --answer)
python src/retrieval/export_csv.py --index_dir ./artifacts \
  --query "search query" --outfile submissions/q123.csv
```

## Official Submission & Evaluation

**📋 See [`EVALUATION.md`](EVALUATION.md) for complete evaluation instructions, CSV formats, and official scoring.**

Quick reference:
- Single query: `python src/retrieval/use.py --query "search" --query_id q123 --task kis`
- Batch creation: `python scripts/make_submission.py --spec queries.json --index_dir ./artifacts`
- Ground truth format: KIS/VQA use `span: [start,end]`, TRAKE uses `spans: [[s1,e1],...]`

### CLI Cheatsheet (Developers)
# The steps below (sampling, indexing, corpus building, training) are for development.
# Competition hosts do NOT need these; use the one-shot command above.
- Validate + preprocess: `python scripts/dataset_validator.py /path/to/dataset`
- Intelligent sampling: `python src/sampling/frames_intelligent.py --dataset_root /path/to/dataset --videos L21 L22 --target_fps 0.5 --use_gpu`
- Build index: `python scripts/index.py --dataset_root /path/to/dataset --videos L21 L22`
  - Add `--flat` for GPU-compatible flat index (faster search with faiss-gpu)
  - Default HNSW index works on CPU but not GPU-accelerated
- Build text corpus: `python scripts/build_text.py --dataset_root /path/to/dataset --videos L21 L22`
- Train re‑ranker: `python src/training/train_reranker.py --index_dir ./artifacts --train_jsonl data/train.jsonl`
- Search (auto‑uses model if present): `python src/retrieval/search_hybrid_rerank.py --index_dir ./artifacts --query "query" --topk 100`
- Export CSV: `python src/retrieval/export_csv.py --index_dir ./artifacts --query "query" --outfile results.csv`

## Dependencies

Core runtime
- Python: 3.9
- PyTorch: >=2.1 (CUDA optional via `environment-gpu.yml`)
- torchvision, torchaudio (paired with PyTorch)

Retrieval and vision
- open_clip_torch: >=2.24.0 (CLIP models + preprocess)
- FAISS: `faiss-cpu` (or `faiss-gpu` in GPU env)
- OpenCV: conda `opencv`
- Pillow: >=10.0
- decord: >=0.6.0 (video IO; optional but recommended)

Data & utils
- numpy: >=1.24
- scipy: >=1.11.0
- pandas: >=2.0
- pyarrow: >=14.0.0 (Parquet engine)
- tqdm, pyyaml, joblib

Search / ranking
- rank_bm25: >=0.2.2
- scikit-learn: >=1.4 (Logistic Regression reranker)
- lightgbm: 4.5.0 (optional; default path uses sklearn HGBT)

Visualization (optional)
- fiftyone: for dataset inspection and plugin integration

Notes
- Use `setup_env.sh` to create a conda environment: CPU (`environment.yml`) or GPU (`environment-gpu.yml`).
- `requirements.txt` mirrors these dependencies for pip-only setups; prefer conda envs for GPU.

## Competition Understanding

### AIC 2024 Event Retrieval Challenge Tasks

#### Textual KIS (Known-Item Search)
- Input: Natural language description of a video moment
- Output: `video_id,frame_idx` pairs in CSV format
- Scoring: 1.0 if both video_id matches AND frame_idx is within acceptable range, else 0.0

#### Q&A Mode  
- Input: Question about a specific moment in video
- Output: `video_id,frame_idx,answer` in CSV format
- Scoring: 1.0 if video_id, frame_idx, AND answer are all correct, else 0.0

#### TRAKE (Temporal Alignment)
- Input: Description of multi-step action requiring multiple moments
- Output: `video_id,frame1,frame2,...,frameN` in CSV format  
- Scoring: Percentage of moments correctly identified within acceptable ranges

### Evaluation System
- Final Score: Average of best scores at Top-1, Top-5, Top-20, Top-50, and Top-100
- Submission: Up to 100 candidates per query
- Format: CSV files without headers
 - KIS: `video_id,frame_idx`; VQA: `video_id,frame_idx,answer`; TRAKE: `video_id,frame1,frame2,...,frameN`

## Dataset Structure

```
dataset_root/
  videos/
    L21_V001.mp4                  # Video files (auto-discovered)
  keyframes/                      # Competition-provided keyframes
    L21_V001/
      001.png 002.png ...
  keyframes_intelligent/          # Generated intelligent samples
    L21_V001/
      001.png 002.png ...
  meta/
    L21_V001.map_keyframe.csv     # Frame mapping with importance scores
    L21_V001.media_info.json      # Video metadata (optional)
    objects/
      L21_V001/
        001.json 002.json ...     # Object detection (optional)
  features/
    L21_V001.npy                  # Precomputed features (optional)
  artifacts/                      # Generated by system
    *.index                       # Search indices
    *.jsonl                       # Text corpus
    *.joblib                      # Trained models
```

## Repository Structure

```
AIC_FTML_dev/
├── smart_pipeline.py             # Automated processing pipeline
├── setup_env.sh                  # Conda environment setup
├── config.py                     # Configuration settings
├── utils.py                      # Shared utilities
├── src/
│   ├── sampling/
│   │   └── frames_intelligent.py # Advanced frame sampling
│   ├── retrieval/
│   │   ├── search.py             # Dense search (baseline)
│   │   ├── search_hybrid.py      # Hybrid search
│   │   ├── search_hybrid_rerank.py # Best performance method
│   │   ├── use.py                # One-shot user CLI (Top-100 CSV)
│   │   └── export_csv.py         # Competition export
│   └── training/
│       ├── train_reranker.py     # Logistic regression re-ranker
│       └── train_reranker_gbm.py # Gradient boosting re-ranker
├── scripts/
│   ├── dataset_validator.py      # Dataset preprocessing
│   ├── index.py                  # Search index builder
│   ├── build_text.py             # Text corpus builder
│   ├── multiframe_rerank.py      # Temporal context scoring
│   ├── prf_expand.py             # Query expansion
│   └── download_models.py        # Helper to fetch trained reranker
│   └── dataset_downloader.py     # Download and arrange competition dataset
├── eval/
│   └── evaluate.py               # Competition evaluation
└── plugins/
    └── aic_tools/                # FiftyOne integration
```

## Advanced Usage

### Manual Step-by-Step Processing
For advanced users who want full control:

```bash
# 1) Validate and preprocess dataset
python scripts/dataset_validator.py /path/to/dataset

# 2) Extract intelligent keyframes
python src/sampling/frames_intelligent.py --dataset_root /path/to/dataset \
  --videos L21 L22 L23 --target_fps 0.5 --use_gpu

# 3) Build search infrastructure
python scripts/index.py --dataset_root /path/to/dataset --videos L21 L22 L23
python scripts/build_text.py --dataset_root /path/to/dataset --videos L21 L22 L23

# 4) Train re-ranking model (optional, if training data available)
python src/training/train_reranker.py --index_dir ./artifacts \
  --train_jsonl data/train_dev.jsonl

# 5) Search and export
python src/retrieval/search_hybrid_rerank.py --index_dir ./artifacts \
  --query "search description" --topk 100
python src/retrieval/export_csv.py --index_dir ./artifacts \
  --query "search query" --outfile results.csv
```

### Training Re-ranking Models

Training data format (JSONL, one object per line):
```json
{"query": "event description", "positives": [{"video_id":"L21_V001","frame_idx":2412}]}
{"query": "another query", "positives": [{"video_id":"L22_V003","frame_idx":1500}]}
```

Train models:
```bash
# Logistic regression re-ranker
python src/training/train_reranker.py --index_dir ./artifacts --train_jsonl data/train.jsonl

# Gradient boosting re-ranker (more powerful)
python src/training/train_reranker_gbm.py --index_dir ./artifacts --train_jsonl data/train.jsonl
```

### Model Loading in Inference
- The search CLI automatically loads a trained re‑ranker from `./artifacts/reranker.joblib` if present and uses it to score candidates.
- If no trained model is found, it falls back to a robust fusion baseline (RRF over dense and BM25 ranks).
- The end‑to‑end `smart_pipeline.py` will attempt training automatically if it finds `train*.jsonl` or `dev*.jsonl` under the dataset root.

To use a hosted trained model (recommended for users), download it to `./artifacts/reranker.joblib` via:

```bash
conda run -n aic-ftml-gpu python scripts/download_models.py \
  --model-url <MODEL_URL> --outfile ./artifacts/reranker.joblib
```

### Advanced Features

#### Multi-frame Temporal Re-ranking
```bash
python scripts/multiframe_rerank.py --index_dir ./artifacts \
  --dataset_root /path/to/dataset \
  --query "event description" \
  --cand_csv artifacts/candidates.csv \
  --window 3 --topk 100
```

#### Query Expansion with PRF
```bash
python scripts/prf_expand.py --corpus ./artifacts/text_corpus.jsonl \
  --query "original query" --fb_docs 10 --fb_terms 10
```

## Evaluation

### Running Competition Evaluation

Ground truth format (`gt.json`):
```json
[
  {"query_id":"q1","task":"kis","video_id":"L21_V001","span":[500,510]},
  {"query_id":"q2","task":"vqa","video_id":"L22_V002","span":[800,900],"answer":"red"},
  {"query_id":"q3","task":"trake","video_id":"L23_V001","spans":[[100,110],[200,210],[300,310]]}
]
```

Evaluation commands:
```bash
python eval/evaluate.py --gt ground_truth.json --pred_dir submissions/ --task kis
python eval/evaluate.py --gt ground_truth.json --pred_dir submissions/ --task vqa --normalize_answer
python eval/evaluate.py --gt ground_truth.json --pred_dir submissions/ --task trake
```

## Configuration

### Model Settings
Edit `config.py` to adjust:
- **Model**: Default `ViT-L-14`. Use `ViT-B-32` for limited VRAM, `EVA02-L-14` for best performance
- **Index**: HNSW for speed, FlatIP for accuracy
- **Fusion**: RRF parameter `k=60` (good default)

### Sampling Parameters
- `target_fps`: Frames per second to extract (default: 0.5)
- `window_size`: Temporal context window (default: 16)
- `min_gap`: Minimum frames between selections (default: 8)

## Environment Management

### Conda Environments
- `environment.yml`: CPU-only environment
- `environment-gpu.yml`: GPU-enabled environment with CUDA support
- `setup_env.sh`: Automatic setup with GPU detection

### Environment Commands
```bash
# List environments
conda env list

# Update environment
conda env update -f environment.yml

# Remove environment
conda env remove -n aic-ftml

# Export current environment
conda env export > my-environment.yml
```

## Troubleshooting

### Common Issues
- **Out of Memory**: Reduce `target_fps` or disable GPU with `--no-gpu`
- **No videos found**: Check dataset path and archive extraction
- **Import errors**: Ensure correct conda environment is activated
- **Slow processing**: Enable GPU acceleration or reduce dataset size for testing

### Performance Tips
- Use GPU acceleration when available (`--use_gpu`)
- Process subsets first to test configuration
- Check logs in `dataset_root/logs/` for detailed progress
- Use precomputed features if available to skip embedding computation

## Cloud Deployment

This system is deployment-ready for cloud environments:
- **AWS**: `p3.2xlarge` or `g4dn.xlarge` instances
- **GCP**: `n1-standard-4` with GPU acceleration
- **Minimum**: 16GB RAM + GPU (RTX 4070 or equivalent)

```bash
# On cloud instance (developer training/indexing workflow)
git clone <your-repo>
cd AIC_FTML_dev

# 1) Environment
./setup_env.sh --gpu   # or --cpu
conda activate aic-ftml-gpu

# 2) Download and arrange the competition dataset (optional for devs)
python scripts/dataset_downloader.py --dataset_root /data/aic2025 --csv AIC_2025_dataset_download_link.csv --skip-existing

# 3) Build artifacts (index + corpus)
python scripts/index.py --dataset_root /data/aic2025 --videos L21 L22 L23 L24 L25 L26 L27 L28 L29 L30
python scripts/build_text.py --dataset_root /data/aic2025 --videos L21 L22 L23 L24 L25 L26 L27 L28 L29 L30

# 4) Train reranker (optional if you already have a trained model)
python src/training/train_reranker.py --index_dir ./artifacts --train_jsonl data/train.jsonl

# 5) Package artifacts for host inference
python scripts/package_artifacts.py --artifact_dir ./artifacts --output ./artifacts_bundle.tar.gz --include_model

# Upload ./artifacts_bundle.tar.gz to your hosting and provide its URL to the host

# Optional: assemble a minimal runnable folder for hosts
python scripts/prepare_pipeline_dir.py --outdir my_pipeline --artifact_dir ./artifacts --include_model --force
# my_pipeline/ contains: src/retrieval/use.py, config.py, utils.py, artifacts/*, submissions/
```

### Colab Notebooks
**`notebooks/colab_pipeline.ipynb`** - Complete development pipeline:
- Host inference: Uses pre-built artifacts for instant queries
- Dev pipeline: Downloads dataset, builds artifacts, trains reranker (optional)
- Auto-detects GPU and builds flat index for GPU acceleration
- Outputs ready-to-deploy `my_pipeline/` directory

**`notebooks/colab_official_eval.ipynb`** - Official evaluation & submission:
- Per-query CSV generation with proper naming (`{query_id}.csv`)
- Supports KIS/VQA tasks with ground truth evaluation
- Auto-detects GPU and uses it for faster search
- Lighter notebook focused on inference and scoring

## Performance Expectations

- **Processing**: 10-30 video collections per hour (depending on size)
- **Storage**: 70-90% reduction vs uniform sampling
- **Quality**: 15-25% improvement in recall@20 vs baseline methods
- **Search**: Sub-second response for queries over processed dataset

## Citation

If you use this system in research, please cite:
```
AI Challenge 2024 Intelligent Video Retrieval System
https://github.com/your-repo/AIC_FTML_dev
```

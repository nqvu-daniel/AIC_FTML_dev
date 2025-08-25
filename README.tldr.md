AI Challenge 2024 — Intelligent Video Retrieval System — TL;DR

What this repo is
- CLI-first system for automated video processing, intelligent keyframe sampling, and hybrid visual+text retrieval tuned for the AIC 2024 Event Retrieval challenge.

When to use it
- Competition host (fixed dataset): run one-shot search using prebuilt artifacts and a trained reranker.
- Developers: end-to-end processing of a local dataset (validation, sampling, indexing, optional training, and search).

Quick commands
- Create env: `./setup_env.sh --gpu` (or `--cpu`) then `conda activate aic-ftml-gpu`.
- One-shot search (writes Top-100 CSV to `submissions/`):
  `python src/retrieval/use.py --query "your search"`
  - If artifacts are not present locally: add `--bundle_url <ARTIFACTS_BUNDLE_URL>`
  - If the reranker isn’t present: add `--model_url <MODEL_URL>`
- Developer: process dataset end-to-end: `python smart_pipeline.py /path/to/dataset`.
- Developer: export CSV from an index: `python src/retrieval/export_csv.py --index_dir ./artifacts --query "q" --outfile results.csv`

Where to look
- Core pipeline: `smart_pipeline.py` and `scripts/*` for preprocessing and indexing.
- Sampling: `src/sampling/frames_intelligent.py`.
- Retrieval: `src/retrieval/*.py` (dense, hybrid, rerank, export).
- Training rerankers: `src/training/`.
- Dataset downloader: `scripts/dataset_downloader.py` (pulls & sorts competition data)

Notes & defaults
- Default CLIP: ViT-L-14, pretrained "openai" (configurable in `config.py`).
- Artifacts bundle is for the fixed competition dataset; hosts do not need to rebuild indices.
- Project is optimized for conda; prefer the provided `environment.yml` / `environment-gpu.yml`.
- CLI-first (no web server by default).

If you want more
- I can create shorter per-directory TL;DRs (e.g., `src/retrieval/README.tldr.md`) or add one-liners to individual scripts.

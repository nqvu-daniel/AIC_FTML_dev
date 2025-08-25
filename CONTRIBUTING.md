# Contributing

Thanks for contributing! This guide summarizes environment setup, code style, and workflows used in this repo.

## Environment
- Create the conda env (auto-detects GPU):
  - `./setup_env.sh` (add `--force` to recreate)
- Activate:
  - CPU: `conda activate aic-ftml`
  - GPU: `conda activate aic-ftml-gpu`

## Tooling
- Lint/format: Ruff
  - Install: `pip install ruff pre-commit` (or use conda-forge)
  - Run lint: `make lint`
  - Run format: `make format`
- Pre-commit hooks
  - `make precommit` to install hooks (runs Ruff on commit)

## Code Style
- Python 3.9+; line length 120.
- Prefer explicit exceptions over bare `except:`.
- Avoid unnecessary global state; pass paths/configs via params where possible.
- Keep changes focused; do not refactor unrelated code in the same PR.

## Data/Artifacts
- Do not commit generated artifacts:
  - `artifacts/`, `*.parquet`, `*.faiss`, `*.joblib`, logs, jsonl corpora
- Use `make clean` to remove caches (`__pycache__`, `*.pyc`).

## Testing
- For retrieval/training code, validate on a small sample dataset before large runs.
- Prefer fast, deterministic checks where possible.

## Commits/PRs
- Use clear titles and short body explaining the Why/What.
- Keep PRs small and reviewable.
- Link to issues or add context for changes.

## Notes
- Parquet IO requires `pyarrow` (added in envs/requirements).
- Autocast is CUDA-only; CPU runs use float32 for stability.


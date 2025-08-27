#!/bin/bash
# Setup script for AIC-FTML conda environment

set -e

echo "Setting up AIC-FTML conda environment..."

# Parse flags: default to GPU; allow --cpu / --gpu and --force
FORCE_REMOVE=0
TARGET="gpu"
for arg in "$@"; do
  case "$arg" in
    --force) FORCE_REMOVE=1 ;;
    --cpu) TARGET="cpu" ;;
    --gpu) TARGET="gpu" ;;
  esac
done

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Choose target (default GPU). Warn if GPU requested but not detected.
if [ "$TARGET" = "gpu" ]; then
    ENV_FILE="environment-gpu.yml"; ENV_NAME="aic-ftml-gpu"
    if ! command -v nvidia-smi &> /dev/null; then
        echo "[WARN] nvidia-smi not found. Proceeding to create GPU env anyway."
        echo "       If you don't have a CUDA-capable GPU, consider running with --cpu."
    else
        echo "NVIDIA GPU detected; creating GPU environment..."
    fi
else
    ENV_FILE="environment.yml"; ENV_NAME="aic-ftml"
    echo "Creating CPU-only environment..."
fi

# Remove existing environment if it exists (prompt unless --force)
if conda env list | grep -q "^${ENV_NAME} "; then
    if [ $FORCE_REMOVE -eq 1 ]; then
        echo "Removing existing environment: ${ENV_NAME}"
        conda env remove -n ${ENV_NAME} -y
    else
        read -p "Environment '${ENV_NAME}' exists. Remove and recreate? [y/N] " ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            conda env remove -n ${ENV_NAME} -y
        else
            echo "Keeping existing environment. Exiting."
            exit 0
        fi
    fi
fi

# Create new environment
echo "Creating conda environment from ${ENV_FILE}..."
conda env create -f ${ENV_FILE}

# Activate and verify installation
echo "Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo ""
echo "Setup complete!"

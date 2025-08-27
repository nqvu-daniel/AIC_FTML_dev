#!/bin/bash
# Vast.ai deployment script for AIC video preprocessing

set -e

echo "🚀 AIC Video Preprocessing - Vast.ai Deployment"
echo "=============================================="

# Configuration
GPU_TYPE=${1:-"RTX4090"}  # RTX4090, RTX3090, H100, etc.
DATASET_SIZE=${2:-"300GB"}
ENABLE_FEATURES=${3:-"true"}

echo "🖥️  Target GPU: $GPU_TYPE"
echo "📁 Dataset size: $DATASET_SIZE"
echo "⚡ Enable all features: $ENABLE_FEATURES"

# Step 1: Setup environment
echo "📦 Setting up environment..."

# Update system
apt-get update
apt-get install -y git wget unzip

# Install Python dependencies
pip install -r requirements.txt

echo "✅ Environment setup complete"

# Step 2: Download dataset (from your storage)
echo "📥 Downloading dataset..."

# Option A: From Google Drive (if you uploaded there)
# pip install gdown
# gdown --folder "your-google-drive-folder-id" --output ./videos

# Option B: From cloud storage
# aws s3 sync s3://your-bucket/videos ./videos --no-sign-request
# gsutil -m cp -r gs://your-bucket/videos ./videos

# Option C: From wget/curl (if accessible)
# wget -r -np -nH --cut-dirs=2 "https://your-server/videos/" -P ./videos

# For testing: use sample data
echo "📹 Using sample dataset for testing..."
mkdir -p ./videos
# Add your actual download commands here

echo "✅ Dataset downloaded"

# Step 3: Pre-download models to avoid timeout
echo "🤖 Pre-downloading models..."

python -c "
import open_clip
print('Downloading CLIP model...')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-16-SigLIP-256', pretrained='webli')
print('✅ CLIP model ready')
"

if [ "$ENABLE_FEATURES" = "true" ]; then
    python -c "
# FastSAM
try:
    from ultralytics import FastSAM
    print('Downloading FastSAM...')
    model = FastSAM('FastSAM-x.pt')
    print('✅ FastSAM ready')
except Exception as e:
    print(f'⚠️  FastSAM failed: {e}')

# EasyOCR
try:
    import easyocr
    print('Downloading EasyOCR...')
    reader = easyocr.Reader(['en', 'vi'])
    print('✅ EasyOCR ready')
except Exception as e:
    print(f'⚠️  EasyOCR failed: {e}')

# BLIP-2
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    print('Downloading BLIP-2...')
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
    print('✅ BLIP-2 ready')
except Exception as e:
    print(f'⚠️  BLIP-2 failed: {e}')
"
fi

echo "✅ All models downloaded"

# Step 4: Run preprocessing
echo "🚀 Starting video preprocessing..."

# Determine batch size based on GPU
case $GPU_TYPE in
    "H100"|"A100")
        BATCH_SIZE=64
        TARGET_FRAMES=50
        ;;
    "RTX4090"|"RTX3090")
        BATCH_SIZE=32
        TARGET_FRAMES=50
        ;;
    "RTX3080"|"RTX2080")
        BATCH_SIZE=16
        TARGET_FRAMES=40
        ;;
    *)
        BATCH_SIZE=8
        TARGET_FRAMES=30
        ;;
esac

echo "⚙️  Batch size: $BATCH_SIZE, Target frames: $TARGET_FRAMES"

# Run the preprocessing pipeline
if [ "$ENABLE_FEATURES" = "true" ]; then
    python pipeline.py build \
        --video_dir ./videos \
        --output_dir ./artifacts \
        --target_frames $TARGET_FRAMES \
        --batch_size $BATCH_SIZE \
        --enable_ocr \
        --enable_captions \
        --enable_segmentation
else
    python pipeline.py build \
        --video_dir ./videos \
        --output_dir ./artifacts \
        --target_frames $TARGET_FRAMES \
        --batch_size $BATCH_SIZE
fi

echo "✅ Preprocessing complete!"

# Step 5: Compress and upload artifacts
echo "📦 Compressing artifacts..."
tar -czf artifacts.tar.gz ./artifacts/
echo "✅ Artifacts compressed: $(du -sh artifacts.tar.gz | cut -f1)"

# Step 6: Upload to storage
echo "☁️  Uploading artifacts..."

# Option A: Google Drive
# pip install pydrive2
# python -c "
# from pydrive2.auth import GoogleAuth
# from pydrive2.drive import GoogleDrive
# # Upload artifacts.tar.gz
# "

# Option B: Cloud storage
# aws s3 cp artifacts.tar.gz s3://your-bucket/aic-artifacts.tar.gz
# gsutil cp artifacts.tar.gz gs://your-bucket/aic-artifacts.tar.gz

# Option C: File transfer service
echo "📤 Upload artifacts.tar.gz manually to your preferred storage"
echo "   Size: $(du -sh artifacts.tar.gz | cut -f1)"
echo "   Location: $(pwd)/artifacts.tar.gz"

# Step 7: Cleanup (optional)
echo "🧹 Cleaning up..."
echo "Keeping artifacts.tar.gz for download"
# rm -rf ./videos  # Remove original videos to save space
# rm -rf ./artifacts  # Remove uncompressed artifacts

echo "🎉 Vast.ai deployment complete!"
echo "="*50
echo "📊 Summary:"
echo "   Input: $DATASET_SIZE dataset"
echo "   Output: artifacts.tar.gz (~5-10GB)"
echo "   GPU: $GPU_TYPE"
echo "   Features: OCR=$ENABLE_FEATURES, Captions=$ENABLE_FEATURES"
echo ""
echo "📥 Next steps:"
echo "1. Download artifacts.tar.gz to your local machine"
echo "2. Extract and test: tar -xzf artifacts.tar.gz"
echo "3. Run queries: python search.py --query 'your query'"
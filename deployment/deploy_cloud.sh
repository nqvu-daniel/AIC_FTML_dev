#!/bin/bash
# Cloud deployment script for AIC video preprocessing

set -e

echo "🚀 AIC Video Preprocessing Cloud Deployment"
echo "==========================================="

# Configuration
DATASET_SIZE=${1:-"300GB"}
CLOUD_PROVIDER=${2:-"colab"}
OUTPUT_BUCKET=${3:-"your-drive-folder"}

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Download models if needed
echo "🤖 Downloading models..."
python -c "
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-16-SigLIP-256', pretrained='webli')
print('✅ CLIP model cached')
"

# Optional: Download FastSAM, EasyOCR, BLIP-2 models
if [ "$ENABLE_ALL_FEATURES" = "true" ]; then
    echo "🎯 Downloading additional models..."
    python -c "
import ultralytics
from ultralytics import FastSAM
model = FastSAM('FastSAM-x.pt')
print('✅ FastSAM model cached')

import easyocr
reader = easyocr.Reader(['en', 'vi'])
print('✅ EasyOCR model cached')

from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
print('✅ BLIP-2 model cached')
"
fi

# Run preprocessing based on cloud provider
case $CLOUD_PROVIDER in
    "colab")
        echo "☁️ Running on Google Colab..."
        python pipeline.py build \
            --video_dir /content/drive/MyDrive/AIC_videos \
            --output_dir /content/artifacts \
            --target_frames 50 \
            --batch_size 32 \
            --enable_ocr --enable_captions
        
        # Copy to Google Drive
        cp -r /content/artifacts /content/drive/MyDrive/AIC_artifacts
        ;;
        
    "kaggle")
        echo "🏆 Running on Kaggle..."
        python pipeline.py build \
            --video_dir /kaggle/input/aic-video-dataset \
            --output_dir /kaggle/working/artifacts \
            --target_frames 50 \
            --batch_size 16 \
            --enable_ocr --enable_captions
        ;;
        
    "aws")
        echo "☁️ Running on AWS EC2..."
        # Download from S3
        aws s3 sync s3://your-bucket/videos ./videos
        
        # Process
        python pipeline.py build \
            --video_dir ./videos \
            --output_dir ./artifacts \
            --target_frames 50 \
            --batch_size 32 \
            --enable_ocr --enable_captions
        
        # Upload artifacts
        aws s3 sync ./artifacts s3://your-bucket/artifacts
        ;;
        
    "gcp")
        echo "☁️ Running on Google Cloud..."
        # Download from GCS
        gsutil -m cp -r gs://your-bucket/videos ./videos
        
        # Process
        python pipeline.py build \
            --video_dir ./videos \
            --output_dir ./artifacts \
            --target_frames 50 \
            --batch_size 32 \
            --enable_ocr --enable_captions
        
        # Upload artifacts
        gsutil -m cp -r ./artifacts gs://your-bucket/artifacts
        ;;
esac

echo "✅ Preprocessing complete!"
echo "📊 Artifacts ready for download"
echo "💾 Size estimate: 5-10GB (from ${DATASET_SIZE} videos)"
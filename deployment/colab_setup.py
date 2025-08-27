#!/usr/bin/env python3
"""
Google Colab setup script for AIC video preprocessing
Run this in a Colab notebook cell
"""

import os
import sys
from pathlib import Path

def setup_colab_environment():
    """Set up the environment in Google Colab"""
    
    print("🚀 Setting up AIC Video Preprocessing in Google Colab")
    print("=" * 60)
    
    # Mount Google Drive
    print("📁 Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
    except ImportError:
        print("⚠️  Not running in Colab, skipping drive mount")
    
    # Clone repository (if needed)
    if not Path("/content/AIC_FTML_dev").exists():
        print("📥 Cloning repository...")
        os.system("git clone https://github.com/your-username/AIC_FTML_dev.git /content/AIC_FTML_dev")
        os.chdir("/content/AIC_FTML_dev")
    else:
        os.chdir("/content/AIC_FTML_dev")
        print("✅ Repository already exists")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    os.system("pip install -r requirements.txt")
    
    # Pre-download models to cache them
    print("🤖 Pre-downloading models...")
    
    # CLIP model
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-16-SigLIP-256', 
            pretrained='webli'
        )
        print("✅ CLIP model cached")
    except Exception as e:
        print(f"⚠️  CLIP model download failed: {e}")
    
    # FastSAM model (optional)
    try:
        from ultralytics import FastSAM
        model = FastSAM('FastSAM-x.pt')
        print("✅ FastSAM model cached")
    except Exception as e:
        print(f"⚠️  FastSAM download failed: {e}")
    
    # EasyOCR model (optional)
    try:
        import easyocr
        reader = easyocr.Reader(['en', 'vi'])
        print("✅ EasyOCR model cached")
    except Exception as e:
        print(f"⚠️  EasyOCR download failed: {e}")
    
    # BLIP-2 model (optional)
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
        print("✅ BLIP-2 model cached")
    except Exception as e:
        print(f"⚠️  BLIP-2 download failed: {e}")
    
    print("\n🎯 Setup complete! Ready to process videos.")
    print("\n📋 Next steps:")
    print("1. Upload your videos to Google Drive: /content/drive/MyDrive/AIC_videos/")
    print("2. Run: python pipeline.py build --video_dir /content/drive/MyDrive/AIC_videos")
    print("3. Artifacts will be saved to: /content/drive/MyDrive/AIC_artifacts/")
    
    return True

def run_preprocessing():
    """Run the video preprocessing pipeline"""
    
    video_dir = "/content/drive/MyDrive/AIC_videos"
    output_dir = "/content/AIC_artifacts"
    drive_output = "/content/drive/MyDrive/AIC_artifacts"
    
    # Check if video directory exists
    if not Path(video_dir).exists():
        print(f"❌ Video directory not found: {video_dir}")
        print("Please upload videos to Google Drive first!")
        return False
    
    # Count videos
    video_files = list(Path(video_dir).rglob("*.mp4"))
    print(f"📹 Found {len(video_files)} video files")
    
    if len(video_files) == 0:
        print("❌ No .mp4 files found!")
        return False
    
    # Run preprocessing
    print("🚀 Starting video preprocessing...")
    
    cmd = f"""
    python pipeline.py build \
        --video_dir {video_dir} \
        --output_dir {output_dir} \
        --target_frames 50 \
        --batch_size 32 \
        --enable_ocr \
        --enable_captions
    """
    
    print(f"Running: {cmd}")
    result = os.system(cmd)
    
    if result == 0:
        print("✅ Preprocessing completed successfully!")
        
        # Copy to Google Drive for persistence
        print("💾 Copying artifacts to Google Drive...")
        os.system(f"cp -r {output_dir} {drive_output}")
        print(f"✅ Artifacts saved to: {drive_output}")
        
        # Show summary
        artifact_size = os.system(f"du -sh {drive_output}")
        print(f"📊 Artifact size: ~5-10GB")
        
        return True
    else:
        print("❌ Preprocessing failed!")
        return False

def download_artifacts_to_local():
    """Generate code to download artifacts to local machine"""
    
    print("\n📥 To download artifacts to your local machine:")
    print("=" * 50)
    print("""
    # Option 1: Direct download from Google Drive
    # 1. Go to Google Drive
    # 2. Find AIC_artifacts folder  
    # 3. Right-click → Download (will zip automatically)
    
    # Option 2: Using gdown (programmatic)
    pip install gdown
    
    # Get the Google Drive folder ID from the URL
    # Then download using gdown
    """)

if __name__ == "__main__":
    # Setup environment
    setup_colab_environment()
    
    # Ask user what to do
    print("\n🤔 What would you like to do?")
    print("1. Just setup (you'll run preprocessing manually)")
    print("2. Run preprocessing now")
    print("3. Show download instructions")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "2":
            run_preprocessing()
        elif choice == "3":
            download_artifacts_to_local()
        else:
            print("✅ Setup complete! Run preprocessing when ready.")
            
    except:
        print("✅ Setup complete! Run preprocessing manually when ready.")
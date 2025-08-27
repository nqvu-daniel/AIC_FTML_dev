#!/usr/bin/env python3
"""
Test preprocessing on 1/10th dataset in Google Colab
Run this first to validate everything works before Vast.ai
"""

import os
import sys
from pathlib import Path
import random

def setup_colab_test():
    """Setup for testing with subset of data"""
    
    print("🧪 AIC Video Preprocessing - Colab Test (1/10th Dataset)")
    print("=" * 60)
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except ImportError:
        print("⚠️  Not in Colab - assuming local test")
    
    # Setup paths
    full_video_dir = Path("/content/drive/MyDrive/AIC_videos")
    test_video_dir = Path("/content/test_videos")
    output_dir = Path("/content/test_artifacts")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    os.system("pip install -q -r requirements.txt")
    
    return full_video_dir, test_video_dir, output_dir

def create_test_subset(full_video_dir, test_video_dir, fraction=0.1):
    """Create subset of videos for testing"""
    
    print(f"📹 Creating test subset ({fraction*100:.0f}% of dataset)")
    
    if not full_video_dir.exists():
        print(f"❌ Full dataset not found: {full_video_dir}")
        print("Please upload videos to Google Drive first!")
        return False
    
    # Find all videos
    video_files = list(full_video_dir.rglob("*.mp4"))
    if not video_files:
        print("❌ No .mp4 files found!")
        return False
    
    print(f"📊 Found {len(video_files)} total videos")
    
    # Select random subset
    n_test = max(1, int(len(video_files) * fraction))
    test_files = random.sample(video_files, n_test)
    
    print(f"🎯 Selected {n_test} videos for testing")
    
    # Create test directory and copy files
    test_video_dir.mkdir(parents=True, exist_ok=True)
    
    for i, video_file in enumerate(test_files):
        dest = test_video_dir / f"test_{i:03d}_{video_file.name}"
        print(f"📁 Copying {video_file.name} -> {dest.name}")
        os.system(f"cp '{video_file}' '{dest}'")
    
    print(f"✅ Test subset ready: {n_test} videos in {test_video_dir}")
    return True

def run_test_preprocessing(test_video_dir, output_dir):
    """Run preprocessing on test subset"""
    
    print("🚀 Running preprocessing on test subset...")
    
    # Smaller parameters for testing
    cmd = f"""python pipeline.py build \
        --video_dir {test_video_dir} \
        --output_dir {output_dir} \
        --target_frames 20 \
        --batch_size 8 \
        --enable_ocr \
        --enable_captions"""
    
    print(f"🔧 Command: {cmd}")
    
    result = os.system(cmd)
    
    if result == 0:
        print("✅ Test preprocessing successful!")
        
        # Check outputs
        artifacts_dir = output_dir / "artifacts"
        if artifacts_dir.exists():
            print("📊 Generated artifacts:")
            for artifact in artifacts_dir.iterdir():
                size = os.path.getsize(artifact) / (1024*1024)  # MB
                print(f"   {artifact.name}: {size:.1f} MB")
            
            return True
    else:
        print("❌ Test preprocessing failed!")
        return False

def test_search_functionality(output_dir):
    """Test search functionality"""
    
    print("🔍 Testing search functionality...")
    
    try:
        # Import pipeline
        sys.path.append(str(Path.cwd() / "src"))
        from pipeline.unified_pipeline import UnifiedVideoPipeline
        
        # Initialize pipeline
        pipeline = UnifiedVideoPipeline(
            output_dir=output_dir,
            artifact_dir=output_dir / "artifacts"
        )
        
        # Test queries
        test_queries = [
            "person walking",
            "outdoor scene", 
            "text or writing",
            "indoor room"
        ]
        
        for query in test_queries:
            print(f"  🔍 Testing query: '{query}'")
            results = pipeline.search(query, k=5)
            print(f"     Found {len(results)} results")
            
            if results:
                top_result = results[0]
                print(f"     Top: {top_result.video_id} frame {top_result.frame_idx} (score: {top_result.score:.3f})")
        
        print("✅ Search functionality working!")
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def estimate_full_scale(test_results, test_videos, full_videos):
    """Estimate full-scale requirements"""
    
    print("📊 Estimating full-scale requirements...")
    
    if not test_results:
        print("⚠️  Cannot estimate - test failed")
        return
    
    scale_factor = full_videos / test_videos
    
    print(f"🔢 Scale factor: {scale_factor:.1f}x")
    print(f"📹 Test videos: {test_videos}")
    print(f"📹 Full videos: {full_videos}")
    
    # Estimate artifacts size
    test_artifacts_dir = Path("/content/test_artifacts/artifacts")
    if test_artifacts_dir.exists():
        test_size_mb = sum(os.path.getsize(f) for f in test_artifacts_dir.rglob("*") if f.is_file()) / (1024*1024)
        full_size_mb = test_size_mb * scale_factor
        
        print(f"💾 Test artifacts: {test_size_mb:.1f} MB")
        print(f"💾 Estimated full artifacts: {full_size_mb:.0f} MB ({full_size_mb/1024:.1f} GB)")
        
        # Estimate processing time (rough)
        if scale_factor > 1:
            estimated_hours = scale_factor * 0.5  # Assume 30min per 10% of dataset
            print(f"⏱️  Estimated processing time: {estimated_hours:.1f} hours")
            
            # Cost estimate for Vast.ai
            rtx4090_cost = 0.3  # $/hour estimate
            estimated_cost = estimated_hours * rtx4090_cost
            print(f"💰 Estimated Vast.ai cost (RTX4090): ${estimated_cost:.2f}")

def main():
    """Main test function"""
    
    # Setup
    full_video_dir, test_video_dir, output_dir = setup_colab_test()
    
    # Create test subset
    success = create_test_subset(full_video_dir, test_video_dir, fraction=0.1)
    if not success:
        return
    
    # Run test preprocessing
    success = run_test_preprocessing(test_video_dir, output_dir)
    if not success:
        return
    
    # Test search
    test_search_functionality(output_dir)
    
    # Estimate full scale
    test_videos = len(list(test_video_dir.glob("*.mp4")))
    full_videos = len(list(full_video_dir.glob("*.mp4"))) if full_video_dir.exists() else test_videos * 10
    estimate_full_scale(success, test_videos, full_videos)
    
    print("\n🎉 Test complete!")
    print("✅ Ready for full-scale deployment on Vast.ai")
    print("\n📋 Next steps:")
    print("1. Upload full dataset to accessible storage")
    print("2. Rent GPU on Vast.ai (RTX4090 recommended)")  
    print("3. Run: bash vast_ai_deploy.sh RTX4090 300GB true")
    print("4. Download artifacts (~5GB) when complete")

if __name__ == "__main__":
    main()
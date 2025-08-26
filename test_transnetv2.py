#!/usr/bin/env python3
"""
Simple test script to verify TransNetV2 implementation.
"""
import sys
from pathlib import Path
from src.segmentation.transnetv2 import segment_video, load_transnetv2_model

def test_transnetv2_loading():
    """Test if TransNetV2 can be loaded."""
    print("Testing TransNetV2 model loading...")
    try:
        model = load_transnetv2_model()
        if model is not None:
            print("✓ TransNetV2 model loaded successfully")
            return True
        else:
            print("✗ TransNetV2 model loading failed")
            return False
    except Exception as e:
        print(f"✗ TransNetV2 model loading error: {e}")
        return False

def test_segment_video_api():
    """Test the segment_video API with a dummy dataset."""
    print("\nTesting segment_video API...")
    
    # Create a dummy dataset structure
    test_root = Path("/tmp/test_dataset")
    videos_dir = test_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # This should fail gracefully since no video exists
        segments, fps, rep_frames = segment_video(
            dataset_root=test_root,
            video_id="nonexistent_video",
            use_transnetv2=False  # Use OpenCV fallback for testing
        )
        print("✗ Expected FileNotFoundError but didn't get one")
        return False
    except FileNotFoundError:
        print("✓ segment_video correctly handles missing video files")
        return True
    except Exception as e:
        print(f"✗ Unexpected error in segment_video: {e}")
        return False

def main():
    """Run all tests."""
    print("TransNetV2 Implementation Test\n" + "="*40)
    
    tests = [
        test_transnetv2_loading,
        test_segment_video_api,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*40)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
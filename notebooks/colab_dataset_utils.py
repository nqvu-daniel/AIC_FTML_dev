"""
Colab utilities for academic-grade dataset downloading with CSV filtering
Matches the pattern from your example for easy integration
"""

import os
import subprocess
import pathlib
import time
import csv
import tempfile
from typing import List


def filter_csv_for_videos(csv_path: str, video_list: List[str], output_path: str = None) -> str:
    """
    Filter the CSV file to only include entries for specified videos + essential metadata
    Returns path to filtered CSV file
    """
    if not pathlib.Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Create temp file if no output path specified
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        output_path = temp_file.name
        temp_file.close()

    filtered_rows = []
    total_rows = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            filtered_rows.append(header)

        for row in reader:
            total_rows += 1
            if not row or len(row) < 2:
                continue
            
            # Get filename from row (usually last or second-to-last column)
            filename = row[-2].strip() if len(row) >= 2 else ""
            filename_upper = filename.upper()

            # Always include essential metadata files (needed for all videos)
            essential_files = [
                'MAP-KEYFRAMES-AIC25-B1.ZIP',
                'MEDIA-INFO-AIC25-B1.ZIP',
                'OBJECTS-AIC25-B1.ZIP',
                'CLIP-FEATURES-32-AIC25-B1.ZIP',
                'CLIP-FEATURES-AIC25-B1.ZIP'
            ]

            is_essential = any(essential in filename_upper for essential in essential_files)
            is_target_video = any(vid.upper() in filename_upper for vid in video_list)

            if is_essential or is_target_video:
                filtered_rows.append(row)

    # Write filtered CSV
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(filtered_rows)

    print(f"📊 Filtered CSV: {len(filtered_rows)-1}/{total_rows} entries for videos {video_list} + essential metadata")
    return output_path


def download_aic_dataset_with_filtering(
    csv_file: str,
    dataset_root: str,
    videos: List[str] = None,
    test_mode: bool = True,
    skip_existing: bool = True
) -> bool:
    """
    Download AIC dataset with CSV filtering - matches your Colab example pattern
    """
    
    # Configuration
    if test_mode and not videos:
        videos = ['L21', 'L22']
        print("🧪 TEST MODE ENABLED: Only processing L21-L22")
    
    if not videos:
        videos = ['L21', 'L22', 'L23', 'L24', 'L25', 'L26', 'L27', 'L28', 'L29', 'L30']
    
    print(f"🏗️ Academic-Grade Dataset Download")
    print(f"📁 Dataset root: {dataset_root}")
    print(f"🎯 Videos to process: {videos}")
    print(f"📄 CSV file: {csv_file}")
    
    start_time = time.time()
    
    if not pathlib.Path(csv_file).exists():
        print(f"❌ CSV file {csv_file} not found")
        print("📝 Make sure 'AIC_2025_dataset_download_link.csv' exists in the current directory")
        return False

    try:
        # Create filtered CSV for our target videos
        filtered_csv_path = filter_csv_for_videos(csv_file, videos)
        
        print("🎯 Starting dataset download with progress tracking...")
        cmd = [
            'python', 'utils/dataset_downloader.py',
            '--dataset_root', dataset_root,
            '--csv', filtered_csv_path,
            '--skip-existing' if skip_existing else '--no-skip-existing'
        ]
        print(f"📝 Command: {' '.join(cmd)}")

        # Run with real-time output to show download progress
        import sys
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 universal_newlines=True, bufsize=1)

        print("📊 Download Progress:")
        print("-" * 50)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                sys.stdout.flush()

        return_code = process.poll()

        # Clean up temp file
        try:
            os.unlink(filtered_csv_path)
        except:
            pass

        if return_code != 0:
            print("❌ Dataset download failed!")
            return False

        elapsed = time.time() - start_time
        print(f"\n🎉 Dataset download completed in {elapsed:.1f} seconds")

        # Debug: Check what was actually extracted
        print("\n🔍 Verifying extracted structure:")
        dataset_path = pathlib.Path(dataset_root)
        if dataset_path.exists():
            subdirs = ['videos', 'keyframes', 'map_keyframes', 'media_info', 'objects', 'features']
            for subdir in subdirs:
                subdir_path = dataset_path / subdir
                if subdir_path.exists():
                    files = list(subdir_path.rglob('*'))
                    print(f"  📁 {subdir}/: {len(files)} items")
                    # Show first few items for verification
                    for item in files[:3]:
                        rel_path = item.relative_to(subdir_path)
                        item_type = "📂" if item.is_dir() else "📄"
                        print(f"    {item_type} {rel_path}")
                    if len(files) > 3:
                        print(f"    ... and {len(files) - 3} more")
                else:
                    print(f"  ⚠️ {subdir}/: NOT FOUND")
        
        print(f"\n✅ Academic dataset ready for TransNet-V2 processing!")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


# Simple usage example for Colab
def setup_aic_dataset_colab(
    csv_file: str = 'AIC_2025_dataset_download_link.csv',
    dataset_root: str = '/content/aic2025',
    test_mode: bool = True
) -> bool:
    """
    One-liner setup function for Colab notebooks
    """
    return download_aic_dataset_with_filtering(
        csv_file=csv_file,
        dataset_root=dataset_root,
        videos=['L21', 'L22'] if test_mode else None,
        test_mode=test_mode,
        skip_existing=True
    )


if __name__ == "__main__":
    # Example usage
    success = setup_aic_dataset_colab(
        csv_file='AIC_2025_dataset_download_link.csv',
        dataset_root='/content/aic2025',
        test_mode=True  # Only L21/L22 for academic competition
    )
    
    if success:
        print("🎉 Dataset setup complete!")
    else:
        print("❌ Dataset setup failed!")
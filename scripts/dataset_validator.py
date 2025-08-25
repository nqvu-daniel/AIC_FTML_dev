#!/usr/bin/env python3
"""
Smart dataset validation and preprocessing for AIC dataset.
Automatically detects issues, fixes problems, and optimizes structure.
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetValidator:
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.issues = []
        self.fixes_applied = []
        
    def validate_and_fix(self, auto_fix=True):
        """Run complete validation and auto-fix pipeline"""
        logger.info("Starting dataset validation and preprocessing...")
        
        # Step 1: Extract archives if needed
        self._extract_archives(auto_fix)
        
        # Step 2: Validate directory structure
        self._validate_structure(auto_fix)
        
        # Step 3: Validate video files
        self._validate_videos(auto_fix)
        
        # Step 4: Validate keyframes
        self._validate_keyframes(auto_fix)
        
        # Step 5: Validate metadata
        self._validate_metadata(auto_fix)
        
        # Step 6: Create missing directories
        self._create_missing_directories(auto_fix)
        
        # Report results
        self._generate_report()
        
        return len(self.issues) == 0
    
    def _extract_archives(self, auto_fix=True):
        """Extract ZIP archives automatically"""
        logger.info("Checking for ZIP archives to extract...")
        
        zip_files = list(self.dataset_root.glob("*.zip"))
        
        if not zip_files:
            logger.info("No ZIP archives found")
            return
        
        for zip_file in zip_files:
            extract_dir = self._determine_extract_location(zip_file)
            
            if extract_dir.exists() and any(extract_dir.iterdir()):
                logger.info(f"Skipping {zip_file.name} - already extracted")
                continue
                
            if auto_fix:
                logger.info(f"Extracting {zip_file.name} to {extract_dir}")
                try:
                    with zipfile.ZipFile(zip_file) as zf:
                        zf.extractall(extract_dir)
                    self.fixes_applied.append(f"Extracted {zip_file.name}")
                except Exception as e:
                    self.issues.append(f"Failed to extract {zip_file.name}: {e}")
            else:
                self.issues.append(f"Archive {zip_file.name} needs extraction")
    
    def _determine_extract_location(self, zip_file):
        """Smart extraction location detection"""
        name = zip_file.stem.lower()
        
        if "keyframes" in name:
            return self.dataset_root / "keyframes"
        elif "videos" in name:
            return self.dataset_root / "videos"
        elif "clip-features" in name:
            return self.dataset_root / "features"
        elif "media-info" in name:
            return self.dataset_root / "media_info"
        elif "objects" in name:
            return self.dataset_root / "objects"
        elif "map-keyframes" in name:
            return self.dataset_root / "map_keyframes"
        else:
            return self.dataset_root / "misc" / zip_file.stem
    
    def _validate_structure(self, auto_fix=True):
        """Validate and create directory structure"""
        logger.info("Validating directory structure...")
        
        required_dirs = [
            "videos",
            "keyframes", 
            "keyframes_all",
            "keyframes_intelligent",
            "map_keyframes",
            "media_info",
            "objects",
            "features",
            "artifacts"
        ]
        
        for dir_path in required_dirs:
            full_path = self.dataset_root / dir_path
            if not full_path.exists():
                if auto_fix:
                    full_path.mkdir(parents=True, exist_ok=True)
                    self.fixes_applied.append(f"Created directory {dir_path}")
                else:
                    self.issues.append(f"Missing directory: {dir_path}")
    
    def _validate_videos(self, auto_fix=True):
        """Validate video files"""
        logger.info("Validating video files...")
        
        videos_dir = self.dataset_root / "videos"
        if not videos_dir.exists():
            self.issues.append("Videos directory not found")
            return
        
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))
        
        if not video_files:
            self.issues.append("No video files found")
            return
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Validate each video
        corrupted_videos = []
        
        for video_file in tqdm(video_files, desc="Validating videos"):
            try:
                cap = cv2.VideoCapture(str(video_file))
                if not cap.isOpened():
                    corrupted_videos.append(video_file)
                    continue
                    
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                if frame_count == 0 or fps == 0:
                    corrupted_videos.append(video_file)
                
                cap.release()
                
            except Exception as e:
                corrupted_videos.append(video_file)
                logger.warning(f"Error validating {video_file}: {e}")
        
        if corrupted_videos:
            for video in corrupted_videos:
                self.issues.append(f"Corrupted/unreadable video: {video.name}")
    
    def _validate_keyframes(self, auto_fix=True):
        """Validate keyframe structure"""
        logger.info("Validating keyframes...")
        
        keyframes_dir = self.dataset_root / "keyframes"
        if not keyframes_dir.exists():
            self.issues.append("Keyframes directory not found")
            return
        
        # Check keyframe subdirectories
        keyframe_dirs = [d for d in keyframes_dir.iterdir() if d.is_dir()]
        
        if not keyframe_dirs:
            self.issues.append("No keyframe subdirectories found")
            return
        
        logger.info(f"Found {len(keyframe_dirs)} keyframe collections")
        
        # Validate keyframe images
        for kf_dir in keyframe_dirs:
            png_files = list(kf_dir.glob("*.png"))
            jpg_files = list(kf_dir.glob("*.jpg"))
            
            total_images = len(png_files) + len(jpg_files)
            
            if total_images == 0:
                self.issues.append(f"No keyframe images in {kf_dir.name}")
                continue
            
            # Sample some images to check validity
            sample_images = (png_files + jpg_files)[:min(10, total_images)]
            corrupted_images = []
            
            for img_file in sample_images:
                try:
                    img = cv2.imread(str(img_file))
                    if img is None or img.size == 0:
                        corrupted_images.append(img_file)
                except Exception:
                    corrupted_images.append(img_file)
            
            if corrupted_images:
                self.issues.append(f"Corrupted keyframe images in {kf_dir.name}: {len(corrupted_images)} files")
    
    def _validate_metadata(self, auto_fix=True):
        """Validate metadata files"""
        logger.info("Validating metadata...")
        
        meta_dir = self.dataset_root / "meta"
        if not meta_dir.exists():
            self.issues.append("Meta directory not found")
            return
        
        # Check for mapping files in map_keyframes directory
        map_keyframes_dir = self.dataset_root / "map_keyframes"
        if map_keyframes_dir.exists():
            mapping_files = list(map_keyframes_dir.glob("*.csv"))
            if not mapping_files:
                self.issues.append("No keyframe mapping files found")
            else:
                # Validate mapping file structure
                for mapping_file in mapping_files:
                    try:
                        df = pd.read_csv(mapping_file)
                        required_columns = ['n', 'pts_time', 'fps', 'frame_idx']
                        
                        missing_cols = [col for col in required_columns if col not in df.columns]
                        if missing_cols:
                            self.issues.append(f"Mapping file {mapping_file.name} missing columns: {missing_cols}")
                            
                    except Exception as e:
                        self.issues.append(f"Error reading mapping file {mapping_file.name}: {e}")
        
        # Check for media info files in media_info directory
        media_info_dir = self.dataset_root / "media_info"
        if media_info_dir.exists():
            info_files = list(media_info_dir.glob("*.json"))
            if info_files:
                logger.info(f"Found {len(info_files)} media info files")
                
                # Validate JSON structure
                for info_file in info_files:
                    try:
                        with open(info_file) as f:
                            data = json.load(f)
                        # Check if it's valid JSON
                    except Exception as e:
                        self.issues.append(f"Invalid media info file {info_file.name}: {e}")
        
        # Check objects directory
        objects_dir = self.dataset_root / "objects"
        if objects_dir.exists():
            object_subdirs = [d for d in objects_dir.iterdir() if d.is_dir()]
            logger.info(f"Found {len(object_subdirs)} object detection collections")
    
    def _create_missing_directories(self, auto_fix=True):
        """Create any missing required directories"""
        if auto_fix:
            dirs_to_create = [
                "artifacts",
                "submissions", 
                "logs",
                "temp"
            ]
            
            for dir_name in dirs_to_create:
                dir_path = self.dataset_root / dir_name
                if not dir_path.exists():
                    dir_path.mkdir(exist_ok=True)
                    self.fixes_applied.append(f"Created directory {dir_name}")
    
    def _generate_report(self):
        """Generate validation report"""
        logger.info("\n" + "="*50)
        logger.info("DATASET VALIDATION REPORT")
        logger.info("="*50)
        
        if not self.issues:
            logger.info("✅ Dataset validation PASSED - No issues found!")
        else:
            logger.warning(f"❌ Dataset validation found {len(self.issues)} issues:")
            for i, issue in enumerate(self.issues, 1):
                logger.warning(f"  {i}. {issue}")
        
        if self.fixes_applied:
            logger.info(f"\n✅ Applied {len(self.fixes_applied)} automatic fixes:")
            for i, fix in enumerate(self.fixes_applied, 1):
                logger.info(f"  {i}. {fix}")
        
        # Generate summary statistics
        self._generate_statistics()
    
    def _generate_statistics(self):
        """Generate dataset statistics"""
        logger.info("\nDATASET STATISTICS:")
        logger.info("-" * 30)
        
        # Count videos
        videos_dir = self.dataset_root / "videos"
        if videos_dir.exists():
            video_count = len(list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi")))
            logger.info(f"Video files: {video_count}")
        
        # Count keyframe collections
        keyframes_dir = self.dataset_root / "keyframes"
        if keyframes_dir.exists():
            kf_dirs = [d for d in keyframes_dir.iterdir() if d.is_dir()]
            logger.info(f"Keyframe collections: {len(kf_dirs)}")
            
            # Count total keyframes
            total_keyframes = 0
            for kf_dir in kf_dirs:
                png_count = len(list(kf_dir.glob("*.png")))
                jpg_count = len(list(kf_dir.glob("*.jpg")))
                total_keyframes += png_count + jpg_count
            logger.info(f"Total keyframes: {total_keyframes}")
        
        # Count metadata files
        map_keyframes_dir = self.dataset_root / "map_keyframes"
        if map_keyframes_dir.exists():
            mapping_count = len(list(map_keyframes_dir.glob("*.csv")))
            logger.info(f"Mapping files: {mapping_count}")
        
        media_info_dir = self.dataset_root / "media_info"
        if media_info_dir.exists():
            info_count = len(list(media_info_dir.glob("*.json")))
            logger.info(f"Media info files: {info_count}")
        
        # Check for features
        features_dir = self.dataset_root / "features"
        if features_dir.exists():
            feature_files = list(features_dir.glob("*.npy"))
            logger.info(f"Precomputed feature files: {len(feature_files)}")

def main():
    parser = argparse.ArgumentParser(description="Dataset validation and preprocessing")
    parser.add_argument("dataset_root", help="Root directory of AIC dataset")
    parser.add_argument("--no-auto-fix", action="store_true", help="Don't automatically fix issues")
    parser.add_argument("--extract-only", action="store_true", help="Only extract archives")
    
    args = parser.parse_args()
    
    if not Path(args.dataset_root).exists():
        logger.error(f"Dataset root does not exist: {args.dataset_root}")
        return 1
    
    validator = DatasetValidator(args.dataset_root)
    
    if args.extract_only:
        validator._extract_archives(not args.no_auto_fix)
        return 0
    
    success = validator.validate_and_fix(not args.no_auto_fix)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
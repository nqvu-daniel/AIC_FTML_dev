#!/usr/bin/env python3
"""
Automated intelligent video processing pipeline for AIC dataset.
Discovers videos, validates structure, and processes with smart algorithms.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoVideoProcessor:
    def __init__(self, dataset_root, max_workers=None):
        self.dataset_root = Path(dataset_root)
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.discovered_videos = []
        
    def discover_videos(self):
        """Automatically discover all video files and collections in dataset"""
        logger.info("Discovering videos in dataset...")
        
        # Look for video patterns
        video_patterns = [
            "Videos_*.zip",
            "*.mp4", "*.avi", "*.mov", "*.mkv",
            "L*_V*.mp4",  # AIC pattern
        ]
        
        videos = set()
        
        # Scan for video files
        for pattern in video_patterns:
            matches = list(self.dataset_root.glob(f"**/{pattern}"))
            for match in matches:
                if "Videos_" in match.name:
                    # Extract L## pattern from Videos_L##_*.zip
                    video_id = match.name.split('_')[1] 
                    videos.add(video_id)
                elif match.suffix in ['.mp4', '.avi', '.mov', '.mkv']:
                    # Direct video file
                    video_id = match.stem.split('_')[0]  # L21_V001 -> L21
                    videos.add(video_id)
        
        # Also check keyframes directories for video IDs
        keyframes_dir = self.dataset_root / "keyframes"
        if keyframes_dir.exists():
            for subdir in keyframes_dir.iterdir():
                if subdir.is_dir():
                    video_id = subdir.name.split('_')[0]  # L21_V001 -> L21
                    videos.add(video_id)
        
        # Look for precomputed keyframes
        for pattern in ["Keyframes_*.zip"]:
            matches = list(self.dataset_root.glob(pattern))
            for match in matches:
                video_id = match.name.split('_')[1].replace('.zip', '')
                videos.add(video_id)
        
        self.discovered_videos = sorted(list(videos))
        logger.info(f"Discovered {len(self.discovered_videos)} video collections: {self.discovered_videos}")
        return self.discovered_videos
    
    def validate_dataset_structure(self):
        """Validate dataset has required components"""
        logger.info("Validating dataset structure...")
        
        issues = []
        recommendations = []
        
        # Check for essential directories/files
        required_paths = [
            ("videos", "directory"),
            ("keyframes", "directory"), 
            ("meta", "directory")
        ]
        
        for path_name, path_type in required_paths:
            path = self.dataset_root / path_name
            if not path.exists():
                issues.append(f"Missing {path_type}: {path}")
            
        # Check for metadata files
        meta_dir = self.dataset_root / "meta"
        if meta_dir.exists():
            # Look for mapping and info files
            map_files = list(meta_dir.glob("*.map_keyframe.csv"))
            info_files = list(meta_dir.glob("*.media_info.json"))
            
            if not map_files:
                issues.append("No keyframe mapping files found")
            if not info_files:
                recommendations.append("No media info files - hybrid search will be limited")
        
        # Check for precomputed features
        features_dir = self.dataset_root / "features"
        clip_features = list(self.dataset_root.glob("**/clip-features*.zip"))
        
        if features_dir.exists() or clip_features:
            recommendations.append("Found precomputed features - will use for faster processing")
        
        # Report validation results
        if issues:
            logger.warning(f"Dataset validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        if recommendations:
            logger.info("Dataset optimization opportunities:")
            for rec in recommendations:
                logger.info(f"  + {rec}")
        
        return len(issues) == 0
    
    def check_processing_status(self):
        """Check what's already been processed to avoid redundant work"""
        artifacts_dir = Path("./artifacts")
        
        status = {
            "index_built": False,
            "text_corpus_built": False,
            "intelligent_frames_extracted": False,
            "models_trained": False
        }
        
        if artifacts_dir.exists():
            # Check for index files
            if list(artifacts_dir.glob("*.index")) or list(artifacts_dir.glob("faiss_*")):
                status["index_built"] = True
                
            # Check for text corpus
            if list(artifacts_dir.glob("*corpus*.jsonl")):
                status["text_corpus_built"] = True
                
            # Check for trained models
            if list(artifacts_dir.glob("reranker*.joblib")):
                status["models_trained"] = True
        
        # Check for intelligent keyframes
        intelligent_dir = self.dataset_root / "keyframes_intelligent"
        if intelligent_dir.exists() and any(intelligent_dir.iterdir()):
            status["intelligent_frames_extracted"] = True
        
        logger.info(f"Processing status: {status}")
        return status
    
    def run_intelligent_sampling(self, video_ids, force_reprocess=False):
        """Run intelligent frame sampling on discovered videos"""
        logger.info(f"Running intelligent sampling on {len(video_ids)} video collections...")
        
        status = self.check_processing_status()
        if status["intelligent_frames_extracted"] and not force_reprocess:
            logger.info("Intelligent frames already extracted. Use --force to reprocess.")
            return True
        
        try:
            cmd = [
                sys.executable, "src/sampling/frames_intelligent_fast.py",
                "--dataset_root", str(self.dataset_root),
                "--videos"] + video_ids + [
                "--mode", "ultra_fast",
                "--use_gpu"  # Try GPU acceleration
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Intelligent sampling completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Intelligent sampling failed: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return False
    
    def build_search_index(self, video_ids, force_reprocess=False):
        """Build search index for all videos"""
        logger.info("Building search index...")
        
        status = self.check_processing_status()
        if status["index_built"] and not force_reprocess:
            logger.info("Search index already built. Use --force to rebuild.")
            return True
        
        try:
            cmd = [
                sys.executable, "scripts/index.py",
                "--dataset_root", str(self.dataset_root),
                "--videos"] + video_ids
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Search index built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Index building failed: {e}")
            return False
    
    def build_text_corpus(self, video_ids, force_reprocess=False):
        """Build text corpus for hybrid search"""
        logger.info("Building text corpus for hybrid search...")
        
        status = self.check_processing_status()
        if status["text_corpus_built"] and not force_reprocess:
            logger.info("Text corpus already built. Use --force to rebuild.")
            return True
        
        try:
            cmd = [
                sys.executable, "scripts/build_text.py", 
                "--dataset_root", str(self.dataset_root),
                "--videos"] + video_ids
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Text corpus built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Text corpus building failed: {e}")
            return False
    
    def process_all(self, force_reprocess=False, skip_intelligent=False):
        """Run complete automated processing pipeline"""
        logger.info("Starting automated video processing pipeline")
        
        # Step 1: Discover videos
        video_ids = self.discover_videos()
        if not video_ids:
            logger.error("No videos discovered in dataset")
            return False
        
        # Step 2: Validate dataset
        if not self.validate_dataset_structure():
            logger.warning("Dataset validation failed - proceeding anyway")
        
        # Step 3: Intelligent sampling (optional)
        if not skip_intelligent:
            if not self.run_intelligent_sampling(video_ids, force_reprocess):
                logger.warning("Intelligent sampling failed - continuing with existing keyframes")
        
        # Step 4: Build search index
        if not self.build_search_index(video_ids, force_reprocess):
            logger.error("Failed to build search index")
            return False
        
        # Step 5: Build text corpus
        if not self.build_text_corpus(video_ids, force_reprocess):
            logger.warning("Failed to build text corpus - hybrid search may be limited")
        
        logger.info("Automated processing pipeline completed successfully!")
        logger.info(f"Processed {len(video_ids)} video collections")
        logger.info("You can now run searches with:")
        logger.info("  python src/retrieval/search_hybrid_rerank.py --index_dir ./artifacts --query 'your query' --topk 100")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Automated intelligent video processing")
    parser.add_argument("dataset_root", help="Root directory of AIC dataset")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if already done")
    parser.add_argument("--skip-intelligent", action="store_true", help="Skip intelligent sampling step")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--discover-only", action="store_true", help="Only discover videos, don't process")
    
    args = parser.parse_args()
    
    if not Path(args.dataset_root).exists():
        logger.error(f"Dataset root does not exist: {args.dataset_root}")
        return 1
    
    processor = AutoVideoProcessor(args.dataset_root, args.workers)
    
    if args.discover_only:
        processor.discover_videos()
        processor.validate_dataset_structure()
        return 0
    
    success = processor.process_all(args.force, args.skip_intelligent)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
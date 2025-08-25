#!/usr/bin/env python3
"""
Automated intelligent video processing pipeline.
One command to rule them all - discovers, validates, and processes everything smartly.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import subprocess
import multiprocessing
from tqdm import tqdm
import logging

# Setup basic logging to stdout; file handler is added per-dataset in __init__
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class SmartPipeline:
    def __init__(self, dataset_root, max_workers=None, use_gpu=True):
        self.dataset_root = Path(dataset_root)
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.use_gpu = use_gpu
        self.start_time = time.time()
        
        # Create logs directory and attach file handler
        logs_dir = self.dataset_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(str(logs_dir / 'pipeline.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        # Avoid adding multiple handlers if running multiple times
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(fh)
    
    def log_progress(self, stage, message, elapsed_time=None):
        """Log progress with timing"""
        if elapsed_time is None:
            elapsed_time = time.time() - self.start_time
        
        logger.info(f"[{stage}] ({elapsed_time:.1f}s) {message}")
    
    def run_command(self, cmd, stage="Unknown", timeout=3600):
        """Run command with logging and error handling"""
        self.log_progress(stage, f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            self.log_progress(stage, "✅ Completed successfully")
            return True, result.stdout
            
        except subprocess.CalledProcessError as e:
            self.log_progress(stage, f"❌ Failed with exit code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return False, e.stderr
            
        except subprocess.TimeoutExpired:
            self.log_progress(stage, f"❌ Timed out after {timeout}s")
            return False, "Timeout"
    
    def validate_dataset(self):
        """Step 1: Validate and preprocess dataset"""
        self.log_progress("VALIDATION", "Starting dataset validation and preprocessing...")
        
        cmd = [
            sys.executable, "scripts/dataset_validator.py",
            str(self.dataset_root)
        ]
        
        success, output = self.run_command(cmd, "VALIDATION")
        return success
    
    def run_intelligent_sampling(self):
        """Step 2: Run advanced intelligent sampling"""
        self.log_progress("SAMPLING", "Starting advanced intelligent frame sampling...")
        
        # Discover videos using auto_process script
        cmd = [sys.executable, "scripts/auto_process.py", str(self.dataset_root)]
        success, output = self.run_command(cmd, "DISCOVERY")
        
        if not success:
            self.log_progress("SAMPLING", "❌ Video discovery failed")
            return False
        
        # Parse discovered video IDs from output
        discovered_videos = [line.strip() for line in output.strip().split('\n') if line.strip() and line.strip().startswith('L')]
        
        if not discovered_videos:
            self.log_progress("SAMPLING", "❌ No videos discovered")
            return False
        
        self.log_progress("SAMPLING", f"Processing {len(discovered_videos)} video collections: {discovered_videos}")
        
        cmd = [
            sys.executable, "src/sampling/frames_intelligent.py",
            "--dataset_root", str(self.dataset_root),
            "--videos"] + discovered_videos + [
            "--target_fps", "0.5"
        ]
        
        if self.use_gpu:
            cmd.append("--use_gpu")
        
        success, output = self.run_command(cmd, "SAMPLING", timeout=1800)  # 30 minutes max
        return success
    
    def build_search_infrastructure(self):
        """Step 3: Build search index and text corpus"""
        self.log_progress("INDEXING", "Building search infrastructure...")
        
        discovered_videos = self._discover_video_ids()
        
        # Build search index
        cmd = [
            sys.executable, "scripts/index.py",
            "--dataset_root", str(self.dataset_root),
            "--videos"] + discovered_videos
        
        success, output = self.run_command(cmd, "INDEXING", timeout=3600)
        if not success:
            return False
        
        # Build text corpus for hybrid search
        cmd = [
            sys.executable, "scripts/build_text.py",
            "--dataset_root", str(self.dataset_root),
            "--videos"] + discovered_videos
        
        success, output = self.run_command(cmd, "TEXT_CORPUS", timeout=1800)
        return success
    
    def train_models(self):
        """Step 4: Train re-ranking models (if training data available)"""
        self.log_progress("TRAINING", "Checking for training data...")
        
        # Look for training data
        train_files = list(self.dataset_root.glob("**/train*.jsonl"))
        train_files.extend(list(self.dataset_root.glob("**/dev*.jsonl")))
        
        if not train_files:
            self.log_progress("TRAINING", "No training data found - skipping model training")
            return True
        
        self.log_progress("TRAINING", f"Found training data: {[f.name for f in train_files]}")
        
        # Train re-ranker
        for train_file in train_files:
            cmd = [
                sys.executable, "src/training/train_reranker.py",
                "--index_dir", "./artifacts",
                "--train_jsonl", str(train_file)
            ]
            
            success, output = self.run_command(cmd, "TRAINING")
            if success:
                break  # Use first successful training
        
        return True  # Don't fail pipeline if training fails
    
    def _discover_video_ids(self):
        """Discover video IDs in dataset"""
        video_ids = set()
        
        # Check videos directory
        videos_dir = self.dataset_root / "videos"
        if videos_dir.exists():
            for video_file in videos_dir.glob("*.mp4"):
                video_id = video_file.stem.split('_')[0]  # L21_V001 -> L21
                video_ids.add(video_id)
        
        # Check keyframes directory
        keyframes_dir = self.dataset_root / "keyframes"
        if keyframes_dir.exists():
            for subdir in keyframes_dir.iterdir():
                if subdir.is_dir():
                    video_id = subdir.name.split('_')[0]  # L21_V001 -> L21
                    video_ids.add(video_id)
        
        # Check for ZIP files
        for zip_file in self.dataset_root.glob("*L*.zip"):
            try:
                # Extract L## pattern from various naming conventions
                name = zip_file.stem
                if "L" in name:
                    parts = name.split('_')
                    for part in parts:
                        if part.startswith('L') and len(part) >= 3:
                            video_ids.add(part[:3])  # L21, L22, etc.
            except Exception:
                continue
        
        return sorted(list(video_ids))
    
    def run_test_search(self):
        """Step 5: Run test search to verify everything works"""
        self.log_progress("TESTING", "Running test search to verify pipeline...")
        
        test_queries = [
            "person walking",
            "car driving", 
            "building",
            "people talking"
        ]
        
        for query in test_queries:
            cmd = [
                sys.executable, "src/retrieval/search_hybrid_rerank.py",
                "--index_dir", "./artifacts",
                "--query", query,
                "--topk", "10"
            ]
            
            success, output = self.run_command(cmd, "TESTING", timeout=300)
            if success:
                self.log_progress("TESTING", f"✅ Test search successful for '{query}'")
                return True
            else:
                self.log_progress("TESTING", f"❌ Test search failed for '{query}'")
        
        return False
    
    def generate_final_report(self):
        """Generate final pipeline report"""
        total_time = time.time() - self.start_time
        
        self.log_progress("COMPLETE", f"Pipeline completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        # Generate statistics
        stats = self._collect_statistics()
        
        logger.info("\n" + "="*60)
        logger.info("SMART PIPELINE COMPLETION REPORT")
        logger.info("="*60)
        
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        logger.info("\nNext steps:")
        logger.info("1. Run searches:")
        logger.info("   python src/retrieval/search_hybrid_rerank.py --index_dir ./artifacts --query 'your query' --topk 100")
        logger.info("2. Export results:")
        logger.info("   python src/retrieval/export_csv.py --index_dir ./artifacts --query 'query' --outfile result.csv")
        logger.info("3. Evaluate:")
        logger.info("   python eval/evaluate.py --gt ground_truth.json --pred_dir submissions/ --task kis")
        
        # Save report to file
        report_file = self.dataset_root / "logs" / "pipeline_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'total_time_seconds': total_time,
                'statistics': stats,
                'timestamp': time.time()
            }, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {report_file}")
    
    def _collect_statistics(self):
        """Collect pipeline statistics"""
        stats = {}
        
        # Count videos
        videos_dir = self.dataset_root / "videos"
        if videos_dir.exists():
            video_count = len(list(videos_dir.glob("*.mp4")))
            stats["Videos processed"] = video_count
        
        # Count intelligent keyframes
        intelligent_dir = self.dataset_root / "keyframes_intelligent"
        if intelligent_dir.exists():
            total_intelligent = 0
            collections = 0
            for subdir in intelligent_dir.iterdir():
                if subdir.is_dir():
                    collections += 1
                    total_intelligent += len(list(subdir.glob("*.png")))
            stats["Intelligent keyframe collections"] = collections
            stats["Total intelligent keyframes"] = total_intelligent
        
        # Check artifacts
        artifacts_dir = Path("./artifacts")
        if artifacts_dir.exists():
            stats["Index files"] = len(list(artifacts_dir.glob("*.index"))) + len(list(artifacts_dir.glob("faiss_*")))
            stats["Text corpus files"] = len(list(artifacts_dir.glob("*corpus*.jsonl")))
            stats["Trained models"] = len(list(artifacts_dir.glob("*.joblib")))
        
        return stats
    
    def run_full_pipeline(self):
        """Run the complete automated pipeline"""
        logger.info("Starting Smart Video Processing Pipeline")
        logger.info(f"Dataset: {self.dataset_root}")
        logger.info(f"Workers: {self.max_workers}")
        logger.info(f"GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
        
        steps = [
            ("Dataset Validation", self.validate_dataset),
            ("Intelligent Sampling", self.run_intelligent_sampling), 
            ("Search Infrastructure", self.build_search_infrastructure),
            ("Model Training", self.train_models),
            ("Pipeline Testing", self.run_test_search)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            try:
                self.log_progress("PIPELINE", f"Starting: {step_name}")
                success = step_func()
                
                if success:
                    self.log_progress("PIPELINE", f"✅ Completed: {step_name}")
                else:
                    self.log_progress("PIPELINE", f"❌ Failed: {step_name}")
                    failed_steps.append(step_name)
                    
            except Exception as e:
                self.log_progress("PIPELINE", f"❌ Error in {step_name}: {e}")
                failed_steps.append(step_name)
        
        # Generate final report
        self.generate_final_report()
        
        if failed_steps:
            logger.warning(f"Pipeline completed with {len(failed_steps)} failed steps: {failed_steps}")
            return False
        else:
            logger.info("🎉 Smart pipeline completed successfully!")
            return True

def main():
    parser = argparse.ArgumentParser(description="Smart automated video processing pipeline")
    parser.add_argument("dataset_root", help="Root directory of AIC dataset")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--skip-validation", action="store_true", help="Skip dataset validation step")
    parser.add_argument("--skip-sampling", action="store_true", help="Skip intelligent sampling step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    
    args = parser.parse_args()
    
    if not Path(args.dataset_root).exists():
        logger.error(f"Dataset root does not exist: {args.dataset_root}")
        return 1
    
    pipeline = SmartPipeline(
        args.dataset_root, 
        args.workers, 
        not args.no_gpu
    )
    
    success = pipeline.run_full_pipeline()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

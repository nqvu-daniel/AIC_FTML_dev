"""
Colab utilities for AIC FTML pipeline
Provides helper functions for dataset setup, search results display, and evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import tempfile
from typing import List, Dict, Any
from IPython.display import display, HTML


def setup_aic_dataset(csv_file: Path, dataset_dir: Path, use_sample: bool = True, sample_size: int = 10) -> pd.DataFrame:
    """
    Setup AIC dataset from CSV file or create demo data.
    
    Args:
        csv_file: Path to CSV file with download links
        dataset_dir: Directory to store dataset
        use_sample: Whether to use sample data for testing
        sample_size: Number of sample videos
        
    Returns:
        DataFrame with video information
    """
    if use_sample or not csv_file.exists():
        print(f"📝 Using demo dataset ({sample_size} sample videos)")
        # Create demo dataset info
        demo_videos = []
        for i in range(1, sample_size + 1):
            demo_videos.append({
                'video_id': f'L21_V{i:03d}',
                'title': f'AIC Demo Video {i}',
                'duration': 120.0 + (i * 30),
                'fps': 25.0,
                'frames': int((120.0 + (i * 30)) * 25.0)
            })
        return pd.DataFrame(demo_videos)
    
    else:
        print(f"📊 Loading dataset info from {csv_file}")
        # Load actual CSV and process
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(df)} entries from CSV")
            return df
        except Exception as e:
            print(f"⚠️ Error loading CSV: {e}")
            return setup_aic_dataset(csv_file, dataset_dir, use_sample=True, sample_size=sample_size)


def display_search_results(results: List[Any], query: str, max_display: int, keyframes_dir: Path) -> None:
    """
    Display search results in a formatted way for notebooks.
    
    Args:
        results: List of search result objects
        query: Original search query
        max_display: Maximum number of results to display
        keyframes_dir: Directory containing keyframe images
    """
    print(f"🔍 Search Results for: '{query}'")
    print("=" * 60)
    
    for i, result in enumerate(results[:max_display]):
        print(f"\n#{i+1}. Video: {result.video_id}")
        print(f"   Frame: {result.frame_idx}")
        print(f"   Score: {result.score:.4f}")
        
        if hasattr(result, 'metadata') and result.metadata:
            print(f"   Metadata: {result.metadata}")
        
        # Try to display keyframe image if available
        keyframe_path = keyframes_dir / result.video_id / f"frame_{result.frame_idx:06d}.jpg"
        if keyframe_path.exists():
            try:
                from IPython.display import Image as IPImage
                display(IPImage(filename=str(keyframe_path), width=200))
            except:
                print(f"   🖼️ Keyframe: {keyframe_path}")
        else:
            print(f"   📁 Keyframe path: {keyframe_path} (not found)")


def export_search_results(results: List[Any], query: str, format_type: str = 'csv') -> str:
    """
    Export search results to file.
    
    Args:
        results: List of search result objects
        query: Original search query
        format_type: Export format ('csv', 'json', 'parquet')
        
    Returns:
        Filename of exported results
    """
    # Create export data
    export_data = []
    for result in results:
        row = {
            'query': query,
            'video_id': result.video_id,
            'frame_idx': result.frame_idx,
            'score': result.score,
            'timestamp': getattr(result, 'timestamp', result.frame_idx * 2.0)
        }
        if hasattr(result, 'metadata') and result.metadata:
            row.update(result.metadata)
        export_data.append(row)
    
    # Generate filename
    safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_query = safe_query.replace(' ', '_')[:50]
    timestamp = int(time.time())
    filename = f"search_results_{safe_query}_{timestamp}.{format_type}"
    
    # Export based on format
    df = pd.DataFrame(export_data)
    
    if format_type == 'csv':
        df.to_csv(filename, index=False)
    elif format_type == 'json':
        df.to_json(filename, orient='records', indent=2)
    elif format_type == 'parquet':
        df.to_parquet(filename, index=False)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    print(f"📄 Exported {len(results)} results to {filename}")
    return filename


def evaluate_search_performance(search_function) -> pd.DataFrame:
    """
    Evaluate search performance with sample queries.
    
    Args:
        search_function: Function to test (takes query, mode, k parameters)
        
    Returns:
        DataFrame with evaluation results
    """
    test_queries = [
        "news anchor speaking",
        "weather forecast",
        "sports highlights",
        "interview scene",
        "outdoor reporting"
    ]
    
    search_modes = ['hybrid', 'vector', 'text']
    results = []
    
    print("📊 Evaluating search performance...")
    
    for mode in search_modes:
        for query in test_queries:
            start_time = time.time()
            try:
                search_results = search_function(query, mode=mode, k=20)
                search_time = (time.time() - start_time) * 1000
                
                # Calculate metrics
                avg_score = np.mean([r.score for r in search_results]) if search_results else 0
                diversity = len(set(r.video_id for r in search_results)) / len(search_results) if search_results else 0
                
                results.append({
                    'query': query,
                    'mode': mode,
                    'num_results': len(search_results),
                    'search_time_ms': search_time,
                    'avg_score': avg_score,
                    'diversity': diversity
                })
            except Exception as e:
                print(f"⚠️ Error testing {mode} with '{query}': {e}")
                results.append({
                    'query': query,
                    'mode': mode,
                    'num_results': 0,
                    'search_time_ms': 0,
                    'avg_score': 0,
                    'diversity': 0
                })
    
    return pd.DataFrame(results)


def plot_performance_comparison(eval_results: pd.DataFrame) -> None:
    """
    Plot performance comparison charts.
    
    Args:
        eval_results: DataFrame with evaluation results
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Search Performance Analysis', fontsize=16)
        
        # Average search time by mode
        avg_time = eval_results.groupby('mode')['search_time_ms'].mean()
        axes[0, 0].bar(avg_time.index, avg_time.values)
        axes[0, 0].set_title('Average Search Time (ms)')
        axes[0, 0].set_ylabel('Time (ms)')
        
        # Average score by mode
        avg_score = eval_results.groupby('mode')['avg_score'].mean()
        axes[0, 1].bar(avg_score.index, avg_score.values)
        axes[0, 1].set_title('Average Result Score')
        axes[0, 1].set_ylabel('Score')
        
        # Diversity by mode
        avg_diversity = eval_results.groupby('mode')['diversity'].mean()
        axes[1, 0].bar(avg_diversity.index, avg_diversity.values)
        axes[1, 0].set_title('Result Diversity')
        axes[1, 0].set_ylabel('Diversity (unique videos / total)')
        
        # Search time distribution
        for mode in eval_results['mode'].unique():
            mode_data = eval_results[eval_results['mode'] == mode]['search_time_ms']
            axes[1, 1].hist(mode_data, alpha=0.7, label=mode, bins=10)
        axes[1, 1].set_title('Search Time Distribution')
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("📊 Matplotlib not available for plotting")
    except Exception as e:
        print(f"📊 Plotting error: {e}")


def create_training_data_sample(metadata_df: pd.DataFrame, num_examples: int = 30) -> List[Dict[str, Any]]:
    """
    Create sample training data for reranking models.
    
    Args:
        metadata_df: DataFrame with video metadata
        num_examples: Number of training examples to create
        
    Returns:
        List of training examples
    """
    training_data = []
    
    # Sample queries based on video content
    query_templates = [
        "news anchor speaking",
        "weather forecast",
        "sports highlights", 
        "interview scene",
        "outdoor reporting",
        "indoor studio",
        "people talking",
        "text on screen",
        "graphics and charts",
        "live broadcast"
    ]
    
    for i in range(num_examples):
        # Pick random query and video
        query = np.random.choice(query_templates)
        
        if len(metadata_df) > 0:
            # Use actual metadata
            video_sample = metadata_df.sample(1).iloc[0]
            video_id = video_sample.get('video_id', f'V{i:03d}')
        else:
            # Use demo data
            video_id = f'L21_V{i%10 + 1:03d}'
        
        # Create positive example
        positive_frame = np.random.randint(0, 1000)
        
        training_example = {
            'query': query,
            'positives': [
                {
                    'video_id': video_id,
                    'frame_idx': positive_frame,
                    'score': 0.8 + np.random.random() * 0.2
                }
            ],
            'negatives': [
                {
                    'video_id': video_id,
                    'frame_idx': positive_frame + np.random.randint(50, 200),
                    'score': np.random.random() * 0.3
                }
            ]
        }
        training_data.append(training_example)
    
    return training_data


def save_artifacts_summary(artifact_dir: Path) -> Dict[str, Any]:
    """
    Save summary of generated artifacts.
    
    Args:
        artifact_dir: Directory containing artifacts
        
    Returns:
        Summary dictionary
    """
    artifact_dir = Path(artifact_dir)
    summary = {
        'timestamp': time.time(),
        'artifacts': []
    }
    
    if artifact_dir.exists():
        for file_path in artifact_dir.rglob('*'):
            if file_path.is_file():
                summary['artifacts'].append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'type': file_path.suffix
                })
    
    # Save summary
    summary_file = artifact_dir / 'artifacts_summary.json'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Artifacts summary saved to {summary_file}")
    return summary
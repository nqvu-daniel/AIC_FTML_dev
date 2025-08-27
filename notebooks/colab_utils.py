"""
Utility functions for the AIC FTML Colab notebook
Keeps the notebook clean by moving large functions here
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from IPython.display import display, HTML, Image as IPImage
import matplotlib.pyplot as plt
import seaborn as sns


def download_with_progress(url, filename):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def setup_aic_dataset(csv_file, dataset_dir, use_sample=True, sample_size=10):
    """Setup AIC dataset from CSV or create sample for demo"""
    
    if csv_file.exists():
        print(f"✅ Found AIC dataset CSV: {csv_file}")
        df_links = pd.read_csv(csv_file)
        
        if use_sample:
            df_links = df_links.head(sample_size)
            print(f"📊 Using sample of {len(df_links)} videos")
        else:
            print(f"📊 Processing full dataset: {len(df_links)} videos")
        
        return df_links
    else:
        print(f"ℹ️ No AIC CSV found at {csv_file}")
        print("Creating sample metadata for demo...")
        
        # Create sample video metadata for testing
        sample_data = []
        for i in range(sample_size):
            sample_data.append({
                'video_id': f'L21_V{i+1:03d}',
                'title': f'HTV News Broadcast {i+1}',
                'description': 'Daily news program with anchor presenting current events',
                'keywords': 'news, HTV, anchor, broadcast, Vietnam',
                'duration': np.random.randint(300, 1800),  # 5-30 minutes
                'view_count': np.random.randint(1000, 100000)
            })
        
        df_sample = pd.DataFrame(sample_data)
        return df_sample


def display_search_results(results, query, max_display=10, keyframes_dir=None):
    """Display search results with images if available"""
    if not results:
        print("No results found")
        return
        
    print(f"\n🔍 Search Results for: '{query}'")
    print("=" * 60)
    
    # Create results table
    results_data = []
    for result in results[:max_display]:
        results_data.append({
            'Rank': getattr(result, 'rank', len(results_data) + 1),
            'Video ID': result.video_id,
            'Frame': result.frame_idx,
            'Score': f"{result.score:.4f}",
            'Type': result.metadata.get('search_type', 'hybrid')
        })
    
    display(pd.DataFrame(results_data))
    
    # Show sample images if available
    if keyframes_dir and Path(keyframes_dir).exists():
        print(f"\n📸 Sample Images:")
        images_shown = 0
        
        for result in results[:3]:
            # Try common frame naming patterns
            possible_paths = [
                Path(keyframes_dir) / f"{result.video_id}" / f"frame_{result.frame_idx:06d}.jpg",
                Path(keyframes_dir) / f"{result.video_id}_frame_{result.frame_idx:06d}.jpg",
                Path(keyframes_dir) / f"{result.video_id}" / f"{result.frame_idx:06d}.jpg",
            ]
            
            for img_path in possible_paths:
                if img_path.exists():
                    try:
                        rank = getattr(result, 'rank', images_shown + 1)
                        display(HTML(f"<h4>#{rank}: {result.video_id} - Frame {result.frame_idx} (Score: {result.score:.3f})</h4>"))
                        display(IPImage(filename=str(img_path), width=300))
                        images_shown += 1
                        break
                    except:
                        continue
        
        if images_shown == 0:
            print("⚠️ No frame images found for display")


def export_search_results(results, query, output_format="csv", output_dir="./output"):
    """Export search results to file"""
    if not results:
        print("No results to export")
        return None
    
    # Create export dataframe
    export_data = []
    for i, result in enumerate(results):
        export_data.append({
            'rank': getattr(result, 'rank', i + 1),
            'video_id': result.video_id,
            'frame_idx': result.frame_idx,
            'score': result.score,
            'search_type': result.metadata.get('search_type', 'hybrid'),
            'query': query,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    export_df = pd.DataFrame(export_data)
    
    # Generate filename
    safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in query)
    safe_query = safe_query.replace(' ', '_')[:30]
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if output_format.lower() == 'csv':
        filename = output_dir / f"search_results_{safe_query}_{timestamp}.csv"
        export_df.to_csv(filename, index=False)
    elif output_format.lower() == 'json':
        filename = output_dir / f"search_results_{safe_query}_{timestamp}.json"
        export_df.to_json(filename, orient='records', indent=2)
    else:
        filename = output_dir / f"search_results_{safe_query}_{timestamp}.parquet"
        export_df.to_parquet(filename, index=False)
    
    print(f"✅ Exported {len(results)} results to {filename}")
    return filename


def evaluate_search_performance(search_function, test_queries=None):
    """Evaluate search performance across different queries"""
    
    if test_queries is None:
        test_queries = [
            "news anchor speaking",
            "television broadcast",
            "daily news program", 
            "studio presentation",
            "reporter interview"
        ]
    
    search_modes = ['vector', 'text', 'hybrid']
    results = []
    
    for mode in search_modes:
        for query in test_queries:
            start_time = time.time()
            try:
                search_results = search_function(query, mode=mode, k=20)
                search_time = time.time() - start_time
                
                # Calculate metrics
                if search_results:
                    unique_videos = len(set(r.video_id for r in search_results))
                    avg_score = np.mean([r.score for r in search_results])
                    diversity = unique_videos / len(search_results)
                else:
                    unique_videos = 0
                    avg_score = 0
                    diversity = 0
                
                results.append({
                    'mode': mode,
                    'query': query,
                    'num_results': len(search_results) if search_results else 0,
                    'unique_videos': unique_videos,
                    'avg_score': avg_score,
                    'search_time_ms': search_time * 1000,
                    'diversity': diversity
                })
            except Exception as e:
                print(f"Error testing {mode} search for '{query}': {e}")
                continue
    
    return pd.DataFrame(results)


def plot_performance_comparison(eval_results):
    """Plot performance comparison charts"""
    
    if len(eval_results) == 0:
        print("No evaluation results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Search time by mode
    sns.boxplot(data=eval_results, x='mode', y='search_time_ms', ax=axes[0,0])
    axes[0,0].set_title('Search Time by Mode')
    axes[0,0].set_ylabel('Time (ms)')
    
    # Average score by mode
    sns.boxplot(data=eval_results, x='mode', y='avg_score', ax=axes[0,1])
    axes[0,1].set_title('Average Score by Mode')
    axes[0,1].set_ylabel('Score')
    
    # Diversity by mode
    sns.boxplot(data=eval_results, x='mode', y='diversity', ax=axes[1,0])
    axes[1,0].set_title('Result Diversity by Mode')
    axes[1,0].set_ylabel('Diversity')
    
    # Results count by mode
    sns.boxplot(data=eval_results, x='mode', y='num_results', ax=axes[1,1])
    axes[1,1].set_title('Results Count by Mode')
    axes[1,1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    summary = eval_results.groupby('mode')[['search_time_ms', 'avg_score', 'diversity']].mean()
    print("\n📊 Performance Summary by Mode:")
    display(summary.round(3))


def create_training_data_sample(metadata_df, num_examples=20):
    """Create sample training data from metadata"""
    
    training_data = []
    
    # Sample queries based on common video content
    sample_queries = [
        "news anchor speaking",
        "television broadcast", 
        "daily news program",
        "reporter presentation",
        "studio setting",
        "professional broadcast",
        "media interview",
        "news program",
        "anchor desk",
        "broadcasting studio"
    ]
    
    for i in range(num_examples):
        # Random query
        query = np.random.choice(sample_queries)
        
        # Random positive examples (frames from videos)
        num_positives = np.random.randint(1, 4)
        positive_samples = metadata_df.sample(num_positives)
        
        positives = []
        for _, row in positive_samples.iterrows():
            positives.append({
                'video_id': row['video_id'],
                'frame_idx': row['frame_idx']
            })
        
        training_data.append({
            'query': query,
            'positives': positives
        })
    
    return training_data


def save_artifacts_summary(artifacts_dir):
    """Create and save summary of generated artifacts"""
    
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        print("Artifacts directory not found")
        return
    
    summary = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'artifacts': {}
    }
    
    # Check for common artifact files
    artifact_files = {
        'vector_index.faiss': 'FAISS vector similarity index',
        'index_metadata.parquet': 'Frame metadata with video IDs and timestamps',
        'bm25_index.pkl': 'BM25 text search index',
        'text_corpus.jsonl': 'Text corpus for search',
        'reranker.joblib': 'Trained reranking model',
        'embeddings.npy': 'CLIP embeddings array',
        'config.json': 'Pipeline configuration'
    }
    
    for filename, description in artifact_files.items():
        filepath = artifacts_dir / filename
        if filepath.exists():
            file_size = filepath.stat().st_size
            summary['artifacts'][filename] = {
                'description': description,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'exists': True
            }
        else:
            summary['artifacts'][filename] = {
                'description': description,
                'exists': False
            }
    
    # Save summary
    summary_file = artifacts_dir / 'artifacts_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Artifacts Summary:")
    for filename, info in summary['artifacts'].items():
        status = "✅" if info['exists'] else "❌"
        size_info = f" ({info['size_mb']}MB)" if info['exists'] else ""
        print(f"  {status} {filename}{size_info}")
        print(f"      {info['description']}")
    
    return summary
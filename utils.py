import os, io, faiss, numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_image(path: Path):
    with Image.open(path) as im:
        return im.convert("RGB")

def save_faiss(index, path: Path):
    faiss.write_index(index, str(path))

def load_faiss(path: Path):
    return faiss.read_index(str(path))

def as_type(x: np.ndarray, dtype: str):
    return x.astype(np.float16) if dtype == "float16" else x.astype(np.float32)

def normalize_rows(x: np.ndarray):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def build_hnsw_index(vecs: np.ndarray, m=32, efC=200):
    d = vecs.shape[1]
    index = faiss.IndexHNSWFlat(d, m)
    index.hnsw.efConstruction = efC
    index.add(vecs)
    return index

def build_flat_index(vecs: np.ndarray):
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    return index

def to_parquet(df: pd.DataFrame, path: Path):
    df.to_parquet(path, index=False)

def from_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

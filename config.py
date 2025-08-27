from pathlib import Path

# Where artifacts (faiss index and mapping) will be saved
ARTIFACT_DIR = Path("./artifacts")

# Model (OpenCLIP) - T4 optimized: SigLIP2 L/16-256 (82.5% ImageNet, multilingual)
MODEL_NAME = "ViT-L-16-SigLIP-256"
MODEL_PRETRAINED = "webli"

# Default CLIP model (smaller fallback for compatibility)
DEFAULT_CLIP_MODEL = "ViT-B-32-quickgelu"
DEFAULT_CLIP_PRETRAINED = "openai"

# Embedding dtype for storage
EMB_DTYPE = "float16"  # 'float32' if you want exact

# Image size used by the OpenCLIP preprocess for the chosen model will be handled automatically


# --- Advanced encoders (switchable) ---
# Valid values (tested): 
#   - 'EVA02-L-14' (OpenCLIP) 
#   - 'ViT-L-14' or 'ViT-L-14-336' (OpenAI/LAION weights via open_clip)
#   - 'siglip-so400m-patch14-384' (SigLIP; requires timm/open_clip build with checkpoint)
# Choose one that fits your VRAM (4070 Ti OK for L/14 @ 224; try 336 with grad off).
ADVANCED_MODELS = [
    "EVA02-L-14",
    "ViT-L-14-336",
    "ViT-L-14",
]
# Default advanced pick; can be overridden via CLI
ADV_MODEL_DEFAULT = "EVA02-L-14"

# --- Experimental presets ---
# Note: These are convenience presets for --experimental mode.
# They assume the checkpoint is available in your environment.
# You can override with --exp-model and --exp-pretrained at runtime.
EXPERIMENTAL_PRESETS = {
    # High-quality English retrieval with classic CLIP recipe
    "bigg": ("ViT-bigG-14", "laion2b_s39b_b160k"),
    # Strong large CLIP; good trade-off vs bigG
    "h14": ("ViT-H-14", "laion2b_s32b_b79k"),
    # EVA02 (open_clip compatible) – lighter than 18B giants
    "eva02l14": ("EVA02-L-14", "laion2b_s32b_b82k"),
    # SigLIP family (requires compatible open_clip build/checkpoints)
    "siglip-so400m-14-384": ("siglip-so400m-patch14-384", "webli"),
    "siglip-l16-384": ("siglip-L-16-384", "webli"),
    # SigLIP 2 variants (available now!)
    "siglip2-l16-256": ("ViT-L-16-SigLIP-256", "webli"),
    "siglip2-so400m": ("ViT-SO400M-14-SigLIP", "webli"),
    "siglip2-gopt": ("ViT-gopt-16-SigLIP2-384", "webli"),
    # EVA-CLIP for maximum quality (if VRAM allows)
    "eva-clip-18b": ("EVA-CLIP-18B", "merged2b_s34b_b88k"),
}

# Fallback order used when --experimental is set without explicit --exp-model
EXPERIMENTAL_FALLBACK_ORDER = [
    "bigg",
    "h14",
    "eva02l14",
]

# --- Video semantics (optional) ---
# Enable building and using segment-level video embeddings aggregated from consecutive frames.
VIDEO_SEMANTICS_ENABLED = False  # default off; enable via CLI flags
VIDEO_SEGMENT_WINDOW = 8         # frames per segment
VIDEO_SEGMENT_STRIDE = 4         # hop (frames)
VIDEO_POOLING = "mean"           # 'mean'|'max'|'meanmax'|'ema'
VIDEO_INDEX_NAME = "video_index.faiss"
VIDEO_MAPPING_NAME = "video_mapping.parquet"

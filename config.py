from pathlib import Path

# Where artifacts (faiss index and mapping) will be saved
ARTIFACT_DIR = Path("./artifacts")

# Model (OpenCLIP)
MODEL_NAME = "ViT-L-14"      # good trade-off; try 'ViT-L-14' if you have VRAM
MODEL_PRETRAINED = "openai"  # 'laion2b_s32b_b82k' also works

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


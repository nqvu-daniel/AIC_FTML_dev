# 📦 AIC FTML Dependencies Guide

This guide resolves NumPy conflicts and ensures smooth installation in Google Colab and other environments.

## 🚨 The NumPy Conflict Problem

Different packages require conflicting NumPy versions:
- **Older packages** (like some OpenCV versions): require `numpy<2.0`
- **Newer packages** (like latest OpenCV): require `numpy>=2.0`

This creates the error: `"some need new numpy some need old wtf"`

## ✅ Our Solution: Smart Installation Strategy

### For Google Colab

Use the enhanced installation cell in `notebooks/AIC_FTML_Colab_AllInOne_Fixed.ipynb`:

```python
# Step 1: Clean conflicting packages
conflicting_packages = [
    "numpy", "opencv-python", "opencv-contrib-python", 
    "thinc", "spacy", "scikit-learn", "scipy"
]
for pkg in conflicting_packages:
    !pip uninstall -y -q {pkg} || true

# Step 2: Install compatible NumPy first
!pip install -q "numpy>=1.24.0,<2.0.0"

# Step 3: Install packages in dependency order
# (See full installation cell for details)
```

### For Local Development

```bash
# Option 1: Use conda (recommended)
conda create -n aic_ftml python=3.9
conda activate aic_ftml
conda install -c conda-forge opencv numpy scipy scikit-learn
pip install -r requirements.txt

# Option 2: Fresh pip environment
python -m venv aic_ftml
source aic_ftml/bin/activate  # or `aic_ftml\Scripts\activate` on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## 🎯 Academic Competition Optimizations

### GPU FAISS Installation
```python
# Auto-detects GPU and installs appropriate version
if torch.cuda.is_available():
    !pip install faiss-gpu  # For cloud/GPU training
else:
    !pip install faiss-cpu  # For local development
```

### TransNet-V2 Academic Requirements
```bash
pip install transnetv2-pytorch>=1.0.5  # 100-frame context for academic excellence
pip install ffmpeg-python              # Required for video processing
```

## 🧪 Testing Your Installation

Run the test script to verify everything works:
```bash
python test_installation.py
```

Expected output:
```
🧪 AIC FTML Installation Test
==================================================
🔧 Testing Critical Dependencies:
✅ NumPy
✅ PyTorch - Version: 2.1+ (CUDA available)
✅ OpenCV - Version: 4.8+
✅ TransNet-V2
✅ FAISS - Basic functionality OK
✅ OpenCLIP
...
🎉 Installation test PASSED!
```

## 🔧 Troubleshooting

### "opencv-contrib-python requires numpy>=2; but you have numpy 1.26.4"
**Solution**: Use our staged installation approach that installs compatible NumPy first.

### "TransNet-V2 import failed" 
**Solution**: 
```bash
pip install transnetv2-pytorch>=1.0.5
pip install ffmpeg-python
```

### "FAISS GPU not working"
**Solution**: The notebook auto-detects and installs the correct version.

### "Package conflicts during pip install"
**Solution**: Use `--no-deps` first, then reinstall with deps:
```bash
pip install --no-deps package_name
pip install package_name
```

## 🏆 Academic Excellence Features Enabled

With correct installation, you get:
- ✅ **TransNet-V2**: 100-frame context shot boundary detection
- ✅ **GPU FAISS**: Accelerated vector search for large datasets  
- ✅ **OpenCLIP**: State-of-the-art multilingual embeddings
- ✅ **FastSAM**: 50x faster than SAM2 for segmentation
- ✅ **BLIP-2**: Academic-grade image captioning
- ✅ **EasyOCR**: Multilingual text detection
- ✅ **Smart Dependencies**: No more version conflicts

## 📝 Notes for Academic Competition

1. **L21/L22 Dataset**: Optimized for AIC 2025 competition requirements
2. **CSV Filtering**: Smart dataset downloading with essential metadata
3. **Colab Ready**: One-click deployment in Google Colab
4. **Scalable**: Handles full dataset when `TEST_MODE=False`

Your academic pipeline is now ready for competition-grade processing! 🎉
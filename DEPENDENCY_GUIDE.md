# Dependency Installation Guide

This project now has **conservative dependency handling** that won't mess up your Colab or cloud environments.

## 🔧 How It Works Now

### **Default Behavior (Safe)**
- Scripts check for dependencies but **DO NOT auto-install**
- If dependencies are missing, scripts will:
  1. Show clear error message
  2. Suggest installation commands
  3. Exit gracefully

### **Auto-Install Mode (Optional)**
- Set environment variable: `AIC_FORCE_INSTALL=1`
- Only then will scripts auto-install missing dependencies
- Only installs what's actually missing

## 📦 Installation Options

### **Option 1: Manual (Recommended for Colab/Cloud)**
```bash
pip install -r requirements.txt
```

### **Option 2: Conda Environment**
```bash
conda env create -f environment.yml
conda activate aic-ftml
```

### **Option 3: Auto-Install Mode**
```bash
export AIC_FORCE_INSTALL=1
python scripts/smart_pipeline.py --help
```

### **Option 4: Use Setup Script**
```bash
python setup_dependencies.py --gpu --test
```

## 🚀 For Colab Users

The colab_pipeline.ipynb already has proper dependency installation built-in. You don't need to do anything extra.

## ☁️ For Cloud Environments

1. **First run**: Install dependencies manually
   ```bash
   pip install -r requirements.txt
   ```

2. **Subsequent runs**: Scripts will work normally without trying to install anything

## 🔍 What Changed

- **Before**: Scripts aggressively auto-installed dependencies (causing conflicts)
- **After**: Scripts only warn about missing dependencies and provide guidance
- **Auto-install**: Only happens if you explicitly set `AIC_FORCE_INSTALL=1`

## 🛠️ Troubleshooting

### Missing Dependencies Error
```
Missing dependency: open_clip_torch
Please install dependencies with: pip install -r requirements.txt
```

**Solution**: Run `pip install -r requirements.txt`

### Want Auto-Install
```bash
export AIC_FORCE_INSTALL=1
python scripts/smart_pipeline.py
```

### Colab Issues
The colab notebook handles dependencies automatically. If issues persist:
1. Restart runtime
2. Clear outputs
3. Run setup cells again

## 🎯 Benefits

- ✅ **No more environment conflicts**
- ✅ **Works in Colab, cloud, and local**
- ✅ **Clear error messages**
- ✅ **Optional auto-install when needed**
- ✅ **Respects existing installations**

The smart pipeline and other scripts now work reliably across all environments!
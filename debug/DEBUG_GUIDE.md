# 🐛 Bayesian LoRA Debug Guide

**The Complete Guide to Debugging Bayesian LoRA Projects**

---

## 🚀 Quick Start

### **Run Debug Suite:**
```bash
# Quick check (recommended for daily use)
python3 debug/debug_suite.py --quick

# Full comprehensive check (recommended before deployments)
python3 debug/debug_suite.py

# Verbose output for detailed troubleshooting
python3 debug/debug_suite.py --verbose
```

---

## 📋 What the Debug Suite Checks

The debug suite performs **30 comprehensive checks** across 6 categories:

### **1. Environment Checks (4 checks)**
- ✅ Python version compatibility (3.9+)
- ✅ CUDA availability and version
- ✅ Working directory verification
- ✅ Virtual environment status

### **2. Package Installation (3 checks)**
- ✅ Package importability
- ✅ Pip installation status
- ✅ Editable install verification

### **3. File Structure (13 checks)**
- ✅ All critical directories exist
- ✅ All `__init__.py` files present
- ✅ Configuration files available
- ✅ Source code structure intact

### **4. Module Imports (5 checks)**
- ✅ Basic package import
- ✅ Data module imports
- ✅ Models module imports
- ✅ Samplers module imports
- ✅ Utils module imports

### **5. Model Creation (2 checks)**
- ✅ HuggingFace LoRA model creation
- ✅ LoRA parameter counting and verification

### **6. Data Loading (3 checks)**
- ✅ Dataset metadata loading
- ✅ Dataloader creation
- ✅ Sample data processing

---

## 🔧 Common Issues & Solutions

### **Issue 1: "No module named 'bayesian_lora'"**
```bash
# Solution: Install package in editable mode
pip3 install -e .

# If that fails, try:
pip3 install -e . --force-reinstall
```

### **Issue 2: "No module named 'bayesian_lora.data'"**
```bash
# Check if __init__.py file exists and has content
ls -la src/bayesian_lora/data/__init__.py
cat src/bayesian_lora/data/__init__.py

# If empty/missing, check git tracking
git status src/bayesian_lora/data/
```

### **Issue 3: LoRA Parameters = 0**
```bash
# Check model configuration
python3 -c "
from bayesian_lora.models.hf_lora import build_huggingface_lora_model
model = build_huggingface_lora_model({
    'name': 'bert-base-uncased',
    'num_labels': 2,
    'lora': {'r': 16, 'alpha': 32, 'dropout': 0.1}
})
from bayesian_lora.utils.lora_params import count_lora_parameters
print(f'LoRA parameters: {count_lora_parameters(model):,}')
"
```

### **Issue 4: CUDA Not Available**
```bash
# Check PyTorch CUDA support
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check CUDA version compatibility
nvidia-smi
python3 -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
```

### **Issue 5: Package Not Found in Pip**
```bash
# Check installation status
pip3 show bayesian-lora

# Check if in editable mode
pip3 show bayesian-lora | grep "Editable project location"

# Reinstall if needed
pip3 install -e . --force-reinstall
```

---

## 🎯 Usage Scenarios

### **Local Development:**
```bash
# Before committing code
python3 debug/debug_suite.py --quick

# Before pushing to remote
python3 debug/debug_suite.py

# When troubleshooting issues
python3 debug/debug_suite.py --verbose
```

### **Cluster Deployment:**
```bash
# Quick verification after setup
python3 debug/debug_suite.py --quick

# Full diagnostics if issues arise
python3 debug/debug_suite.py

# Integration with SLURM scripts
python3 debug/debug_suite.py --quick || exit 1
```

### **CI/CD Integration:**
```bash
# Exit code 0 = all checks passed
# Exit code 1 = some checks failed
python3 debug/debug_suite.py
```

---

## 📊 Understanding Debug Output

### **Sample Output:**
```
============================================================
🔍 BAYESIAN LORA DEBUG REPORT
============================================================

📋 ENVIRONMENT CHECKS:
----------------------------------------
✅ PASS python_version
❌ FAIL cuda_available
✅ PASS correct_directory
❌ FAIL venv_active

📋 PACKAGE CHECKS:
----------------------------------------
✅ PASS package_imported
✅ PASS pip_installed
❌ FAIL editable_install

📊 SUMMARY: 27/30 checks passed
⚠️  Some checks failed. Check the details above.
============================================================

🔧 SUGGESTED FIXES:
  🔧 Activate virtual environment: source .venv/bin/activate
```

### **What Each Section Means:**
- **✅ PASS**: This check succeeded
- **❌ FAIL**: This check failed - needs attention
- **📊 SUMMARY**: Overall status (X/Y checks passed)
- **🔧 SUGGESTED FIXES**: Actionable solutions for failed checks

---

## 🚨 Troubleshooting Guide

### **If Debug Suite Fails to Import:**
```bash
# Check if you're in the right directory
pwd
ls -la debug/debug_suite.py

# Check Python path
python3 -c "import sys; print(sys.path)"

# Check Python version
python3 --version
```

### **If Some Checks Fail:**
1. **Read the error messages** - they contain specific details
2. **Check the suggested fixes** - the suite provides actionable advice
3. **Run individual checks** to isolate issues
4. **Check the documentation** for common solutions

### **If All Checks Pass:**
🎉 **Your environment is ready!** You can proceed with:
- Running experiments
- Training models
- Deploying to clusters
- Contributing code

---

## 🏗️ Project Structure

### **Critical Files (Checked by Debug Suite):**
```
Bayesian-LORA/
├── src/
│   └── bayesian_lora/
│       ├── __init__.py          # Package initialization
│       ├── data/
│       │   ├── __init__.py      # Data module
│       │   ├── glue_datasets.py # GLUE datasets
│       │   └── cifar.py         # CIFAR datasets
│       ├── models/
│       │   ├── __init__.py      # Models module
│       │   └── hf_lora.py       # HuggingFace LoRA models
│       ├── samplers/
│       │   ├── __init__.py      # Samplers module
│       │   └── sgld.py          # SGLD sampler
│       └── utils/
│           ├── __init__.py      # Utils module
│           └── lora_params.py   # LoRA parameter utilities
├── configs/                      # Configuration files
├── scripts/                      # Training scripts
├── pyproject.toml               # Project configuration
└── setup.py                     # Package setup
```

---

## 🔍 Advanced Debugging

### **Custom Debug Checks:**
```python
# Add custom checks to debug_suite.py
def check_custom_functionality(self) -> Dict[str, bool]:
    """Check custom functionality."""
    results = {}
    
    try:
        # Your custom check here
        results['custom_check'] = True
        self.log("✅ Custom functionality working")
    except Exception as e:
        results['custom_check'] = False
        self.log(f"❌ Custom functionality failed: {e}")
    
    return results
```

### **Integration with Other Tools:**
```bash
# Run debug suite and capture output
python3 debug/debug_suite.py > debug_report.txt 2>&1

# Check exit code
echo $?

# Parse results programmatically
python3 debug/debug_suite.py | grep "SUMMARY"
```

---

## 📚 Additional Resources

### **Project Documentation:**
- **Main README**: `../README.md`
- **Requirements**: `../requirements_lora.txt`
- **Configuration**: `../configs/`
- **Training Scripts**: `../scripts/`

### **External Resources:**
- **PyTorch Documentation**: https://pytorch.org/docs/
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/
- **PEFT Documentation**: https://huggingface.co/docs/peft/

---

## 🤝 Contributing

### **Adding New Debug Checks:**
1. **Add to `debug_suite.py`** rather than creating new scripts
2. **Update this guide** with new features
3. **Test locally** before committing
4. **Keep it simple** - one comprehensive tool is better than many scattered ones

### **Reporting Issues:**
1. **Run debug suite first** to gather diagnostic information
2. **Include debug output** in your issue report
3. **Describe the problem** clearly and concisely
4. **Provide steps to reproduce** the issue

---

## 🎯 Best Practices

### **Daily Workflow:**
1. **Before starting work**: `python3 debug/debug_suite.py --quick`
2. **Before committing**: `python3 debug/debug_suite.py --quick`
3. **Before pushing**: `python3 debug/debug_suite.py`
4. **When troubleshooting**: `python3 debug/debug_suite.py --verbose`

### **Cluster Workflow:**
1. **After setup**: `python3 debug/debug_suite.py --quick`
2. **Before experiments**: `python3 debug/debug_suite.py --quick`
3. **If issues arise**: `python3 debug/debug_suite.py`

### **CI/CD Workflow:**
1. **Integration**: Run debug suite as part of build process
2. **Deployment**: Run debug suite before deploying
3. **Monitoring**: Run debug suite periodically to catch issues

---

## 🏆 Final Notes

### **Why This Approach Works:**
- **🎯 Single Source of Truth**: One script instead of 20+
- **📊 Systematic Coverage**: No gaps in verification
- **🔧 Actionable Results**: Clear fixes for common issues
- **📚 Comprehensive Documentation**: Everything in one place
- **🚀 Easy Maintenance**: Update one file instead of many

### **Remember:**
- **Always run debug suite** before reporting issues
- **Use `--quick` for daily checks**, full check for deployments
- **Check suggested fixes** before asking for help
- **Keep the debug suite updated** as the project evolves

---

**Status**: **COMPREHENSIVE AND READY** 🚀

**This guide contains everything you need to debug Bayesian LoRA projects effectively!** ✨

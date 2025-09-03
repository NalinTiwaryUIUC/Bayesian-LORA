# 🐛 Bayesian LoRA Comprehensive Debug Guide

**The Complete Guide to Debugging and Testing Bayesian LoRA Projects**

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

The debug suite performs **50+ comprehensive checks** across 11 categories:

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
- ✅ Scripts and runs directories

### **4. Module Imports (7 checks)**
- ✅ Basic package import
- ✅ Data module imports (CIFAR + GLUE)
- ✅ Models module imports (CIFAR + LoRA)
- ✅ Samplers module imports
- ✅ Utils module imports

### **5. CIFAR Models (4 checks)**
- ✅ ResNet-18 CIFAR model creation
- ✅ WideResNet-28-10 CIFAR model creation
- ✅ ResNet forward pass functionality
- ✅ WideResNet forward pass functionality

### **6. LoRA Models (3 checks)**
- ✅ LoRA model creation with RoBERTa
- ✅ LoRA parameter counting and verification
- ✅ LoRA forward pass functionality

### **7. SGLD Samplers (5 checks)**
- ✅ SGLD sampler creation
- ✅ ASGLD sampler creation
- ✅ SAM-SGLD sampler creation
- ✅ SAM-SGLD Rank-1 sampler creation
- ✅ SGLD step functionality

### **8. Data Loading (3 checks)**
- ✅ CIFAR-10 dataset loading
- ✅ GLUE MRPC dataset loading
- ✅ Data sample format validation

### **9. Configuration Files (5 checks)**
- ✅ CIFAR-10 ResNet-18 SGLD config
- ✅ CIFAR-100 WideResNet-28-10 SGLD config
- ✅ CIFAR-100 WideResNet-28-10 SAM-SGLD config
- ✅ CIFAR-100 WideResNet-28-10 SAM-SGLD Rank-1 config
- ✅ MRPC RoBERTa LoRA SGLD config

### **10. Training Scripts (4 checks)**
- ✅ CIFAR training script (`train.py`)
- ✅ CIFAR evaluation script (`eval.py`)
- ✅ LoRA training script (`train_mrpc_lora.py`)
- ✅ LoRA evaluation script (`eval_mrpc_lora.py`)

### **11. Makefile Targets (3 checks)**
- ✅ Makefile existence
- ✅ Make help target functionality
- ✅ Make clean target functionality

---

## 🔧 Common Issues & Solutions

### **Issue 1: "No module named 'bayesian_lora'"**
```bash
# Solution: Install package in editable mode
pip3 install -e .

# If that fails, try:
pip3 install -e . --force-reinstall
```

### **Issue 2: CIFAR Model Creation Fails**
```bash
# Check if torchvision is installed
pip3 install torchvision

# Test model creation manually
python3 -c "
from bayesian_lora.models.resnet_cifar import ResNetCIFAR
model = ResNetCIFAR(depth=18, num_classes=10)
print('✅ ResNet created successfully')
"
```

### **Issue 3: LoRA Model Creation Fails**
```bash
# Check if transformers and peft are installed
pip3 install transformers peft

# Test LoRA model creation manually
python3 -c "
from bayesian_lora.models.hf_lora import LoRAModel
from transformers import AutoModelForSequenceClassification
base_model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
lora_model = LoRAModel(base_model, r=8, alpha=16.0)
print('✅ LoRA model created successfully')
"
```

### **Issue 4: SGLD Sampler Creation Fails**
```bash
# Check if samplers can be imported
python3 -c "
from bayesian_lora.samplers.sgld import SGLDSampler
import torch
test_model = torch.nn.Linear(10, 2)
sampler = SGLDSampler(test_model, temperature=1.0, step_size=1e-4)
print('✅ SGLD sampler created successfully')
"
```

### **Issue 5: Data Loading Fails**
```bash
# For CIFAR data issues
python3 -c "
from bayesian_lora.data.cifar import get_cifar_dataset
train_ds, test_ds = get_cifar_dataset('cifar10', 'data/cifar-10-batches-py')
print(f'✅ CIFAR data loaded: {len(train_ds)} train, {len(test_ds)} test')
"

# For GLUE data issues
python3 -c "
from bayesian_lora.data.glue_datasets import MRPCDataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
dataset = MRPCDataset(split='train', tokenizer=tokenizer, max_length=128)
print(f'✅ MRPC data loaded: {len(dataset)} samples')
"
```

### **Issue 6: Configuration Files Not Found**
```bash
# Check if configs directory exists
ls -la configs/

# Check if specific config files exist
ls -la configs/*.yaml

# Verify YAML syntax
python3 -c "
import yaml
with open('configs/cifar10_resnet18_sgld.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('✅ Config file loaded successfully')
"
```

### **Issue 7: Training Scripts Fail to Import**
```bash
# Check if scripts can be imported
python3 -c "
import scripts.train
import scripts.eval
import scripts.train_mrpc_lora
import scripts.eval_mrpc_lora
print('✅ All training scripts imported successfully')
"
```

### **Issue 8: Makefile Targets Don't Work**
```bash
# Check if Makefile exists
ls -la Makefile

# Test make help
make help

# Test make clean
make clean

# Check if make is available
which make
```

---

## 🎯 Usage Scenarios

### **Local Development:**
```bash
# Before starting work
python3 debug/debug_suite.py --quick

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

### **Experiment Preparation:**
```bash
# Before running CIFAR experiments
python3 debug/debug_suite.py --quick

# Before running LoRA experiments
python3 debug/debug_suite.py --quick

# Before running all experiments
python3 debug/debug_suite.py
```

---

## 📊 Understanding Debug Output

### **Sample Output:**
```
======================================================================
🔍 BAYESIAN LORA COMPREHENSIVE DEBUG REPORT
======================================================================

📋 ENVIRONMENT CHECKS:
--------------------------------------------------
✅ PASS Python Version
✅ PASS Cuda Available
✅ PASS Correct Directory
❌ FAIL Venv Active

📋 CIFAR MODELS CHECKS:
--------------------------------------------------
✅ PASS Resnet Created
✅ PASS Wideresnet Created
✅ PASS Resnet Forward
✅ PASS Wideresnet Forward

📋 LORA MODELS CHECKS:
--------------------------------------------------
✅ PASS Lora Model Created
✅ PASS Lora Parameters
✅ PASS Lora Forward

📋 SGLD SAMPLERS CHECKS:
--------------------------------------------------
✅ PASS Sgld Sampler Created
✅ PASS Asgld Sampler Created
✅ PASS Sam Sgld Sampler Created
✅ PASS Sam Sgld R1 Sampler Created
✅ PASS Sgld Step Works

📊 SUMMARY: 45/50 checks passed
✅ MOST CHECKS PASSED! Minor issues detected but environment is functional.
======================================================================

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
- Running CIFAR experiments with all SGLD variants
- Running LoRA experiments with MRPC dataset
- Using all training and evaluation scripts
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
│       │   ├── hf_lora.py       # HuggingFace LoRA models
│       │   ├── resnet_cifar.py  # CIFAR ResNet models
│       │   └── wide_resnet.py   # CIFAR WideResNet models
│       ├── samplers/
│       │   ├── __init__.py      # Samplers module
│       │   └── sgld.py          # SGLD samplers (all variants)
│       └── utils/
│           ├── __init__.py      # Utils module
│           └── lora_params.py   # LoRA parameter utilities
├── configs/                      # All experiment configurations
├── scripts/                      # All training/evaluation scripts
├── pyproject.toml               # Project configuration
├── setup.py                     # Package setup
├── Makefile                     # Build automation
└── requirements_lora.txt        # Dependencies
```

### **Experiment Types Supported:**
- **CIFAR-10 ResNet-18 SGLD**
- **CIFAR-100 WideResNet-28-10 SGLD**
- **CIFAR-100 WideResNet-28-10 SAM-SGLD**
- **CIFAR-100 WideResNet-28-10 SAM-SGLD Rank-1**
- **MRPC RoBERTa LoRA SGLD**

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

# Use in CI/CD pipelines
python3 debug/debug_suite.py || (echo "Debug checks failed" && exit 1)
```

### **Performance Testing:**
```bash
# Test model creation performance
time python3 -c "
from bayesian_lora.models.resnet_cifar import ResNetCIFAR
model = ResNetCIFAR(depth=18, num_classes=10)
"

# Test data loading performance
time python3 -c "
from bayesian_lora.data.cifar import get_cifar_dataset
train_ds, test_ds = get_cifar_dataset('cifar10', 'data/cifar-10-batches-py')
"
```

---

## 📚 Additional Resources

### **Project Documentation:**
- **Main README**: `../README.md`
- **Requirements**: `../requirements_lora.txt`
- **Configuration**: `../configs/`
- **Training Scripts**: `../scripts/`
- **Makefile**: `../Makefile`

### **External Resources:**
- **PyTorch Documentation**: https://pytorch.org/docs/
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/
- **PEFT Documentation**: https://huggingface.co/docs/peft/
- **TorchVision Documentation**: https://pytorch.org/vision/stable/

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

### **Experiment Workflow:**
1. **Before CIFAR experiments**: `python3 debug/debug_suite.py --quick`
2. **Before LoRA experiments**: `python3 debug/debug_suite.py --quick`
3. **Before all experiments**: `python3 debug/debug_suite.py --quick`

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
- **🧪 Comprehensive Testing**: Tests all experiment types and components

### **Remember:**
- **Always run debug suite** before reporting issues
- **Use `--quick` for daily checks**, full check for deployments
- **Check suggested fixes** before asking for help
- **Keep the debug suite updated** as the project evolves
- **Test all experiment types** before running experiments

---

**Status**: **COMPREHENSIVE AND READY** 🚀

**This guide contains everything you need to debug and test Bayesian LoRA projects effectively!** ✨

**Supports all experiment types: CIFAR (ResNet/WideResNet) + LoRA (RoBERTa) with all SGLD variants!** 🎯

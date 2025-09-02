# Debugging Tools for Bayesian LoRA

This directory contains debugging and troubleshooting tools for the Bayesian LoRA project.

## 🚀 **Quick Start**

When you encounter issues on the cluster, run these in order:

1. **`cluster_troubleshooting.py`** - Comprehensive environment check
2. **`debug_lora.py`** - LoRA-specific debugging
3. **`test_model_direct.py`** - Model creation testing
4. **`comprehensive_test.py`** - Full system validation

## 📁 **File Organization**

### **Primary Debugging Tools**
- **`cluster_troubleshooting.py`** - Main troubleshooting script (run first!)
- **`debug_lora.py`** - LoRA parameter creation debugging
- **`test_model_direct.py`** - Isolated model testing

### **Validation Tools**
- **`comprehensive_test.py`** - Complete system validation
- **`test_import.py`** - Basic import testing

## 🔧 **Usage Guide**

### **1. Environment Check (Always Run First)**
```bash
python3 debug/cluster_troubleshooting.py
```
- Checks Python version, CUDA, dependencies
- Verifies all imports work
- Tests basic functionality

### **2. LoRA Debugging**
```bash
python3 debug/debug_lora.py
```
- Investigates LoRA parameter creation
- Tests different target module configurations
- Useful when LoRA parameters are 0

### **3. Model Testing**
```bash
python3 debug/test_model_direct.py
```
- Tests model creation step-by-step
- Checks LoRA parameter counts
- Isolates model-related issues

### **4. Full Validation**
```bash
python3 debug/comprehensive_test.py
```
- Runs complete system test
- Verifies all components work together
- Final check before running experiments

## 🚨 **Common Issues & Solutions**

### **Import Errors**
```bash
# Solution 1: Install package
pip3 install -e .

# Solution 2: Add src to PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:$(pwd)/src
```

### **LoRA Parameters = 0**
```bash
# Run debugging tools
python3 debug/debug_lora.py
python3 debug/test_model_direct.py
```

### **CUDA Issues**
```bash
# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

## 📋 **Debugging Workflow**

1. **Identify Issue**: Run `cluster_troubleshooting.py`
2. **Specific Debug**: Use targeted tools based on error
3. **Fix Issue**: Apply solution
4. **Verify Fix**: Run `comprehensive_test.py`
5. **Submit Job**: Use `sbatch experiment.sbatch`

## 🎯 **When to Use Each Tool**

- **`cluster_troubleshooting.py`**: Any issue, first step
- **`debug_lora.py`**: LoRA parameter problems
- **`test_model_direct.py`**: Model creation failures
- **`comprehensive_test.py`**: After fixes, before experiments
- **`test_import.py`**: Basic import verification

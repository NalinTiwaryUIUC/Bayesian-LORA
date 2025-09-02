# 🚨 Cluster Import Issue Fix Guide

## **Problem Identified**
The cluster is having issues with `pip install -e .` due to:
- Deprecated `setup.py develop` command
- Pip version conflicts
- Virtual environment issues

## **✅ Solution Applied**

### **1. Batch Script Updated**
- **Removed** problematic `pip install -e .`
- **Added** direct dependency installation
- **Set** `PYTHONPATH` explicitly
- **Added** pip upgrade step

### **2. __init__.py Fixed**
- **Replaced** direct imports with lazy imports
- **Added** error handling for missing modules
- **Prevented** circular import issues

### **2. New Installation Flow**
```bash
# Old (problematic):
pip3 install -e .

# New (working):
pip3 install --upgrade pip
pip3 install -r requirements_lora.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### **3. New Import Test**
```bash
# Simple import test (recommended first):
python3 debug/simple_import_test.py

# Full import test (if simple test passes):
python3 debug/test_import.py
```

## **🔧 Manual Fix on Cluster**

If the batch script still has issues, run these commands manually:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Upgrade pip
python3 -m pip install --upgrade pip

# 3. Install dependencies
pip3 install -r requirements_lora.txt

# 4. Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 5. Test import
python3 debug/test_import.py
```

## **🚨 Alternative Solutions**

### **If PYTHONPATH doesn't work:**
```bash
# Option 1: Add to .bashrc
echo 'export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"' >> ~/.bashrc
source ~/.bashrc

# Option 2: Use sys.path in Python
python3 -c "
import sys
sys.path.insert(0, '$(pwd)/src')
import bayesian_lora
print('✅ Import successful!')
"
```

### **If virtual environment is corrupted:**
```bash
# Remove and recreate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements_lora.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## **✅ Verification Commands**

```bash
# Test basic import
python3 debug/test_import.py

# Test full system
python3 debug/cluster_troubleshooting.py

# Test all tools
python3 debug/run_debug.py
```

## **🎯 Expected Result**
All debug tools should pass with:
- ✅ Environment check: 5/5
- ✅ LoRA debugging: Working
- ✅ Model testing: Working  
- ✅ Full validation: 5/5
- ✅ **Overall: 4/4 tools passed**

## **💡 Pro Tips**
- **Always set PYTHONPATH** before running Python
- **Use `debug/cluster_troubleshooting.py`** first
- **Check pip version** and upgrade if needed
- **Avoid `pip install -e .`** on problematic clusters

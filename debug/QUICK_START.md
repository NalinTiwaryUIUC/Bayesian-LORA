# 🚀 Quick Start - Cluster Debugging

## **Immediate Actions (Run These First)**

### **1. Quick Health Check**
```bash
python3 debug/cluster_troubleshooting.py
```
- **When**: Any issue, first step
- **What**: Checks environment, imports, basic functionality
- **Expected**: All 5 checks should pass

### **2. Full System Test**
```bash
python3 debug/run_debug.py
```
- **When**: After fixes, before experiments
- **What**: Runs all debugging tools in sequence
- **Expected**: All 4 tools should pass

## **🚨 Common Issues & Quick Fixes**

### **Import Errors**
```bash
# Fix 1: Install package
pip3 install -e .

# Fix 2: Add src to PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:$(pwd)/src

# Fix 3: Install requirements
pip3 install -r requirements_lora.txt
```

### **LoRA Parameters = 0**
```bash
# Debug LoRA specifically
python3 debug/debug_lora.py

# Test model creation
python3 debug/test_model_direct.py
```

### **Configuration Issues**
```bash
# Check config file exists
ls -la configs/experiment_sst2_bert_sgld.yaml

# Verify config syntax
python3 -c "import yaml; yaml.safe_load(open('configs/experiment_sst2_bert_sgld.yaml'))"
```

## **📋 Debugging Workflow**

1. **Run**: `python3 debug/cluster_troubleshooting.py`
2. **If issues**: Use specific debug tools
3. **After fixes**: Run `python3 debug/run_debug.py`
4. **Submit job**: `sbatch experiment.sbatch`

## **🎯 Tool Summary**

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `cluster_troubleshooting.py` | Environment check | **Always first** |
| `debug_lora.py` | LoRA debugging | LoRA parameter issues |
| `test_model_direct.py` | Model testing | Model creation failures |
| `comprehensive_test.py` | Full validation | After fixes |
| `run_debug.py` | Run all tools | Final verification |

## **💡 Pro Tips**

- **Always start** with `cluster_troubleshooting.py`
- **Check paths** if config loading fails
- **Verify CUDA** if GPU issues occur
- **Use `run_debug.py`** for comprehensive testing
- **Keep debug output** for troubleshooting

## **🚀 Ready to Submit?**

If `python3 debug/run_debug.py` shows **"All 4 tools passed"**, you're ready to submit your job:

```bash
sbatch experiment.sbatch
```

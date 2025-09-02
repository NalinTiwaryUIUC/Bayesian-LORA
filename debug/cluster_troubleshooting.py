#!/usr/bin/env python3
"""
Comprehensive troubleshooting script for cluster issues.
Run this on the cluster if you encounter problems.
"""

import sys
import os
import traceback

def check_environment():
    """Check cluster environment and dependencies."""
    print("=== Cluster Environment Check ===\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check environment variables
    print(f"\nEnvironment variables:")
    print(f"  - PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"  - TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'Not set')}")
    
    # Check CUDA
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    return True

def check_imports():
    """Check all critical imports."""
    print("\n=== Import Check ===\n")
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("datasets", "Datasets"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("numpy", "NumPy"),
        ("sklearn", "scikit-learn"),
    ]
    
    all_imports_ok = True
    for module_name, display_name in imports_to_test:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✓ {display_name}: {version}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def check_bayesian_lora():
    """Check Bayesian LoRA module."""
    print("\n=== Bayesian LoRA Module Check ===\n")
    
    try:
        import bayesian_lora
        print(f"✓ bayesian_lora module: {bayesian_lora.__version__}")
        
        # Test submodule imports
        submodules = [
            ("bayesian_lora.models.hf_lora", "HF LoRA Models"),
            ("bayesian_lora.data.glue_datasets", "GLUE Datasets"),
            ("bayesian_lora.samplers.sgld", "SGLD Samplers"),
            ("bayesian_lora.utils.lora_params", "LoRA Utils"),
        ]
        
        for module_path, display_name in submodules:
            try:
                __import__(module_path)
                print(f"  ✓ {display_name}")
            except ImportError as e:
                print(f"  ❌ {display_name}: {e}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ bayesian_lora module import failed: {e}")
        return False

def test_model_creation():
    """Test model creation on cluster."""
    print("\n=== Model Creation Test ===\n")
    
    try:
        from bayesian_lora.models.hf_lora import build_huggingface_lora_model
        
        config = {
            'name': 'bert-base-uncased',
            'num_labels': 2,
            'lora': {'rank': 4, 'alpha': 8.0, 'dropout': 0.1}
        }
        
        print("Creating model...")
        model = build_huggingface_lora_model(config)
        print(f"✓ Model created: {type(model).__name__}")
        
        # Check LoRA parameters
        from bayesian_lora.utils.lora_params import get_lora_parameters
        lora_params = get_lora_parameters(model)
        print(f"✓ LoRA parameters: {len(lora_params)} groups")
        
        if lora_params:
            total_lora = sum(p.numel() for p in lora_params)
            print(f"  Total LoRA parameters: {total_lora:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading on cluster."""
    print("\n=== Data Loading Test ===\n")
    
    try:
        from bayesian_lora.models.hf_lora import build_huggingface_lora_model
        from bayesian_lora.data.glue_datasets import create_dataloaders
        
        # Create minimal model for tokenizer
        config = {
            'name': 'bert-base-uncased',
            'num_labels': 2,
            'lora': {'rank': 4, 'alpha': 8.0, 'dropout': 0.1}
        }
        
        model = build_huggingface_lora_model(config)
        tokenizer = model.get_tokenizer()
        
        print("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            dataset_name='sst2',
            tokenizer=tokenizer,
            batch_size=8,  # Small batch for testing
            max_length=32   # Short length for testing
        )
        
        print(f"✓ Dataloaders created:")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        
        # Test one batch
        batch = next(iter(train_loader))
        print(f"✓ Sample batch loaded:")
        print(f"  - input_ids: {batch['input_ids'].shape}")
        print(f"  - attention_mask: {batch['attention_mask'].shape}")
        print(f"  - labels: {batch['labels'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all troubleshooting checks."""
    print("🔧 Cluster Troubleshooting for Bayesian LoRA\n")
    
    checks = [
        ("Environment", check_environment),
        ("Imports", check_imports),
        ("Bayesian LoRA Module", check_bayesian_lora),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
                print(f"✓ {check_name} check passed\n")
            else:
                print(f"❌ {check_name} check failed\n")
        except Exception as e:
            print(f"❌ {check_name} check crashed: {e}\n")
            traceback.print_exc()
    
    print(f"=== Troubleshooting Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All checks passed! Your environment is ready.")
        return True
    else:
        print("⚠️  Some checks failed. Check the output above for issues.")
        print("\nCommon solutions:")
        print("1. Activate virtual environment: source .venv/bin/activate")
        print("2. Install package: pip3 install -e .")
        print("3. Install requirements: pip3 install -r requirements_lora.txt")
        print("4. Check PYTHONPATH: export PYTHONPATH=${PYTHONPATH}:$(pwd)/src")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

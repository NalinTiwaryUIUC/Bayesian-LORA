#!/usr/bin/env python3
"""
Comprehensive test script to check for potential errors in the Bayesian LoRA codebase.
"""

import sys
import os
import traceback

def test_imports():
    """Test all module imports."""
    print("=== Testing Module Imports ===")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import peft
        print(f"✓ PEFT: {peft.__version__}")
    except ImportError as e:
        print(f"❌ PEFT import failed: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets import failed: {e}")
        return False
    
    return True

def test_bayesian_lora_imports():
    """Test Bayesian LoRA module imports."""
    print("\n=== Testing Bayesian LoRA Module ===")
    
    try:
        import bayesian_lora
        print(f"✓ bayesian_lora module: {bayesian_lora.__version__}")
    except ImportError as e:
        print(f"❌ bayesian_lora module import failed: {e}")
        return False
    
    try:
        from bayesian_lora.models.hf_lora import build_huggingface_lora_model
        print("✓ hf_lora module imported")
    except ImportError as e:
        print(f"❌ hf_lora import failed: {e}")
        return False
    
    try:
        from bayesian_lora.data.glue_datasets import create_dataloaders, get_dataset_metadata
        print("✓ glue_datasets module imported")
    except ImportError as e:
        print(f"❌ glue_datasets import failed: {e}")
        return False
    
    try:
        from bayesian_lora.samplers.sgld import sgld_step, asgld_step, sam_sgld_step
        print("✓ sgld module imported")
    except ImportError as e:
        print(f"❌ sgld import failed: {e}")
        return False
    
    try:
        from bayesian_lora.utils.lora_params import get_lora_parameters, count_lora_parameters
        print("✓ lora_params module imported")
    except ImportError as e:
        print(f"❌ lora_params import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation with minimal config."""
    print("\n=== Testing Model Creation ===")
    
    try:
        from bayesian_lora.models.hf_lora import build_huggingface_lora_model
        from bayesian_lora.utils.lora_params import get_lora_parameters
        
        # Minimal config for testing
        config = {
            'name': 'bert-base-uncased',
            'num_labels': 2,
            'lora': {
                'rank': 4,
                'alpha': 8.0,
                'dropout': 0.1
            }
        }
        
        model = build_huggingface_lora_model(config)
        print("✓ Model creation successful")
        print(f"  - Model type: {type(model).__name__}")
        
        # Check if LoRA parameters were created
        lora_params = get_lora_parameters(model)
        print(f"  - LoRA parameters found: {len(lora_params)}")
        if lora_params:
            total_lora = sum(p.numel() for p in lora_params)
            print(f"  - Total LoRA parameters: {total_lora:,}")
        else:
            print("  - ⚠️  No LoRA parameters found - this might indicate an issue")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration file loading."""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        import yaml
        
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "experiment_sst2_bert_sgld.yaml")
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Configuration loaded successfully")
        print(f"  - Dataset: {config['data']['name']}")
        print(f"  - Model: {config['model']['name']}")
        print(f"  - Sampler: {config['sampler']['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        traceback.print_exc()
        return False

def test_sampler_functions():
    """Test sampler function calls."""
    print("\n=== Testing Sampler Functions ===")
    
    try:
        import torch
        from bayesian_lora.samplers.sgld import sgld_step
        
        # Create dummy parameters and gradients
        x = torch.randn(100, requires_grad=True)
        grad = torch.randn_like(x)
        
        # Test SGLD step
        x_new = sgld_step(x, grad, eps=1e-4, tau=1.0)
        print("✓ SGLD step function works")
        
        return True
        
    except Exception as e:
        print(f"❌ Sampler test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Comprehensive Bayesian LoRA Tests\n")
    
    tests = [
        ("Basic Imports", test_imports),
        ("Bayesian LoRA Imports", test_bayesian_lora_imports),
        ("Configuration Loading", test_config_loading),
        ("Model Creation", test_model_creation),
        ("Sampler Functions", test_sampler_functions),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            traceback.print_exc()
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your codebase is ready to run.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix the issues before running experiments.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

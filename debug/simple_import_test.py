#!/usr/bin/env python3
"""
Simple import test for cluster debugging.
This script tests basic imports without complex dependencies.
"""

import sys
import os

def test_basic_imports():
    """Test basic Python imports."""
    print("=== Testing Basic Python Imports ===")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        import peft
        print(f"✅ PEFT: {peft.__version__}")
    except ImportError as e:
        print(f"❌ PEFT: {e}")
        return False
    
    return True

def test_bayesian_lora_import():
    """Test bayesian_lora module import."""
    print("\n=== Testing Bayesian LoRA Import ===")
    
    # Check if src directory exists
    src_path = os.path.join(os.getcwd(), 'src')
    if not os.path.exists(src_path):
        print(f"❌ src directory not found at: {src_path}")
        return False
    
    # Check if bayesian_lora directory exists
    lora_path = os.path.join(src_path, 'bayesian_lora')
    if not os.path.exists(lora_path):
        print(f"❌ bayesian_lora directory not found at: {lora_path}")
        return False
    
    print(f"✅ Project structure found:")
    print(f"   src: {src_path}")
    print(f"   bayesian_lora: {lora_path}")
    
    # List contents
    print(f"\n📁 Contents of src/:")
    try:
        for item in os.listdir(src_path):
            item_path = os.path.join(src_path, item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}/")
                try:
                    subitems = os.listdir(item_path)
                    for subitem in subitems[:5]:  # Show first 5 items
                        print(f"     - {subitem}")
                    if len(subitems) > 5:
                        print(f"     ... and {len(subitems) - 5} more")
                except:
                    print(f"     (cannot list contents)")
            else:
                print(f"   📄 {item}")
    except Exception as e:
        print(f"   Error listing contents: {e}")
    
    # Try to add src to Python path
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"\n✅ Added {src_path} to Python path")
    
    # Try to import
    try:
        import bayesian_lora
        print(f"✅ bayesian_lora module imported successfully!")
        print(f"   Version: {bayesian_lora.__version__}")
        print(f"   Author: {bayesian_lora.__author__}")
        return True
    except ImportError as e:
        print(f"❌ bayesian_lora import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing bayesian_lora: {e}")
        return False

def test_simple_functionality():
    """Test simple functionality without complex imports."""
    print("\n=== Testing Simple Functionality ===")
    
    try:
        # Try to import just the models module directly
        sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
        
        from bayesian_lora.models.hf_lora import build_huggingface_lora_model
        print("✅ Models module imported successfully")
        
        # Try to create a simple config
        config = {
            'name': 'bert-base-uncased',
            'num_labels': 2,
            'lora': {'rank': 4, 'alpha': 8.0, 'dropout': 0.1}
        }
        
        print("✅ Configuration created successfully")
        print(f"   Model: {config['name']}")
        print(f"   LoRA rank: {config['lora']['rank']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Models module import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Simple Import Test for Cluster Debugging\n")
    
    tests = [
        ("Basic Python Imports", test_basic_imports),
        ("Bayesian LoRA Import", test_bayesian_lora_import),
        ("Simple Functionality", test_simple_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed\n")
            else:
                print(f"❌ {test_name} failed\n")
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}\n")
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your environment is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test Import Paths Script
This script tests different ways to import bayesian_lora to identify path issues.
"""

import os
import sys
import subprocess

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def test_python_path():
    """Test current Python path."""
    print_header("Current Python Path Analysis")
    
    print("Current working directory:", os.getcwd())
    print("\nPython path entries:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    # Check if src directory is in path
    src_path = os.path.join(os.getcwd(), "src")
    if src_path in sys.path:
        print(f"\n✅ src directory found in Python path at index {sys.path.index(src_path)}")
    else:
        print(f"\n❌ src directory NOT found in Python path")
        print(f"   Expected: {src_path}")
    
    # Check if bayesian_lora directory exists
    bayesian_lora_path = os.path.join(src_path, "bayesian_lora")
    if os.path.exists(bayesian_lora_path):
        print(f"✅ bayesian_lora directory exists: {bayesian_lora_path}")
    else:
        print(f"❌ bayesian_lora directory does not exist: {bayesian_lora_path}")

def test_direct_imports():
    """Test different import approaches."""
    print_header("Testing Different Import Approaches")
    
    # Test 1: Direct import (current approach)
    print("1. Testing direct import...")
    try:
        import bayesian_lora
        print("   ✅ Direct import successful!")
        print(f"   Version: {bayesian_lora.__version__}")
        print(f"   File: {bayesian_lora.__file__}")
    except Exception as e:
        print(f"   ❌ Direct import failed: {e}")
    
    # Test 2: Import with explicit path manipulation
    print("\n2. Testing import with explicit path...")
    try:
        src_path = os.path.join(os.getcwd(), "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            print(f"   Added {src_path} to Python path")
        
        import bayesian_lora
        print("   ✅ Path-manipulated import successful!")
        print(f"   Version: {bayesian_lora.__version__}")
        print(f"   File: {bayesian_lora.__file__}")
    except Exception as e:
        print(f"   ❌ Path-manipulated import failed: {e}")
    
    # Test 3: Import from specific location
    print("\n3. Testing import from specific location...")
    try:
        bayesian_lora_path = os.path.join(os.getcwd(), "src", "bayesian_lora")
        sys.path.insert(0, bayesian_lora_path)
        print(f"   Added {bayesian_lora_path} to Python path")
        
        import bayesian_lora
        print("   ✅ Specific location import successful!")
        print(f"   Version: {bayesian_lora.__version__}")
        print(f"   File: {bayesian_lora.__file__}")
    except Exception as e:
        print(f"   ❌ Specific location import failed: {e}")

def test_submodule_imports():
    """Test importing specific submodules."""
    print_header("Testing Submodule Imports")
    
    # Reset Python path to original
    original_path = sys.path.copy()
    
    # Test different path configurations
    test_configs = [
        ("Original path", original_path),
        ("src in path", [os.path.join(os.getcwd(), "src")] + original_path),
        ("bayesian_lora in path", [os.path.join(os.getcwd(), "src", "bayesian_lora")] + original_path),
        ("Both paths", [os.path.join(os.getcwd(), "src"), os.path.join(os.getcwd(), "src", "bayesian_lora")] + original_path)
    ]
    
    for config_name, test_path in test_configs:
        print(f"\n--- Testing: {config_name} ---")
        sys.path = test_path
        
        try:
            # Test basic import
            import bayesian_lora
            print(f"   ✅ Basic import successful")
            
            # Test submodule imports
            try:
                from bayesian_lora.data import glue_datasets
                print(f"   ✅ data.glue_datasets import successful")
            except Exception as e:
                print(f"   ❌ data.glue_datasets import failed: {e}")
            
            try:
                from bayesian_lora.models import hf_lora
                print(f"   ✅ models.hf_lora import successful")
            except Exception as e:
                print(f"   ❌ models.hf_lora import failed: {e}")
            
            try:
                from bayesian_lora.samplers import sgld
                print(f"   ✅ samplers.sgld import successful")
            except Exception as e:
                print(f"   ❌ samplers.sgld import failed: {e}")
            
            try:
                from bayesian_lora.utils import lora_params
                print(f"   ✅ utils.lora_params import successful")
            except Exception as e:
                print(f"   ❌ utils.lora_params import failed: {e}")
                
        except Exception as e:
            print(f"   ❌ Basic import failed: {e}")
    
    # Restore original path
    sys.path = original_path

def test_file_structure():
    """Test the actual file structure."""
    print_header("File Structure Analysis")
    
    current_dir = os.getcwd()
    src_dir = os.path.join(current_dir, "src")
    bayesian_lora_dir = os.path.join(src_dir, "bayesian_lora")
    
    print(f"Current directory: {current_dir}")
    print(f"src directory: {src_dir}")
    print(f"bayesian_lora directory: {bayesian_lora_dir}")
    
    # Check if directories exist
    print(f"\nDirectory existence:")
    print(f"  src: {'✅' if os.path.exists(src_dir) else '❌'}")
    print(f"  bayesian_lora: {'✅' if os.path.exists(bayesian_lora_dir) else '❌'}")
    
    # List contents
    if os.path.exists(src_dir):
        print(f"\nsrc contents:")
        try:
            contents = os.listdir(src_dir)
            for item in contents:
                item_path = os.path.join(src_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    print(f"  📄 {item}")
        except Exception as e:
            print(f"  ❌ Error listing src contents: {e}")
    
    if os.path.exists(bayesian_lora_dir):
        print(f"\nbayesian_lora contents:")
        try:
            contents = os.listdir(bayesian_lora_dir)
            for item in contents:
                item_path = os.path.join(bayesian_lora_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                    # Check for __init__.py
                    init_file = os.path.join(item_path, "__init__.py")
                    if os.path.exists(init_file):
                        print(f"    ✅ __init__.py exists")
                    else:
                        print(f"    ❌ __init__.py missing")
                else:
                    print(f"  📄 {item}")
        except Exception as e:
            print(f"  ❌ Error listing bayesian_lora contents: {e}")

def test_pip_installation():
    """Test pip installation details."""
    print_header("Pip Installation Analysis")
    
    try:
        # Get pip show output
        result = subprocess.run(["pip3", "show", "bayesian-lora"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ pip show bayesian-lora output:")
            print(result.stdout)
        else:
            print(f"❌ pip show failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error running pip show: {e}")
    
    try:
        # Get pip list output
        result = subprocess.run(["pip3", "list", "|", "grep", "-i", "bayesian"], 
                              shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("\n✅ pip list | grep -i bayesian output:")
            print(result.stdout)
        else:
            print(f"\n❌ pip list grep failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error running pip list grep: {e}")

def main():
    """Main function."""
    print("🔍 IMPORT PATH TESTING SCRIPT")
    print("This script will help identify why submodule imports are failing.")
    
    try:
        test_python_path()
        test_file_structure()
        test_pip_installation()
        test_direct_imports()
        test_submodule_imports()
        
        print_header("Test Summary")
        print("✅ Import path testing complete!")
        print("📋 Check the output above to identify the path issue.")
        print("🔧 The most likely issues are:")
        print("   - Python path not including src directory")
        print("   - Missing __init__.py files in subdirectories")
        print("   - Package installed to wrong location")
        
    except Exception as e:
        print(f"\n💥 Test script failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

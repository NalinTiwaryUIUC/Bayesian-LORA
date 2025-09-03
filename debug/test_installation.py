#!/usr/bin/env python3
"""
Simple test to verify package installation status.
"""

import sys
import os

def test_installation():
    """Test if the package is properly installed."""
    print("🔍 Testing Package Installation Status\n")
    
    # Test 1: Basic import
    try:
        import bayesian_lora
        print(f"✅ Basic import successful")
        print(f"   Module location: {bayesian_lora.__file__}")
        
        # Check if it's in site-packages or local
        if 'site-packages' in bayesian_lora.__file__:
            print("   ✅ Package properly installed in site-packages")
            installation_type = "proper"
        else:
            print("   ⚠️  Package imported from local path")
            installation_type = "local"
            
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False
    
    # Test 2: Submodule imports
    print(f"\n=== Testing Submodule Imports ===")
    
    submodules = [
        "bayesian_lora.models.hf_lora",
        "bayesian_lora.data.glue_datasets", 
        "bayesian_lora.samplers.sgld",
        "bayesian_lora.utils.lora_params"
    ]
    
    submodule_ok = True
    for submodule in submodules:
        try:
            __import__(submodule, fromlist=[''])
            print(f"✅ {submodule}")
        except ImportError as e:
            print(f"❌ {submodule}: {e}")
            submodule_ok = False
    
    # Test 3: Check Python path
    print(f"\n=== Python Path Analysis ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    src_in_path = False
    for i, path in enumerate(sys.path):
        if 'src' in path and os.path.abspath(path) == os.path.abspath(os.path.join(os.getcwd(), 'src')):
            print(f"✅ src directory found in Python path at index {i}: {path}")
            src_in_path = True
            break
    
    if not src_in_path:
        print("❌ src directory NOT found in Python path")
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 INSTALLATION STATUS SUMMARY")
    print(f"{'='*50}")
    
    if installation_type == "proper":
        print("🎉 Package is PROPERLY INSTALLED")
        print("   - Should work from anywhere")
        print("   - Submodules should import correctly")
        print("   - No PYTHONPATH manipulation needed")
    elif installation_type == "local":
        if submodule_ok:
            print("✅ Package is working via LOCAL PATH")
            print("   - This is normal for editable installs")
            print("   - Submodules are working")
            print("   - Package is properly configured")
        else:
            print("⚠️  Package has PARTIAL installation")
            print("   - Basic import works")
            print("   - Submodule imports failing")
            print("   - This suggests installation issues")
    
    if not submodule_ok:
        print(f"\n🚨 ISSUE DETECTED:")
        print(f"   - Basic import: ✅ Working")
        print(f"   - Submodule imports: ❌ Failing")
        print(f"   - This explains the cluster errors!")
        print(f"   - Need to fix pip install -e . on cluster")
    
    return submodule_ok

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)

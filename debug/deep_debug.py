#!/usr/bin/env python3
"""
Deep debugging script to understand package installation issues on cluster.
"""

import sys
import os
import subprocess

def check_python_environment():
    """Check Python environment details."""
    print("=== Python Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Current user: {os.getenv('USER', 'Unknown')}")
    print()
    return True

def check_package_installation():
    """Check if the package is properly installed."""
    print("=== Package Installation Check ===")
    
    # Check if package is in site-packages
    try:
        import bayesian_lora
        print(f"✅ bayesian_lora imported from: {bayesian_lora.__file__}")
        
        # Check if it's in site-packages or local
        if 'site-packages' in bayesian_lora.__file__:
            print("✅ Package properly installed in site-packages")
        else:
            print("⚠️  Package imported from local path (not properly installed)")
            
    except ImportError as e:
        print(f"❌ bayesian_lora import failed: {e}")
        return False
    
    return True

def check_submodule_imports():
    """Check if submodules can be imported."""
    print("\n=== Submodule Import Check ===")
    
    submodules = [
        ("bayesian_lora.models.hf_lora", "HF LoRA Models"),
        ("bayesian_lora.data.glue_datasets", "GLUE Datasets"),
        ("bayesian_lora.samplers.sgld", "SGLD Samplers"),
        ("bayesian_lora.utils.lora_params", "LoRA Utils"),
    ]
    
    all_imports_ok = True
    for module_path, display_name in submodules:
        try:
            module = __import__(module_path, fromlist=[''])
            print(f"✅ {display_name}: {module_path}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def check_pip_installation():
    """Check pip installation status."""
    print("\n=== Pip Installation Check ===")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            bayesian_lora_found = False
            
            for line in lines:
                if 'bayesian-lora' in line.lower() or 'bayesian_lora' in line.lower():
                    print(f"✅ Found in pip list: {line.strip()}")
                    bayesian_lora_found = True
                    break
            
            if not bayesian_lora_found:
                print("❌ bayesian-lora not found in pip list")
                return False
        else:
            print(f"❌ pip list failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running pip list: {e}")
        return False
    
    return True

def check_development_installation():
    """Check if this is a development installation."""
    print("\n=== Development Installation Check ===")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'bayesian-lora'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            output = result.stdout
            print("✅ Package found in pip show:")
            
            # Parse the output
            lines = output.split('\n')
            for line in lines:
                if line.startswith('Location:'):
                    print(f"   {line}")
                elif line.startswith('Editable project location:'):
                    print(f"   {line}")
                elif line.startswith('Version:'):
                    print(f"   {line}")
        else:
            print(f"❌ pip show failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running pip show: {e}")
        return False
    
    return True

def check_file_structure():
    """Check the file structure."""
    print("\n=== File Structure Check ===")
    
    src_path = os.path.join(os.getcwd(), 'src')
    if not os.path.exists(src_path):
        print(f"❌ src directory not found at: {src_path}")
        return False
    
    lora_path = os.path.join(src_path, 'bayesian_lora')
    if not os.path.exists(lora_path):
        print(f"❌ bayesian_lora directory not found at: {lora_path}")
        return False
    
    print(f"✅ Project structure found:")
    print(f"   src: {src_path}")
    print(f"   bayesian_lora: {lora_path}")
    
    # Check key files
    key_files = [
        'src/bayesian_lora/__init__.py',
        'src/bayesian_lora/data/__init__.py',
        'src/bayesian_lora/models/__init__.py',
        'src/bayesian_lora/samplers/__init__.py',
        'src/bayesian_lora/utils/__init__.py',
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
    
    return True

def check_egg_info():
    """Check if egg-info was created."""
    print("\n=== Egg-Info Check ===")
    
    egg_info_path = os.path.join(os.getcwd(), 'src', 'Bayesian_LORA.egg-info')
    if os.path.exists(egg_info_path):
        print(f"✅ Egg-info found at: {egg_info_path}")
        
        # List contents
        try:
            contents = os.listdir(egg_info_path)
            print(f"   Contents: {contents}")
        except Exception as e:
            print(f"   Error listing contents: {e}")
    else:
        print(f"❌ Egg-info not found at: {egg_info_path}")
        return False
    
    return True

def main():
    """Run all checks."""
    print("🔍 Deep Debugging for Cluster Package Issues\n")
    
    checks = [
        ("Python Environment", check_python_environment),
        ("File Structure", check_file_structure),
        ("Egg-Info", check_egg_info),
        ("Package Installation", check_package_installation),
        ("Pip Installation", check_pip_installation),
        ("Development Installation", check_development_installation),
        ("Submodule Imports", check_submodule_imports),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            if check_func():
                results.append((check_name, True))
            else:
                results.append((check_name, False))
        except Exception as e:
            print(f"💥 {check_name} crashed: {e}")
            results.append((check_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DEEP DEBUG SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for check_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Package is properly installed.")
        return True
    else:
        print("⚠️  Some checks failed. This explains the import issues.")
        print("\n🔧 Recommended fixes:")
        print("1. Check if pip install -e . actually succeeded")
        print("2. Verify virtual environment is properly activated")
        print("3. Check Python and pip versions")
        print("4. Ensure all __init__.py files exist")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

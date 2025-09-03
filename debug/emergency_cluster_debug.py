#!/usr/bin/env python3
"""
Emergency Cluster Debug Script
This script works even when bayesian_lora package is not properly installed.
It provides basic diagnostics to help identify cluster-specific issues.
"""

import os
import sys
import subprocess
import traceback

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def run_command(cmd, description):
    """Run a command and capture output."""
    print(f"\n📋 {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ Success: {result.stdout.strip()}")
            return result.stdout.strip()
        else:
            print(f"❌ Failed (exit code {result.returncode}): {result.stderr.strip()}")
            return None
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out after 30 seconds")
        return None
    except Exception as e:
        print(f"💥 Error: {e}")
        return None

def check_basic_environment():
    """Check basic Python environment."""
    print_header("Basic Environment Check")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Current user: {os.getenv('USER', 'Unknown')}")
    print(f"Home directory: {os.getenv('HOME', 'Unknown')}")
    
    # Check if we're in a virtual environment
    venv = os.getenv('VIRTUAL_ENV')
    if venv:
        print(f"✅ Virtual environment: {venv}")
    else:
        print("⚠️  No virtual environment detected")

def check_file_structure():
    """Check the file structure without importing."""
    print_header("File Structure Check")
    
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if src directory exists
    src_dir = os.path.join(current_dir, "src")
    if os.path.exists(src_dir):
        print(f"✅ src directory found: {src_dir}")
        
        # List contents of src
        try:
            src_contents = os.listdir(src_dir)
            print(f"📁 src contents: {src_contents}")
            
            # Check bayesian_lora subdirectory
            bayesian_lora_dir = os.path.join(src_dir, "bayesian_lora")
            if os.path.exists(bayesian_lora_dir):
                print(f"✅ bayesian_lora directory found: {bayesian_lora_dir}")
                
                # Check __init__.py files
                init_files = []
                for root, dirs, files in os.walk(bayesian_lora_dir):
                    for file in files:
                        if file == "__init__.py":
                            init_files.append(os.path.relpath(os.path.join(root, file), bayesian_lora_dir))
                
                print(f"📄 __init__.py files found: {len(init_files)}")
                for init_file in init_files[:10]:  # Show first 10
                    print(f"   - {init_file}")
                if len(init_files) > 10:
                    print(f"   ... and {len(init_files) - 10} more")
            else:
                print(f"❌ bayesian_lora directory not found in {src_dir}")
        except PermissionError:
            print("❌ Permission denied accessing src directory")
        except Exception as e:
            print(f"❌ Error listing src contents: {e}")
    else:
        print(f"❌ src directory not found in {current_dir}")
    
    # Check if we're in the right place
    if "Bayesian-LORA" in current_dir:
        print("✅ Appears to be in Bayesian-LORA project directory")
    else:
        print("⚠️  May not be in Bayesian-LORA project directory")

def check_pip_status():
    """Check pip and package status."""
    print_header("Pip and Package Status")
    
    # Check pip version
    pip_version = run_command("pip3 --version", "Checking pip version")
    
    # Check if bayesian-lora is installed
    pip_show = run_command("pip3 show bayesian-lora", "Checking bayesian-lora package status")
    
    # Check pip list
    pip_list = run_command("pip3 list | grep -i bayesian", "Checking for Bayesian packages in pip list")
    
    # Check if we can install packages
    print("\n📦 Testing package installation capability...")
    try:
        # Try to install a simple package to test pip
        result = subprocess.run(
            ["pip3", "install", "--dry-run", "requests"], 
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("✅ pip3 install --dry-run works (can install packages)")
        else:
            print(f"❌ pip3 install --dry-run failed: {result.stderr.strip()}")
    except Exception as e:
        print(f"❌ Error testing pip install: {e}")

def check_python_imports():
    """Check basic Python imports without bayesian_lora."""
    print_header("Basic Python Import Check")
    
    basic_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("datasets", "Datasets"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("numpy", "NumPy"),
        ("sklearn", "scikit-learn")
    ]
    
    for module_name, display_name in basic_packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {display_name}: {version}")
        except ImportError:
            print(f"❌ {display_name}: Not available")
        except Exception as e:
            print(f"⚠️  {display_name}: Error importing - {e}")

def check_editable_install_issues():
    """Check common editable install issues."""
    print_header("Editable Install Issue Analysis")
    
    current_dir = os.getcwd()
    
    # Check if setup.py exists
    setup_py = os.path.join(current_dir, "setup.py")
    if os.path.exists(setup_py):
        print(f"✅ setup.py found: {setup_py}")
        
        # Check setup.py permissions
        try:
            stat_info = os.stat(setup_py)
            print(f"📄 setup.py permissions: {oct(stat_info.st_mode)[-3:]}")
        except Exception as e:
            print(f"❌ Error checking setup.py permissions: {e}")
    else:
        print(f"❌ setup.py not found in {current_dir}")
    
    # Check if pyproject.toml exists
    pyproject_toml = os.path.join(current_dir, "pyproject.toml")
    if os.path.exists(pyproject_toml):
        print(f"✅ pyproject.toml found: {pyproject_toml}")
    else:
        print(f"❌ pyproject.toml not found in {current_dir}")
    
    # Check if src directory has proper structure
    src_dir = os.path.join(current_dir, "src")
    if os.path.exists(src_dir):
        print(f"✅ src directory exists: {src_dir}")
        
        # Check if src is writable
        try:
            test_file = os.path.join(src_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("✅ src directory is writable")
        except Exception as e:
            print(f"❌ src directory is not writable: {e}")
    else:
        print(f"❌ src directory does not exist: {src_dir}")
    
    # Check if we're in a virtual environment
    venv = os.getenv('VIRTUAL_ENV')
    if venv:
        print(f"✅ In virtual environment: {venv}")
        
        # Check if virtual environment is writable
        try:
            test_file = os.path.join(venv, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("✅ Virtual environment is writable")
        except Exception as e:
            print(f"❌ Virtual environment is not writable: {e}")
    else:
        print("⚠️  Not in virtual environment")

def suggest_fixes():
    """Suggest fixes based on the analysis."""
    print_header("Suggested Fixes")
    
    print("🔧 Based on the analysis, here are potential fixes:")
    print()
    
    print("1. **If src directory is not writable:**")
    print("   - Check file permissions: ls -la src/")
    print("   - Fix permissions: chmod -R 755 src/")
    print()
    
    print("2. **If virtual environment is not writable:**")
    print("   - Recreate virtual environment: python3 -m venv .venv")
    print("   - Activate: source .venv/bin/activate")
    print()
    
    print("3. **If pip install -e . fails:**")
    print("   - Try: pip3 install --user -e .")
    print("   - Or: pip3 install --force-reinstall -e .")
    print()
    
    print("4. **If submodules can't be imported:**")
    print("   - Check __init__.py files exist in all subdirectories")
    print("   - Verify Python path includes src directory")
    print()
    
    print("5. **Alternative installation methods:**")
    print("   - pip3 install . (non-editable)")
    print("   - pip3 install -r requirements_lora.txt + export PYTHONPATH")
    print()
    
    print("6. **Check cluster-specific issues:**")
    print("   - Module loading: module list")
    print("   - Python version: python3 --version")
    print("   - Environment variables: env | grep PYTHON")

def main():
    """Main function."""
    print("🚨 EMERGENCY CLUSTER DEBUG SCRIPT")
    print("This script works even when bayesian_lora package is broken!")
    print()
    
    try:
        check_basic_environment()
        check_file_structure()
        check_pip_status()
        check_python_imports()
        check_editable_install_issues()
        suggest_fixes()
        
        print_header("Debug Summary")
        print("✅ Emergency debugging complete!")
        print("📋 Check the output above for issues and suggested fixes.")
        print("🔧 The most common issues are:")
        print("   - File permissions on src/ directory")
        print("   - Virtual environment not writable")
        print("   - Python path not including src/")
        print("   - Missing __init__.py files")
        
    except Exception as e:
        print(f"\n💥 Emergency debug script failed: {e}")
        print("Stack trace:")
        traceback.print_exc()

if __name__ == "__main__":
    main()

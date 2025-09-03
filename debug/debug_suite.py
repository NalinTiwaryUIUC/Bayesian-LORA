#!/usr/bin/env python3
"""
Comprehensive Debug Suite for Bayesian LoRA
===========================================

This script consolidates all debugging functionality into one comprehensive tool.
It provides systematic diagnostics for:
- Environment checks
- Import verification
- Model creation
- Data loading
- Package installation status

Usage:
    python3 debug/debug_suite.py [--quick] [--verbose] [--fix]
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch

class BayesianLoRADebugger:
    """Comprehensive debug suite for Bayesian LoRA project."""
    
    def __init__(self, verbose: bool = False, auto_fix: bool = False):
        self.verbose = verbose
        self.auto_fix = auto_fix
        self.results = {}
        self.project_root = Path(__file__).parent.parent
        self.src_path = self.project_root / "src"
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with level."""
        prefix = f"[{level}]" if level != "INFO" else ""
        print(f"{prefix} {message}")
        
    def run_command(self, command: str, capture_output: bool = True) -> Tuple[int, str, str]:
        """Run shell command and return results."""
        try:
            if capture_output:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=self.project_root)
                return result.returncode, result.stdout, result.stderr
            else:
                result = subprocess.run(command, shell=True, cwd=self.project_root)
                return result.returncode, "", ""
        except Exception as e:
            return -1, "", str(e)
    
    def check_environment(self) -> Dict[str, bool]:
        """Check basic environment setup."""
        self.log("🔍 Checking environment...")
        results = {}
        
        # Python version
        python_version = sys.version_info
        results['python_version'] = python_version.major == 3 and python_version.minor >= 9
        self.log(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # CUDA availability
        results['cuda_available'] = torch.cuda.is_available()
        if results['cuda_available']:
            self.log(f"CUDA available: {torch.version.cuda}")
            self.log(f"GPU count: {torch.cuda.device_count()}")
        else:
            self.log("CUDA not available")
            
        # Working directory
        results['correct_directory'] = "Bayesian-LORA" in os.getcwd()
        self.log(f"Working directory: {os.getcwd()}")
        
        # Virtual environment
        results['venv_active'] = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        self.log(f"Virtual environment: {'Active' if results['venv_active'] else 'Not active'}")
        
        return results
    
    def check_package_installation(self) -> Dict[str, bool]:
        """Check package installation status."""
        self.log("📦 Checking package installation...")
        results = {}
        
        # Check if package is installed
        try:
            import bayesian_lora
            results['package_imported'] = True
            self.log(f"✅ Package imported: {getattr(bayesian_lora, '__version__', 'Unknown version')}")
        except ImportError as e:
            results['package_imported'] = False
            self.log(f"❌ Package import failed: {e}")
            
        # Check pip installation
        returncode, stdout, stderr = self.run_command("pip3 show bayesian-lora")
        results['pip_installed'] = returncode == 0
        if results['pip_installed']:
            self.log("✅ Package found in pip")
        else:
            self.log("❌ Package not found in pip")
            
        # Check editable install
        if results['pip_installed']:
            returncode, stdout, stderr = self.run_command("pip3 show bayesian-lora | grep 'Editable project location'")
            results['editable_install'] = returncode == 0 and "Editable project location" in stdout
            self.log(f"Editable install: {'Yes' if results['editable_install'] else 'No'}")
            
        return results
    
    def check_file_structure(self) -> Dict[str, bool]:
        """Check critical file structure."""
        self.log("📁 Checking file structure...")
        results = {}
        
        # Critical directories
        critical_paths = [
            "src/bayesian_lora",
            "src/bayesian_lora/data",
            "src/bayesian_lora/models",
            "src/bayesian_lora/samplers",
            "src/bayesian_lora/utils",
            "configs",
            "scripts"
        ]
        
        for path in critical_paths:
            full_path = self.project_root / path
            exists = full_path.exists()
            results[f"path_{path.replace('/', '_')}"] = exists
            status = "✅" if exists else "❌"
            self.log(f"{status} {path}")
            
        # Critical files
        critical_files = [
            "src/bayesian_lora/__init__.py",
            "src/bayesian_lora/data/__init__.py",
            "src/bayesian_lora/models/__init__.py",
            "src/bayesian_lora/samplers/__init__.py",
            "src/bayesian_lora/utils/__init__.py",
            "pyproject.toml",
            "setup.py"
        ]
        
        for file_path in critical_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            results[f"file_{file_path.replace('/', '_').replace('.', '_')}"] = exists
            status = "✅" if exists else "❌"
            self.log(f"{status} {file_path}")
            
        return results
    
    def check_imports(self) -> Dict[str, bool]:
        """Check module imports."""
        self.log("🔌 Checking module imports...")
        results = {}
        
        # Test basic import
        try:
            import bayesian_lora
            results['basic_import'] = True
            self.log("✅ Basic import successful")
        except ImportError as e:
            results['basic_import'] = False
            self.log(f"❌ Basic import failed: {e}")
            return results
            
        # Test submodule imports
        submodules = [
            'bayesian_lora.data.glue_datasets',
            'bayesian_lora.models.hf_lora',
            'bayesian_lora.samplers.sgld',
            'bayesian_lora.utils.lora_params'
        ]
        
        for submodule in submodules:
            try:
                importlib.import_module(submodule)
                results[f"import_{submodule.replace('.', '_')}"] = True
                self.log(f"✅ {submodule}")
            except ImportError as e:
                results[f"import_{submodule.replace('.', '_')}"] = False
                self.log(f"❌ {submodule}: {e}")
                
        return results
    
    def check_model_creation(self) -> Dict[str, bool]:
        """Check model creation capabilities."""
        self.log("🤖 Checking model creation...")
        results = {}
        
        try:
            from bayesian_lora.models.hf_lora import build_huggingface_lora_model
            
            # Test model creation
            model_config = {
                'name': 'bert-base-uncased',
                'num_labels': 2,
                'lora': {
                    'r': 16,
                    'alpha': 32,
                    'dropout': 0.1
                }
            }
            
            model = build_huggingface_lora_model(model_config)
            results['model_created'] = True
            self.log("✅ Model created successfully")
            
            # Check LoRA parameters
            from bayesian_lora.utils.lora_params import count_lora_parameters
            lora_count = count_lora_parameters(model)
            results['lora_parameters'] = lora_count > 0
            self.log(f"LoRA parameters: {lora_count:,}")
            
        except Exception as e:
            results['model_created'] = False
            results['lora_parameters'] = False
            self.log(f"❌ Model creation failed: {e}")
            
        return results
    
    def check_data_loading(self) -> Dict[str, bool]:
        """Check data loading capabilities."""
        self.log("📊 Checking data loading...")
        results = {}
        
        try:
            from bayesian_lora.data.glue_datasets import create_dataloaders, get_dataset_metadata
            
            # Test dataset metadata
            metadata = get_dataset_metadata('sst2')
            results['metadata_loaded'] = True
            self.log(f"✅ Dataset metadata: {metadata['name']}")
            
            # Test dataloader creation (with minimal config)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            
            train_loader, val_loader = create_dataloaders(
                dataset_name='sst2',
                tokenizer=tokenizer,
                batch_size=8,
                max_length=32,
                num_workers=0  # Avoid multiprocessing issues
            )
            
            results['dataloaders_created'] = True
            self.log(f"✅ Dataloaders created: {len(train_loader)} train, {len(val_loader)} val")
            
        except Exception as e:
            results['metadata_loaded'] = False
            results['dataloaders_created'] = False
            self.log(f"❌ Data loading failed: {e}")
            
        return results
    
    def run_comprehensive_check(self) -> Dict[str, Dict[str, bool]]:
        """Run all checks."""
        self.log("🚀 Starting comprehensive debug check...")
        
        all_results = {
            'environment': self.check_environment(),
            'package': self.check_package_installation(),
            'structure': self.check_file_structure(),
            'imports': self.check_imports(),
            'model': self.check_model_creation(),
            'data': self.check_data_loading()
        }
        
        return all_results
    
    def generate_report(self, results: Dict[str, Dict[str, bool]]) -> str:
        """Generate comprehensive report."""
        report = []
        report.append("=" * 60)
        report.append("🔍 BAYESIAN LORA DEBUG REPORT")
        report.append("=" * 60)
        
        total_checks = 0
        passed_checks = 0
        
        for category, checks in results.items():
            report.append(f"\n📋 {category.upper()} CHECKS:")
            report.append("-" * 40)
            
            for check_name, passed in checks.items():
                total_checks += 1
                if passed:
                    passed_checks += 1
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                    
                report.append(f"{status} {check_name}")
                
        # Summary
        report.append("\n" + "=" * 60)
        report.append(f"📊 SUMMARY: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            report.append("🎉 ALL CHECKS PASSED! Your environment is ready.")
        else:
            report.append("⚠️  Some checks failed. Check the details above.")
            
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def suggest_fixes(self, results: Dict[str, Dict[str, bool]]) -> List[str]:
        """Suggest fixes based on failed checks."""
        suggestions = []
        
        # Package fixes
        if 'package' in results:
            if not results['package'].get('package_imported', True):
                suggestions.append("🔧 Run: pip3 install -e .")
                
        # Structure fixes
        if 'structure' in results:
            if not results['structure'].get('file_src_bayesian_lora_data___init___py', True):
                suggestions.append("🔧 Check if src/bayesian_lora/data/__init__.py exists and has content")
                
        # Import fixes
        if 'imports' in results:
            if not results['imports'].get('basic_import', True):
                suggestions.append("🔧 Check Python path includes src/ directory")
                suggestions.append("🔧 Verify package installation with: pip3 show bayesian-lora")
                
        # Model fixes
        if 'model' in results:
            if not results['model'].get('model_created', True):
                suggestions.append("🔧 Check model dependencies: transformers, peft, torch")
                
        # Data fixes
        if 'data' in results:
            if not results['data'].get('dataloaders_created', True):
                suggestions.append("🔧 Check dataset dependencies: datasets, tokenizers")
                
        # Environment fixes
        if 'environment' in results:
            if not results['environment'].get('venv_active', True):
                suggestions.append("🔧 Activate virtual environment: source .venv/bin/activate")
                
        return suggestions

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bayesian LoRA Debug Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick checks only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--fix", action="store_true", help="Auto-fix common issues")
    
    args = parser.parse_args()
    
    debugger = BayesianLoRADebugger(verbose=args.verbose, auto_fix=args.fix)
    
    if args.quick:
        # Quick checks
        results = {
            'environment': debugger.check_environment(),
            'package': debugger.check_package_installation(),
            'imports': debugger.check_imports()
        }
    else:
        # Comprehensive checks
        results = debugger.run_comprehensive_check()
    
    # Generate report
    report = debugger.generate_report(results)
    print(report)
    
    # Suggest fixes
    suggestions = debugger.suggest_fixes(results)
    if suggestions:
        print("\n🔧 SUGGESTED FIXES:")
        for suggestion in suggestions:
            print(f"  {suggestion}")
    
    # Exit with appropriate code
    total_checks = sum(len(checks) for checks in results.values())
    passed_checks = sum(sum(checks.values()) for checks in results.values())
    
    if passed_checks == total_checks:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Some checks failed

if __name__ == "__main__":
    main()

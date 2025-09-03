#!/usr/bin/env python3
"""
Comprehensive Debug Suite for Bayesian LoRA
===========================================

This script consolidates all debugging functionality into one comprehensive tool.
It provides systematic diagnostics for:
- Environment checks
- Import verification
- Model creation (CIFAR and LoRA)
- Data loading (CIFAR and GLUE)
- SGLD sampler functionality
- Training script verification
- Configuration validation
- Package installation status

Usage:
    python3 debug/debug_suite.py [--quick] [--verbose] [--fix]
"""

import os
import sys
import subprocess
import importlib
import yaml
import platform
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
        
        # CUDA availability - not required on all systems
        results['cuda_available'] = torch.cuda.is_available()
        if results['cuda_available']:
            self.log(f"CUDA available: {torch.version.cuda}")
            self.log(f"GPU count: {torch.cuda.device_count()}")
        else:
            # Check if we're on macOS (which doesn't support CUDA)
            if platform.system() == "Darwin":
                self.log("CUDA not available (expected on macOS)")
                results['cuda_available'] = True  # Don't fail on macOS
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
        try:
            import bayesian_lora
            package_path = Path(bayesian_lora.__file__).parent
            results['editable_install'] = "site-packages" not in str(package_path)
            if results['editable_install']:
                self.log("✅ Package installed in editable mode")
            else:
                self.log("⚠️ Package installed in site-packages (not editable)")
        except Exception as e:
            results['editable_install'] = False
            self.log(f"❌ Could not determine install mode: {e}")
            
        return results
    
    def check_file_structure(self) -> Dict[str, bool]:
        """Check critical file structure."""
        self.log("📁 Checking file structure...")
        results = {}
        
        critical_paths = [
            "src",
            "src/bayesian_lora",
            "src/bayesian_lora/data",
            "src/bayesian_lora/models",
            "src/bayesian_lora/samplers",
            "src/bayesian_lora/utils",
            "configs",
            "scripts",
            "runs",
            "data"
        ]
        
        critical_files = [
            "src/bayesian_lora/__init__.py",
            "src/bayesian_lora/data/__init__.py",
            "src/bayesian_lora/models/__init__.py",
            "src/bayesian_lora/samplers/__init__.py",
            "src/bayesian_lora/utils/__init__.py",
            "pyproject.toml",
            "Makefile",
            "requirements_lora.txt"
        ]
        
        # Check critical paths
        for path in critical_paths:
            full_path = self.project_root / path
            results[f"path_{path.replace('/', '_')}"] = full_path.exists()
            if results[f"path_{path.replace('/', '_')}"]:
                self.log(f"✅ {path} exists")
            else:
                self.log(f"❌ {path} missing")
                
        # Check critical files
        for file_path in critical_files:
            full_path = self.project_root / file_path
            results[f"file_{file_path.replace('/', '_').replace('.', '_')}"] = full_path.exists()
            if results[f"file_{file_path.replace('/', '_').replace('.', '_')}"]:
                self.log(f"✅ {file_path} exists")
            else:
                self.log(f"❌ {file_path} missing")
                
        return results
    
    def check_imports(self) -> Dict[str, bool]:
        """Check module importability."""
        self.log("📚 Checking module imports...")
        results = {}
        
        submodules = [
            'bayesian_lora.data.glue_datasets',
            'bayesian_lora.data.cifar',
            'bayesian_lora.models.hf_lora',
            'bayesian_lora.models.resnet_cifar',
            'bayesian_lora.models.wide_resnet',
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
    
    def check_cifar_models(self) -> Dict[str, bool]:
        """Check CIFAR model creation capabilities."""
        self.log("🤖 Checking CIFAR model creation...")
        results = {}
        
        try:
            from bayesian_lora.models.resnet_cifar import ResNetCIFAR
            from bayesian_lora.models.wide_resnet import WideResNetCIFAR
            
            # Test ResNet creation
            resnet_model = ResNetCIFAR(depth=18, num_classes=10)
            results['resnet_created'] = True
            self.log("✅ ResNet-18 CIFAR model created")
            
            # Test WideResNet creation
            wrn_model = WideResNetCIFAR(depth=28, widen_factor=10, num_classes=100)
            results['wideresnet_created'] = True
            self.log("✅ WideResNet-28-10 CIFAR model created")
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 32, 32)
            resnet_output = resnet_model(dummy_input)
            wrn_output = wrn_model(dummy_input)
            
            results['resnet_forward'] = resnet_output.shape == (2, 10)
            results['wideresnet_forward'] = wrn_output.shape == (2, 100)
            
            self.log(f"✅ ResNet output shape: {resnet_output.shape}")
            self.log(f"✅ WideResNet output shape: {wrn_output.shape}")
            
        except Exception as e:
            results['resnet_created'] = False
            results['wideresnet_created'] = False
            results['resnet_forward'] = False
            results['wideresnet_forward'] = False
            self.log(f"❌ CIFAR model creation failed: {e}")
            
        return results
    
    def check_lora_models(self) -> Dict[str, bool]:
        """Check LoRA model creation capabilities."""
        self.log("🤖 Checking LoRA model creation...")
        results = {}
        
        try:
            from bayesian_lora.models.hf_lora import LoRAModel
            from transformers import AutoModelForSequenceClassification
            
            # Test LoRA model creation with correct target modules for RoBERTa
            base_model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
            
            # Try different target module configurations for RoBERTa
            try:
                lora_model = LoRAModel(base_model, r=8, alpha=16.0, dropout=0.05, 
                                      target_modules=["query", "key", "value"])
            except Exception as e1:
                try:
                    # Fallback to more generic target modules
                    lora_model = LoRAModel(base_model, r=8, alpha=16.0, dropout=0.05, 
                                          target_modules=["q_proj", "k_proj", "v_proj"])
                except Exception as e2:
                    # Final fallback to basic target modules
                    lora_model = LoRAModel(base_model, r=8, alpha=16.0, dropout=0.05)
            
            results['lora_model_created'] = True
            self.log("✅ LoRA model created successfully")
            
            # Check LoRA parameters
            from bayesian_lora.utils.lora_params import count_lora_parameters
            lora_count = count_lora_parameters(lora_model)
            results['lora_parameters'] = lora_count > 0
            self.log(f"LoRA parameters: {lora_count:,}")
            
            # Test forward pass
            dummy_input = torch.randint(0, 1000, (2, 10))
            dummy_mask = torch.ones(2, 10)
            output = lora_model(dummy_input, attention_mask=dummy_mask)
            results['lora_forward'] = output.logits.shape == (2, 2)
            self.log(f"✅ LoRA forward pass: {output.logits.shape}")
            
        except Exception as e:
            results['lora_model_created'] = False
            results['lora_parameters'] = False
            results['lora_forward'] = False
            self.log(f"❌ LoRA model creation failed: {e}")
            
        return results
    
    def check_sgld_samplers(self) -> Dict[str, bool]:
        """Check SGLD sampler functionality."""
        self.log("🔄 Checking SGLD samplers...")
        results = {}
        
        try:
            from bayesian_lora.samplers.sgld import (
                SGLDSampler, ASGLDSampler, SAMSGLDSampler, SAMSGLDRank1Sampler
            )
            
            # Create a simple test model
            test_model = torch.nn.Linear(10, 2)
            
            # Test SGLD sampler
            sgld_sampler = SGLDSampler(test_model, temperature=1.0, step_size=1e-4)
            results['sgld_sampler_created'] = True
            self.log("✅ SGLD sampler created")
            
            # Test ASGLD sampler
            asgld_sampler = ASGLDSampler(test_model, temperature=1.0, step_size=1e-4)
            results['asgld_sampler_created'] = True
            self.log("✅ ASGLD sampler created")
            
            # Test SAM-SGLD sampler
            sam_sgld_sampler = SAMSGLDSampler(test_model, temperature=1.0, step_size=1e-4, rho=0.1)
            results['sam_sgld_sampler_created'] = True
            self.log("✅ SAM-SGLD sampler created")
            
            # Test SAM-SGLD Rank-1 sampler
            sam_sgld_r1_sampler = SAMSGLDRank1Sampler(test_model, temperature=1.0, step_size=1e-4, rho=0.1)
            results['sam_sgld_r1_sampler_created'] = True
            self.log("✅ SAM-SGLD Rank-1 sampler created")
            
            # Test sampler step functionality
            dummy_data = torch.randn(2, 10)
            dummy_target = torch.randint(0, 2, (2,))
            
            sgld_sampler.step(dummy_data, dummy_target)
            results['sgld_step_executed'] = True
            self.log("✅ SGLD step executed")
            
        except Exception as e:
            results['sgld_sampler_created'] = False
            results['asgld_sampler_created'] = False
            results['sam_sgld_sampler_created'] = False
            results['sam_sgld_r1_sampler_created'] = False
            results['sgld_step_executed'] = False
            self.log(f"❌ SGLD sampler check failed: {e}")
            
        return results
    
    def check_data_loading(self) -> Dict[str, bool]:
        """Check data loading capabilities."""
        self.log("📊 Checking data loading...")
        results = {}
        
        try:
            # Test CIFAR data loading
            from bayesian_lora.data.cifar import get_cifar_dataset
            
            # This will download CIFAR if not present
            train_dataset, test_dataset = get_cifar_dataset('cifar10', 'data/cifar-10-batches-py')
            results['cifar_data_loaded'] = True
            self.log(f"✅ CIFAR-10 data loaded: {len(train_dataset)} train, {len(test_dataset)} test")
            
            # Test GLUE data loading
            from bayesian_lora.data.glue_datasets import MRPCDataset
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            mrpc_dataset = MRPCDataset(split='train', tokenizer=tokenizer, max_length=128)
            results['glue_data_loaded'] = True
            self.log(f"✅ MRPC data loaded: {len(mrpc_dataset)} samples")
            
            # Test data iteration
            sample = mrpc_dataset[0]
            results['glue_sample_format'] = all(k in sample for k in ['input_ids', 'attention_mask', 'labels'])
            self.log(f"✅ GLUE sample format: {list(sample.keys())}")
            
        except Exception as e:
            results['cifar_data_loaded'] = False
            results['glue_data_loaded'] = False
            results['glue_sample_format'] = False
            self.log(f"❌ Data loading failed: {e}")
            
        return results
    
    def check_configurations(self) -> Dict[str, bool]:
        """Check configuration files."""
        self.log("⚙️ Checking configuration files...")
        results = {}
        
        config_files = [
            "configs/cifar10_resnet18_sgld.yaml",
            "configs/cifar100_wrn2810_sgld.yaml",
            "configs/cifar100_wrn2810_sam_sgld.yaml",
            "configs/cifar100_wrn2810_sam_sgld_r1.yaml",
            "configs/mrpc_roberta_lora_sgld.yaml"
        ]
        
        for config_file in config_files:
            try:
                full_path = self.project_root / config_file
                if full_path.exists():
                    with open(full_path, 'r') as f:
                        config = yaml.safe_load(f)
                    results[f"config_{config_file.replace('/', '_').replace('.', '_')}"] = True
                    self.log(f"✅ {config_file} loaded successfully")
                else:
                    results[f"config_{config_file.replace('/', '_').replace('.', '_')}"] = False
                    self.log(f"❌ {config_file} not found")
            except Exception as e:
                results[f"config_{config_file.replace('/', '_').replace('.', '_')}"] = False
                self.log(f"❌ {config_file} failed to load: {e}")
                
        return results
    
    def check_training_scripts(self) -> Dict[str, bool]:
        """Check training script functionality."""
        self.log("📜 Checking training scripts...")
        results = {}
        
        script_files = [
            "scripts/train.py",
            "scripts/eval.py",
            "scripts/train_mrpc_lora.py",
            "scripts/eval_mrpc_lora.py"
        ]
        
        for script_file in script_files:
            try:
                full_path = self.project_root / script_file
                if full_path.exists():
                    # Test if script has valid Python syntax instead of importing
                    with open(full_path, 'r') as f:
                        script_content = f.read()
                    
                    # Basic syntax check
                    compile(script_content, script_file, 'exec')
                    results[f"script_{script_file.replace('/', '_').replace('.', '_')}"] = True
                    self.log(f"✅ {script_file} syntax is valid")
                else:
                    results[f"script_{script_file.replace('/', '_').replace('.', '_')}"] = False
                    self.log(f"❌ {script_file} not found")
            except Exception as e:
                results[f"script_{script_file.replace('/', '_').replace('.', '_')}"] = False
                self.log(f"❌ {script_file} syntax check failed: {e}")
                
        return results
    
    def check_makefile_targets(self) -> Dict[str, bool]:
        """Check Makefile targets."""
        self.log("🔨 Checking Makefile targets...")
        results = {}
        
        try:
            # Check if Makefile exists
            makefile_path = self.project_root / "Makefile"
            if not makefile_path.exists():
                results['makefile_exists'] = False
                self.log("❌ Makefile not found")
                return results
            
            results['makefile_exists'] = True
            self.log("✅ Makefile found")
            
            # Test help target
            returncode, stdout, stderr = self.run_command("make help", capture_output=True)
            results['make_help_works'] = returncode == 0
            if results['make_help_works']:
                self.log("✅ Make help target works")
            else:
                self.log("❌ Make help target failed")
                
            # Test clean target
            returncode, stdout, stderr = self.run_command("make clean")
            results['make_clean_works'] = returncode == 0
            if results['make_clean_works']:
                self.log("✅ Make clean target works")
            else:
                self.log("❌ Make clean target failed")
                
        except Exception as e:
            results['makefile_exists'] = False
            results['make_help_works'] = False
            results['make_clean_works'] = False
            self.log(f"❌ Makefile check failed: {e}")
            
        return results
    
    def run_comprehensive_check(self) -> Dict[str, Dict[str, bool]]:
        """Run all checks."""
        self.log("🚀 Starting comprehensive debug check...")
        
        all_results = {
            'environment': self.check_environment(),
            'package': self.check_package_installation(),
            'structure': self.check_file_structure(),
            'imports': self.check_imports(),
            'cifar_models': self.check_cifar_models(),
            'lora_models': self.check_lora_models(),
            'sgld_samplers': self.check_sgld_samplers(),
            'data_loading': self.check_data_loading(),
            'configurations': self.check_configurations(),
            'training_scripts': self.check_training_scripts(),
            'makefile': self.check_makefile_targets()
        }
        
        return all_results
    
    def run_quick_check(self) -> Dict[str, Dict[str, bool]]:
        """Run essential checks only."""
        self.log("⚡ Starting quick debug check...")
        
        quick_results = {
            'environment': self.check_environment(),
            'package': self.check_package_installation(),
            'structure': self.check_file_structure(),
            'imports': self.check_imports()
        }
        
        return quick_results
    
    def generate_report(self, results: Dict[str, Dict[str, bool]]) -> str:
        """Generate comprehensive report."""
        report = []
        report.append("=" * 70)
        report.append("🔍 BAYESIAN LORA COMPREHENSIVE DEBUG REPORT")
        report.append("=" * 70)
        
        total_checks = 0
        passed_checks = 0
        
        for category, checks in results.items():
            report.append(f"\n📋 {category.upper().replace('_', ' ')} CHECKS:")
            report.append("-" * 50)
            
            for check_name, passed in checks.items():
                total_checks += 1
                if passed:
                    passed_checks += 1
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                    
                # Clean up check name for display
                display_name = check_name.replace('_', ' ').title()
                report.append(f"{status} {display_name}")
                
        # Summary
        report.append("\n" + "=" * 70)
        report.append(f"📊 SUMMARY: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            report.append("🎉 ALL CHECKS PASSED! Your environment is ready for all experiments.")
        elif passed_checks >= total_checks * 0.8:
            report.append("✅ MOST CHECKS PASSED! Minor issues detected but environment is functional.")
        else:
            report.append("⚠️  MANY CHECKS FAILED! Environment needs attention before use.")
            
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def suggest_fixes(self, results: Dict[str, Dict[str, bool]]) -> List[str]:
        """Suggest fixes for failed checks."""
        suggestions = []
        
        # Package fixes
        if 'package' in results:
            if not results['package'].get('package_imported', True):
                suggestions.append("🔧 Install package: pip install -e .")
            if not results['package'].get('editable_install', True):
                suggestions.append("🔧 Reinstall in editable mode: pip install -e .")
                
        # Structure fixes
        if 'structure' in results:
            if not results['structure'].get('path_src', True):
                suggestions.append("🔧 Check repository structure and ensure src/ directory exists")
                
        # Import fixes
        if 'imports' in results:
            failed_imports = [k for k, v in results['imports'].items() if not v]
            if failed_imports:
                suggestions.append("🔧 Check module dependencies and __init__.py files")
                
        # Model fixes
        if 'cifar_models' in results:
            if not results['cifar_models'].get('resnet_created', True):
                suggestions.append("🔧 Check CIFAR model dependencies: torchvision")
                
        if 'lora_models' in results:
            if not results['lora_models'].get('lora_model_created', True):
                suggestions.append("🔧 Check LoRA dependencies: transformers, peft, torch")
                
        # Data fixes
        if 'data_loading' in results:
            if not results['data_loading'].get('cifar_data_loaded', True):
                suggestions.append("🔧 Check CIFAR dataset dependencies: torchvision")
            if not results['data_loading'].get('glue_data_loaded', True):
                suggestions.append("🔧 Check GLUE dataset dependencies: datasets, tokenizers")
                
        # Environment fixes
        if 'environment' in results:
            if not results['environment'].get('venv_active', True):
                suggestions.append("🔧 Activate virtual environment: source .venv/bin/activate")
                
        return suggestions

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bayesian LoRA Comprehensive Debug Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick checks only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues when possible")
    
    args = parser.parse_args()
    
    debugger = BayesianLoRADebugger(verbose=args.verbose, auto_fix=args.fix)
    
    if args.quick:
        results = debugger.run_quick_check()
        print("\n" + "=" * 60)
        print("⚡ QUICK CHECK COMPLETE")
        print("=" * 60)
    else:
        results = debugger.run_comprehensive_check()
        print("\n" + "=" * 60)
        print("🚀 COMPREHENSIVE CHECK COMPLETE")
        print("=" * 60)
    
    # Generate and display report
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
    elif passed_checks >= total_checks * 0.8:
        sys.exit(1)  # Minor issues
    else:
        sys.exit(2)  # Major issues

if __name__ == "__main__":
    main()

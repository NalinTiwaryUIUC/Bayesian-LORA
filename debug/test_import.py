#!/usr/bin/env python3
"""
Test script to verify bayesian_lora module installation.
"""

try:
    print("Testing bayesian_lora module import...")
    import bayesian_lora
    print(f"✓ Successfully imported bayesian_lora version {bayesian_lora.__version__}")
    
    print("\nTesting submodule imports...")
    from bayesian_lora.models.hf_lora import build_huggingface_lora_model
    print("✓ hf_lora module imported")
    
    from bayesian_lora.data.glue_datasets import create_dataloaders, get_dataset_metadata
    print("✓ glue_datasets module imported")
    
    from bayesian_lora.samplers.sgld import sgld_step
    print("✓ sgld module imported")
    
    from bayesian_lora.utils.lora_params import get_lora_parameters
    print("✓ lora_params module imported")
    
    print("\n🎉 All imports successful! Module is properly installed.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're in the project root directory")
    print("2. Activate virtual environment: source .venv/bin/activate")
    print("3. Install package: pip3 install -e .")
    print("4. Install requirements: pip3 install -r requirements_lora.txt")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

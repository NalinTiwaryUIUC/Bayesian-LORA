#!/usr/bin/env python3
"""
Direct test of the model class to isolate LoRA parameter creation issue.
Useful for troubleshooting on the cluster.
"""

from bayesian_lora.models.hf_lora import build_huggingface_lora_model

def test_model_direct():
    """Test model creation directly."""
    print("=== Direct Model Test ===\n")
    
    # Test config
    config = {
        'name': 'bert-base-uncased',
        'num_labels': 2,
        'lora': {
            'rank': 8,
            'alpha': 16.0,
            'dropout': 0.1,
            'target_modules': ["query", "value"]  # Explicitly set
        }
    }
    
    print(f"Config: {config}")
    
    try:
        # Create model
        print("\nCreating model...")
        model = build_huggingface_lora_model(config)
        print(f"✓ Model created: {type(model).__name__}")
        
        # Check model attributes
        print(f"\nModel attributes:")
        print(f"  - hasattr(model, 'peft_model'): {hasattr(model, 'peft_model')}")
        print(f"  - hasattr(model, 'base_model'): {hasattr(model, 'base_model')}")
        print(f"  - hasattr(model, 'lora_config'): {hasattr(model, 'lora_config')}")
        
        if hasattr(model, 'peft_model'):
            print(f"  - peft_model type: {type(model.peft_model)}")
            print(f"  - peft_model attributes: {dir(model.peft_model)[:10]}...")
        
        # Check parameters
        print(f"\nParameter check:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Check PEFT model parameters specifically
        if hasattr(model, 'peft_model'):
            print(f"\nPEFT model parameters:")
            peft_total = sum(p.numel() for p in model.peft_model.parameters())
            peft_trainable = sum(p.numel() for p in model.peft_model.parameters() if p.requires_grad)
            print(f"  - PEFT total: {peft_total:,}")
            print(f"  - PEFT trainable: {peft_trainable:,}")
            
            # Check if LoRA layers exist
            lora_layers = []
            for name, module in model.peft_model.named_modules():
                if 'lora' in name.lower():
                    lora_layers.append(name)
            
            print(f"  - LoRA layers found: {len(lora_layers)}")
            if lora_layers:
                print(f"    First few: {lora_layers[:3]}")
        
        # Try to get LoRA parameters
        print(f"\nTrying to get LoRA parameters...")
        try:
            lora_params = model.get_lora_parameters()
            print(f"  - get_lora_parameters() returned: {len(lora_params)} parameters")
            if lora_params:
                total_lora = sum(p.numel() for p in lora_params)
                print(f"  - Total LoRA parameters: {total_lora:,}")
            else:
                print(f"  - ⚠️  No LoRA parameters returned")
        except Exception as e:
            print(f"  - ❌ get_lora_parameters() failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_direct()

#!/usr/bin/env python3
"""
Debug script to investigate LoRA parameter creation.
Useful for troubleshooting on the cluster.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

def debug_bert_modules():
    """Debug BERT model structure to find target modules."""
    print("=== Debugging BERT Model Structure ===")
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    print(f"Model type: {type(model).__name__}")
    
    # Check all module names
    print("\n=== All Module Names ===")
    for name, module in model.named_modules():
        if "attention" in name.lower() or "query" in name.lower() or "value" in name.lower():
            print(f"  {name}: {type(module).__name__}")
    
    # Check specific attention modules
    print("\n=== Attention Modules ===")
    for name, module in model.named_modules():
        if "attention.self" in name:
            print(f"  {name}: {type(module).__name__}")
            if hasattr(module, 'query'):
                print(f"    - query: {type(module.query).__name__}")
            if hasattr(module, 'key'):
                print(f"    - key: {type(module.key).__name__}")
            if hasattr(module, 'value'):
                print(f"    - value: {type(module.value).__name__}")
    
    # Try to create LoRA config
    print("\n=== Testing LoRA Config ===")
    try:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16.0,
            lora_dropout=0.1,
            target_modules=["query", "value"]
        )
        print(f"✓ LoRA config created: {lora_config}")
        
        # Try to apply LoRA
        peft_model = get_peft_model(model, lora_config)
        print(f"✓ PEFT model created")
        
        # Check parameters
        total_params = sum(p.numel() for p in peft_model.parameters())
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  LoRA ratio: {trainable_params/total_params*100:.2f}%")
        
        if trainable_params == 0:
            print("❌ No trainable parameters found!")
            print("This suggests the target modules weren't found.")
            
            # Try different target module names
            print("\n=== Trying Different Target Module Names ===")
            alternative_targets = [
                ["query", "key", "value"],
                ["self_attn.query", "self_attn.value"],
                ["attention.self.query", "attention.self.value"],
                ["attention.self"]
            ]
            
            for targets in alternative_targets:
                try:
                    alt_config = LoraConfig(
                        task_type=TaskType.SEQ_CLS,
                        r=8,
                        lora_alpha=16.0,
                        lora_dropout=0.1,
                        target_modules=targets
                    )
                    alt_model = get_peft_model(model, alt_config)
                    alt_trainable = sum(p.numel() for p in alt_model.parameters() if p.requires_grad)
                    print(f"  {targets}: {alt_trainable:,} trainable parameters")
                except Exception as e:
                    print(f"  {targets}: Failed - {e}")
        
    except Exception as e:
        print(f"❌ LoRA config failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_bert_modules()

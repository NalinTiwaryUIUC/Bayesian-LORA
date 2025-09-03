#!/usr/bin/env python
"""
HuggingFace LoRA models for Bayesian sampling experiments.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Any, Tuple, Optional

class LoRAModel(nn.Module):
    """
    Generic LoRA model wrapper for HuggingFace models.
    This is the main class expected by the training scripts.
    """
    
    def __init__(self, base_model, r: int = 8, alpha: float = 16.0, 
                 dropout: float = 0.05, target_modules: list = None):
        super().__init__()
        
        self.base_model = base_model
        
        # Set default target modules if not provided
        if target_modules is None:
            target_modules = ["q", "k", "v", "o"]
        
        # Create LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules
        )
        
        # Apply LoRA to the base model
        self.model = get_peft_model(base_model, self.lora_config)
        
        # Freeze base model parameters
        self._freeze_base_model()
        
        # Print parameter information
        self._print_parameter_info()
    
    def _freeze_base_model(self):
        """Freeze all base model parameters, keeping only LoRA parameters trainable."""
        # Freeze base model parameters
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False
        
        # Ensure LoRA parameters remain trainable
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
    
    def _print_parameter_info(self):
        """Print information about trainable vs. frozen parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable (LoRA) parameters: {trainable_params:,}")
        print(f"Frozen (base) parameters: {frozen_params:,}")
        print(f"LoRA ratio: {trainable_params/total_params*100:.2f}%")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the LoRA model."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def get_lora_parameters(self):
        """Get only the LoRA parameters for sampling."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def get_base_parameters(self):
        """Get only the base model parameters (frozen)."""
        return [p for p in self.model.parameters() if not p.requires_grad]
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Create LoRA model from pretrained model name."""
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        return cls(base_model, **kwargs)



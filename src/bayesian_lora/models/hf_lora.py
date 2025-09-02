#!/usr/bin/env python
"""
HuggingFace LoRA models for Bayesian sampling experiments.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Any, Tuple, Optional

class HuggingFaceLoRAModel(nn.Module):
    """
    HuggingFace model with LoRA adaptation for Bayesian sampling.
    
    This class wraps a pre-trained HuggingFace model and applies LoRA
    to enable parameter-efficient fine-tuning while keeping the base
    model frozen for Bayesian sampling of only the LoRA parameters.
    """
    
    def __init__(self, model_name: str, num_labels: int, lora_config: Dict[str, Any]):
        super().__init__()
        
        # Load pre-trained model and tokenizer
        self.model_name = model_name
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            return_dict=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Apply LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_config.get('rank', 8),
            lora_alpha=lora_config.get('alpha', 16.0),
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', None)
        )
        
        # Create PEFT model with LoRA
        self.peft_model = get_peft_model(self.base_model, self.lora_config)
        
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
        for name, param in self.peft_model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
    
    def _print_parameter_info(self):
        """Print information about trainable vs. frozen parameters."""
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Model: {self.model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable (LoRA) parameters: {trainable_params:,}")
        print(f"Frozen (base) parameters: {frozen_params:,}")
        print(f"LoRA ratio: {trainable_params/total_params*100:.2f}%")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the LoRA model."""
        return self.peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def get_lora_parameters(self):
        """Get only the LoRA parameters for sampling."""
        return [p for p in self.peft_model.parameters() if p.requires_grad]
    
    def get_base_parameters(self):
        """Get only the base model parameters (frozen)."""
        return [p for p in self.peft_model.parameters() if not p.requires_grad]
    
    def get_tokenizer(self):
        """Get the tokenizer for text preprocessing."""
        return self.tokenizer

class BERTLoRAModel(HuggingFaceLoRAModel):
    """BERT model with LoRA adaptation."""
    
    def __init__(self, num_labels: int, lora_config: Dict[str, Any]):
        # Set BERT-specific LoRA target modules before calling parent
        if lora_config.get('target_modules') is None:
            lora_config = lora_config.copy()
            lora_config['target_modules'] = ["query", "value"]
        
        super().__init__("bert-base-uncased", num_labels, lora_config)

class RoBERTaLoRAModel(HuggingFaceLoRAModel):
    """RoBERTa model with LoRA adaptation."""
    
    def __init__(self, num_labels: int, lora_config: Dict[str, Any]):
        # Set RoBERTa-specific LoRA target modules before calling parent
        if lora_config.get('target_modules') is None:
            lora_config = lora_config.copy()
            lora_config['target_modules'] = ["query", "value"]
        
        super().__init__("roberta-base", num_labels, lora_config)

class DistilBERTLoRAModel(HuggingFaceLoRAModel):
    """DistilBERT model with LoRA adaptation."""
    
    def __init__(self, num_labels: int, lora_config: Dict[str, Any]):
        # Set DistilBERT-specific LoRA target modules before calling parent
        if lora_config.get('target_modules') is None:
            lora_config = lora_config.copy()
            lora_config['target_modules'] = ["q_lin", "v_lin"]
        
        super().__init__("distilbert-base-uncased", num_labels, lora_config)

def build_huggingface_lora_model(model_config: Dict[str, Any]) -> HuggingFaceLoRAModel:
    """
    Factory function to build the appropriate HuggingFace LoRA model.
    
    Args:
        model_config: Configuration dictionary containing model settings
        
    Returns:
        Configured HuggingFace LoRA model
    """
    model_name = model_config['name']
    num_labels = model_config['num_labels']
    lora_config = model_config['lora']
    
    if 'bert' in model_name.lower():
        return BERTLoRAModel(num_labels, lora_config)
    elif 'roberta' in model_name.lower():
        return RoBERTaLoRAModel(num_labels, lora_config)
    elif 'distilbert' in model_name.lower():
        return DistilBERTLoRAModel(num_labels, lora_config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

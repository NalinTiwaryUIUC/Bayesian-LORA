import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any


class LoRAParams:
    """
    Utility class for managing LoRA parameters.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get LoRA parameters from the model."""
        return get_lora_parameters(self.model)
    
    def get_base_parameters(self) -> List[nn.Parameter]:
        """Get base model parameters."""
        return get_base_parameters(self.model)
    
    def count_lora_parameters(self) -> int:
        """Count total LoRA parameters."""
        return count_lora_parameters(self.model)
    
    def count_base_parameters(self) -> int:
        """Count total base parameters."""
        return count_base_parameters(self.model)


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Extract only LoRA parameters from a model.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        List of LoRA parameters (A and B matrices)
    """
    lora_params = []
    for module in model.modules():
        if hasattr(module, 'get_lora_parameters'):
            lora_params.extend(module.get_lora_parameters())
    return lora_params


def get_base_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Extract base model parameters (non-LoRA).
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        List of base model parameters
    """
    base_params = []
    for module in model.modules():
        if hasattr(module, 'get_base_parameters'):
            base_params.extend(module.get_base_parameters())
    return base_params








def count_lora_parameters(model: nn.Module) -> int:
    """
    Count total number of LoRA parameters.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Total number of LoRA parameters
    """
    lora_params = get_lora_parameters(model)
    return sum(param.numel() for param in lora_params)


def count_base_parameters(model: nn.Module) -> int:
    """
    Count total number of base model parameters.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Total number of base model parameters
    """
    base_params = get_base_parameters(model)
    return sum(param.numel() for param in base_params)





def freeze_base_model(model: nn.Module) -> None:
    """
    Freeze all base model parameters.
    
    Args:
        model: Model with LoRA layers
    """
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters
    for module in model.modules():
        if hasattr(module, 'lora_A'):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
















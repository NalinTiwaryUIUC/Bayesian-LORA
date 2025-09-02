import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any


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


def flatten_lora_params(model: nn.Module) -> torch.Tensor:
    """
    Flatten LoRA parameters into a single vector.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Flattened LoRA parameters as 1D tensor
    """
    lora_params = get_lora_parameters(model)
    if not lora_params:
        raise ValueError("No LoRA parameters found in model")
    
    return torch.cat([param.flatten() for param in lora_params])


def unflatten_lora_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """
    Unflatten parameters back into LoRA layers.
    
    Args:
        model: Model with LoRA layers
        flat_params: Flattened parameter vector
    """
    lora_params = get_lora_parameters(model)
    if not lora_params:
        raise ValueError("No LoRA parameters found in model")
    
    start_idx = 0
    for param in lora_params:
        param_size = param.numel()
        param.data = flat_params[start_idx:start_idx + param_size].reshape(param.shape)
        start_idx += param_size


def flatten_lora_grads(model: nn.Module) -> torch.Tensor:
    """
    Flatten LoRA gradients into a single vector.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Flattened LoRA gradients as 1D tensor
    """
    lora_params = get_lora_parameters(model)
    if not lora_params:
        raise ValueError("No LoRA parameters found in model")
    
    grads = []
    for param in lora_params:
        if param.grad is not None:
            grads.append(param.grad.flatten())
        else:
            grads.append(torch.zeros_like(param).flatten())
    
    return torch.cat(grads)


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


def get_lora_parameter_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get detailed information about LoRA parameters.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Dictionary with LoRA parameter information
    """
    lora_params = get_lora_parameters(model)
    base_params = get_base_parameters(model)
    
    info = {
        'lora_param_count': count_lora_parameters(model),
        'base_param_count': count_base_parameters(model),
        'total_param_count': count_lora_parameters(model) + count_base_parameters(model),
        'lora_ratio': count_lora_parameters(model) / (count_lora_parameters(model) + count_base_parameters(model)),
        'lora_layers': []
    }
    
    # Analyze LoRA layers
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            layer_info = {
                'name': name,
                'rank': module.rank,
                'alpha': module.alpha,
                'in_features': module.lora_A.shape[0],
                'out_features': module.lora_B.shape[1],
                'param_count': module.lora_A.numel() + module.lora_B.numel()
            }
            info['lora_layers'].append(layer_info)
    
    return info


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


def unfreeze_base_model(model: nn.Module) -> None:
    """
    Unfreeze all base model parameters.
    
    Args:
        model: Model with LoRA layers
    """
    for param in model.parameters():
        param.requires_grad = True


def reset_lora_parameters(model: nn.Module) -> None:
    """
    Reset LoRA parameters to initial values.
    
    Args:
        model: Model with LoRA layers
    """
    for module in model.modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Reset A matrix
            nn.init.normal_(module.lora_A, mean=0.0, std=0.02)
            # Reset B matrix to zeros
            nn.init.zeros_(module.lora_B)


def merge_lora_weights(model: nn.Module) -> None:
    """
    Merge LoRA weights into base model weights.
    This permanently incorporates LoRA adaptations.
    
    Args:
        model: Model with LoRA layers
    """
    for module in model.modules():
        if hasattr(module, 'linear') and hasattr(module, 'lora'):
            # For LoRALinear layers
            with torch.no_grad():
                lora_weight = module.lora.scaling * (module.lora.lora_B @ module.lora.lora_A)
                module.linear.weight.data += lora_weight.T


def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA parameters as a state dict.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        State dict containing only LoRA parameters
    """
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            state_dict[f"{name}.lora_A"] = module.lora_A.data.clone()
            state_dict[f"{name}.lora_B"] = module.lora_B.data.clone()
    
    return state_dict


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Load LoRA parameters from a state dict.
    
    Args:
        model: Model with LoRA layers
        state_dict: State dict containing LoRA parameters
    """
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            
            if a_key in state_dict and b_key in state_dict:
                module.lora_A.data = state_dict[a_key]
                module.lora_B.data = state_dict[b_key]

"""
Bayesian LoRA: Bayesian inference for Low-Rank Adaptation of Large Language Models.
"""

__version__ = "0.1.0"
__author__ = "Bayesian LoRA Team"

# Define what should be available
__all__ = [
    "build_huggingface_lora_model",
    "create_dataloaders", 
    "get_dataset_metadata",
    "sgld_step",
    "asgld_step", 
    "sam_sgld_step",
    "sam_sgld_rank_1_step",
    "get_lora_parameters",
    "count_lora_parameters"
]

# Lazy imports to avoid circular import issues
def _import_models():
    """Import models module."""
    try:
        from .models.hf_lora import build_huggingface_lora_model
        return build_huggingface_lora_model
    except ImportError as e:
        print(f"Warning: Could not import models: {e}")
        return None

def _import_data():
    """Import data module."""
    try:
        from .data.glue_datasets import create_dataloaders, get_dataset_metadata
        return create_dataloaders, get_dataset_metadata
    except ImportError as e:
        print(f"Warning: Could not import data: {e}")
        return None, None

def _import_samplers():
    """Import samplers module."""
    try:
        from .samplers.sgld import sgld_step, asgld_step, sam_sgld_step, sam_sgld_rank_1_step
        return sgld_step, asgld_step, sam_sgld_step, sam_sgld_rank_1_step
    except ImportError as e:
        print(f"Warning: Could not import samplers: {e}")
        return None, None, None, None

def _import_utils():
    """Import utils module."""
    try:
        from .utils.lora_params import get_lora_parameters, count_lora_parameters
        return get_lora_parameters, count_lora_parameters
    except ImportError as e:
        print(f"Warning: Could not import utils: {e}")
        return None, None

# Create lazy import functions
def build_huggingface_lora_model(*args, **kwargs):
    """Lazy import for build_huggingface_lora_model."""
    func = _import_models()
    if func is None:
        raise ImportError("Models module not available")
    return func(*args, **kwargs)

def create_dataloaders(*args, **kwargs):
    """Lazy import for create_dataloaders."""
    func, _ = _import_data()
    if func is None:
        raise ImportError("Data module not available")
    return func(*args, **kwargs)

def get_dataset_metadata(*args, **kwargs):
    """Lazy import for get_dataset_metadata."""
    _, func = _import_data()
    if func is None:
        raise ImportError("Data module not available")
    return func(*args, **kwargs)

def sgld_step(*args, **kwargs):
    """Lazy import for sgld_step."""
    func, _, _, _ = _import_samplers()
    if func is None:
        raise ImportError("Samplers module not available")
    return func(*args, **kwargs)

def asgld_step(*args, **kwargs):
    """Lazy import for asgld_step."""
    _, func, _, _ = _import_samplers()
    if func is None:
        raise ImportError("Samplers module not available")
    return func(*args, **kwargs)

def sam_sgld_step(*args, **kwargs):
    """Lazy import for sam_sgld_step."""
    _, _, func, _ = _import_samplers()
    if func is None:
        raise ImportError("Samplers module not available")
    return func(*args, **kwargs)

def sam_sgld_rank_1_step(*args, **kwargs):
    """Lazy import for sam_sgld_rank_1_step."""
    _, _, _, func = _import_samplers()
    if func is None:
        raise ImportError("Samplers module not available")
    return func(*args, **kwargs)

def get_lora_parameters(*args, **kwargs):
    """Lazy import for get_lora_parameters."""
    func, _ = _import_utils()
    if func is None:
        raise ImportError("Utils module not available")
    return func(*args, **kwargs)

def count_lora_parameters(*args, **kwargs):
    """Lazy import for count_lora_parameters."""
    _, func = _import_utils()
    if func is None:
        raise ImportError("Utils module not available")
    return func(*args, **kwargs)

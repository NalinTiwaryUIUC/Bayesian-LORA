"""
Bayesian LoRA: Bayesian inference for Low-Rank Adaptation of Large Language Models.
"""

__version__ = "0.1.0"
__author__ = "Bayesian LoRA Team"

# Import main components
from .models.hf_lora import build_huggingface_lora_model
from .data.glue_datasets import create_dataloaders, get_dataset_metadata
from .samplers.sgld import sgld_step, asgld_step, sam_sgld_step, sam_sgld_rank_1_step
from .utils.lora_params import get_lora_parameters, count_lora_parameters

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
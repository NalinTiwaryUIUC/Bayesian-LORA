"""
Data loading and processing modules for Bayesian LoRA.

This module provides data loaders for various datasets including:
- GLUE benchmark datasets (SST-2, MRPC, etc.)
- CIFAR datasets
- Custom dataset utilities
"""

from .glue_datasets import create_dataloaders, get_dataset_metadata
from .cifar import get_cifar_loaders

__all__ = [
    'create_dataloaders',
    'get_dataset_metadata', 
    'get_cifar_loaders'
]

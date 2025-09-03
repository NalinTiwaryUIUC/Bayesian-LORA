#!/usr/bin/env python
"""
GLUE benchmark and standard NLP datasets for LoRA experiments.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# For evaluation metrics, we'll use the datasets library's built-in metrics
# or fall back to sklearn for basic metrics

class MRPCDataset(Dataset):
    """
    MRPC (Microsoft Research Paraphrase Corpus) dataset for LoRA experiments.
    """
    
    def __init__(self, split: str, tokenizer, max_length: int = 256):
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset from HuggingFace
        self.dataset = load_dataset("glue", "mrpc", split=split)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # MRPC has sentence1 and sentence2
        text = f"{item['sentence1']} [SEP] {item['sentence2']}"
        label = item['label']
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }






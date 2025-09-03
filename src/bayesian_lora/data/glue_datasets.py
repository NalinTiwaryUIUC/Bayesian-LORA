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

class GLUEBenchmarkDataset(Dataset):
    """
    GLUE benchmark dataset wrapper for sequence classification tasks.
    
    Supports all GLUE tasks: SST-2, MRPC, QNLI, RTE, CoLA, STS-B
    with proper tokenization and formatting for HuggingFace models.
    """
    
    def __init__(self, dataset_name: str, split: str, tokenizer, max_length: int = 128):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset from HuggingFace
        self.dataset = load_dataset("glue", dataset_name, split=split)
        
        # Get task-specific information
        self.task_info = self._get_task_info(dataset_name)
        
    def _get_task_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get task-specific information and configuration."""
        task_configs = {
            'sst2': {
                'num_labels': 2,
                'task_type': 'sentiment_analysis',
                'description': 'Stanford Sentiment Treebank v2'
            },
            'mrpc': {
                'num_labels': 2,
                'task_type': 'paraphrase_detection',
                'description': 'Microsoft Research Paraphrase Corpus'
            },
            'qnli': {
                'num_labels': 2,
                'task_type': 'question_answering_nli',
                'description': 'Question-answering Natural Language Inference'
            },
            'rte': {
                'num_labels': 2,
                'task_type': 'textual_entailment',
                'description': 'Recognizing Textual Entailment'
            },
            'cola': {
                'num_labels': 2,
                'task_type': 'linguistic_acceptability',
                'description': 'Corpus of Linguistic Acceptability'
            },
            'sts-b': {
                'num_labels': 1,  # Regression task
                'task_type': 'semantic_textual_similarity',
                'description': 'Semantic Textual Similarity Benchmark'
            }
        }
        return task_configs.get(dataset_name, {})
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle different GLUE task formats
        if self.dataset_name == 'sst2':
            text = item['sentence']
            label = item['label']
        elif self.dataset_name in ['mrpc', 'qnli', 'rte']:
            text = f"{item['sentence1']} [SEP] {item['sentence2']}"
            label = item['label']
        elif self.dataset_name == 'cola':
            text = item['sentence']
            label = item['label']
        elif self.dataset_name == 'sts-b':
            text = f"{item['sentence1']} [SEP] {item['sentence2']}"
            label = float(item['label'])  # Regression task
        else:
            raise ValueError(f"Unsupported GLUE task: {self.dataset_name}")
        
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
            'labels': torch.tensor(label, dtype=torch.long if self.dataset_name != 'sts-b' else torch.float)
        }

class IMDBMovieReviewsDataset(Dataset):
    """
    IMDB movie reviews dataset for sentiment analysis.
    
    Large-scale dataset with 50K movie reviews for binary sentiment classification.
    """
    
    def __init__(self, split: str, tokenizer, max_length: int = 256):
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load IMDB dataset
        self.dataset = load_dataset("imdb", split=split)
        
        # Task information
        self.task_info = {
            'num_labels': 2,
            'task_type': 'sentiment_analysis',
            'description': 'IMDB Movie Reviews'
        }
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

class AGNewsTopicDataset(Dataset):
    """
    AG News topic classification dataset.
    
    Large-scale dataset with 120K news articles for 4-class topic classification.
    """
    
    def __init__(self, split: str, tokenizer, max_length: int = 128):
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load AG News dataset
        self.dataset = load_dataset("ag_news", split=split)
        
        # Task information
        self.task_info = {
            'num_labels': 4,
            'task_type': 'topic_classification',
            'description': 'AG News Topic Classification'
        }
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

def get_dataset_metadata(dataset_name: str) -> Dict[str, Any]:
    """
    Get comprehensive metadata for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing dataset metadata
    """
    metadata = {
        'sst2': {
            'name': 'Stanford Sentiment Treebank v2',
            'task': 'sentiment_analysis',
            'num_labels': 2,
            'train_size': 67349,
            'val_size': 872,
            'test_size': 1821,
            'max_length': 128,
            'description': 'Binary sentiment classification on movie reviews'
        },
        'mrpc': {
            'name': 'Microsoft Research Paraphrase Corpus',
            'task': 'paraphrase_detection',
            'num_labels': 2,
            'train_size': 3668,
            'val_size': 408,
            'test_size': 1725,
            'max_length': 128,
            'description': 'Binary classification of sentence pairs as paraphrases'
        },
        'qnli': {
            'name': 'Question-answering NLI',
            'task': 'question_answering_nli',
            'num_labels': 2,
            'train_size': 104743,
            'val_size': 5463,
            'test_size': 5463,
            'max_length': 128,
            'description': 'Binary classification of question-answer pairs'
        },
        'rte': {
            'name': 'Recognizing Textual Entailment',
            'task': 'textual_entailment',
            'num_labels': 2,
            'train_size': 2490,
            'val_size': 277,
            'test_size': 3000,
            'max_length': 128,
            'description': 'Binary classification of sentence pairs for entailment'
        },
        'imdb': {
            'name': 'IMDB Movie Reviews',
            'task': 'sentiment_analysis',
            'num_labels': 2,
            'train_size': 25000,
            'test_size': 25000,
            'max_length': 256,
            'description': 'Binary sentiment classification on movie reviews'
        },
        'ag_news': {
            'name': 'AG News Topic Classification',
            'task': 'topic_classification',
            'num_labels': 4,
            'train_size': 120000,
            'test_size': 7600,
            'max_length': 128,
            'description': '4-class topic classification on news articles'
        }
    }
    
    return metadata.get(dataset_name, {})

def create_dataset_instance(dataset_name: str, split: str, tokenizer, max_length: int = None) -> Dataset:
    """
    Factory function to create dataset instances.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split ('train', 'validation', 'test')
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Configured dataset instance
    """
    # Get default max_length if not specified
    if max_length is None:
        metadata = get_dataset_metadata(dataset_name)
        max_length = metadata.get('max_length', 128)
    
    # Create appropriate dataset
    if dataset_name in ['sst2', 'mrpc', 'qnli', 'rte', 'cola', 'sts-b']:
        return GLUEBenchmarkDataset(dataset_name, split, tokenizer, max_length)
    elif dataset_name == 'imdb':
        return IMDBMovieReviewsDataset(split, tokenizer, max_length)
    elif dataset_name == 'ag_news':
        return AGNewsTopicDataset(split, tokenizer, max_length)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def create_dataloaders(dataset_name: str, tokenizer, batch_size: int = 32, max_length: int = None, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Handle different split names for different datasets
    if dataset_name in ['sst2', 'mrpc', 'qnli', 'rte', 'cola', 'sts-b']:
        train_split = 'train'
        val_split = 'validation'
    else:
        train_split = 'train'
        val_split = 'test'  # Some datasets don't have separate validation
    
    # Create datasets
    train_dataset = create_dataset_instance(dataset_name, train_split, tokenizer, max_length)
    val_dataset = create_dataset_instance(dataset_name, val_split, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def evaluate_predictions(dataset_name: str, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate predictions using task-specific metrics.
    
    Args:
        dataset_name: Name of the dataset
        predictions: Model predictions
        labels: Ground truth labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    # For now, we'll use basic accuracy for all tasks
    # In a production setting, you might want to use more sophisticated metrics
    
    if dataset_name == 'sts-b':
        # Regression task - use correlation-like metrics
        # For simplicity, we'll use accuracy with a threshold
        correct = np.abs(predictions - labels) < 0.5
        accuracy = correct.mean()
        return {'accuracy': accuracy}
    else:
        # Classification tasks - use accuracy
        correct = predictions == labels
        accuracy = correct.mean()
        return {'accuracy': accuracy}

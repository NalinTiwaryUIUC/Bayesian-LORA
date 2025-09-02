#!/usr/bin/env python3
"""
Setup script for Bayesian-LORA package.
"""

from setuptools import setup, find_packages

setup(
    name="bayesian-lora",
    version="0.1.0",
    description="Bayesian inference for Low-Rank Adaptation of Large Language Models",
    author="Bayesian LoRA Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "tqdm",
        "pyyaml",
        "scikit-learn",
        "matplotlib",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tokenizers>=0.13.0",
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "sentencepiece",
        "protobuf",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "ruff",
        ],
    },
)

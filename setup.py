#!/usr/bin/env python3
"""
Setup script for Bayesian-LORA package.
"""

from setuptools import setup, find_packages

setup(
    name="bayesian-lora",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    # Dependencies are now managed in pyproject.toml
)

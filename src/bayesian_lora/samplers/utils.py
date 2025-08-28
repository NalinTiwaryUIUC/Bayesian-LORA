# src/bayesian_lora/samplers/utils.py
import math
import torch

def cosine_annealed_eps(t: int, t_max: int, eps_min: float, eps_max: float) -> float:
    """
    Cosine-annealed step-size in [eps_min, eps_max] over [0, t_max].
    """
    if t_max <= 0:
        return eps_min
    return eps_min + 0.5 * (eps_max - eps_min) * (1 + math.cos(math.pi * (t % t_max) / t_max))
# src/bayesian_lora/utils/params.py
import torch

def flatten_params(model) -> torch.Tensor:
    """
    Flatten model parameters (requires_grad=True) into one 1D tensor.
    Order = model.parameters() order; unflatten_params assumes the same order.
    """
    out = []
    for p in model.parameters():
        if p.requires_grad:
            out.append(p.detach().reshape(-1))
    return torch.cat(out) if out else torch.empty(0)

@torch.no_grad()
def unflatten_params(model, vector: torch.Tensor) -> None:
    """
    Copy a flat parameter vector back into model parameters in-place.
    Assumes the same order as flatten_params.
    """
    pointer = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        numel = p.numel()
        p.view(-1).copy_(vector[pointer:pointer+numel].to(p.device, dtype=p.dtype))
        pointer += numel
    if pointer != vector.numel():
        raise ValueError(f"Vector length {vector.numel()} ≠ model param size {pointer}")

def flatten_grads(model) -> torch.Tensor:
    """
    Flatten current gradients into a 1D tensor (None → zeros).
    """
    chunks = []
    for p in model.parameters():
        if p.requires_grad:
            g = p.grad
            chunks.append((torch.zeros_like(p) if g is None else g.detach()).reshape(-1))
    return torch.cat(chunks) if chunks else torch.empty(0)
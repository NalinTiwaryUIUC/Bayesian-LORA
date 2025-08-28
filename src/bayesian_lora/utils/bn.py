# src/bayesian_lora/utils/bn.py
import torch
import torch.nn as nn

def _iter_bn_buffers(model):
    """
    Deterministic order of BN buffers: running_mean, running_var, num_batches_tracked.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if getattr(m, "running_mean", None) is not None:
                yield m, "running_mean", m.running_mean
            if getattr(m, "running_var", None) is not None:
                yield m, "running_var", m.running_var
            if getattr(m, "num_batches_tracked", None) is not None:
                yield m, "num_batches_tracked", m.num_batches_tracked

@torch.no_grad()
def get_bn_buffers(model):
    """
    Return a list of BN buffers (cloned) in fixed order for later restoration.
    """
    out = []
    for _, _, t in _iter_bn_buffers(model):
        out.append(t.detach().clone())
    return out

@torch.no_grad()
def load_bn_buffers(model, buffers):
    """
    Load BN buffers (from get_bn_buffers) back into the model.
    """
    i = 0
    for _, _, t in _iter_bn_buffers(model):
        src = buffers[i].to(device=t.device, dtype=t.dtype)
        t.copy_(src)
        i += 1
    if i != len(buffers):
        raise ValueError("BN buffer count mismatch in load_bn_buffers.")
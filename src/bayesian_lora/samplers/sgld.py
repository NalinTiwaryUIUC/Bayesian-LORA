# src/bayesian_lora/samplers/sgld.py
import math
import torch

@torch.no_grad()
def sgld_step(x, grad, eps, tau, debug=False, step=None):
    """
    Single SGLD update on a flat parameter vector x.

    x   : flat params tensor
    grad: gradient tensor (same shape as x)
    eps : step size
    tau : temperature (often 1.0)
    """
    noise_std = math.sqrt(2 * tau * eps)
    noise = torch.randn_like(x) * noise_std
    x = x - eps * grad + noise

    if debug and step is not None and step % 50 == 0:
        print(f"[SGLD {step}] ‖grad‖={grad.norm().item():.4f}, ‖x‖={x.norm().item():.4f}")
    return x

@torch.no_grad()
def asgld_step(x, m, v, grad, eps, beta1, beta2, a, lambd, tau, step=None, debug=False):
    # Update moment estimates
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad * grad)
    # Adaptive drift
    A = m / (v.sqrt() + lambd)
    # Noise
    noise_std = math.sqrt(2 * tau * eps)
    noise = torch.randn_like(x) * noise_std
    # Update
    x = x - eps * (grad + a * A) + noise

    if debug and step is not None and step % 50 == 0:
        print(f"[ASGLD {step}] ‖grad‖={grad.norm().item():.4f}, "
              f"‖m‖={m.norm().item():.4f}, ‖v‖={v.norm().item():.4f}, "
              f"‖A‖={A.norm().item():.4f}, eps={eps:.3e}")
    return x, m, v

@torch.no_grad()
def sam_sgld_step(x, grad_fn, eps, tau, rho, lambd, debug=False, step=None):
    """
    Single SAM-SGLD update, assuming grad_fn returns ∇_x log p(x|D) (i.e., data + prior if you want).
    """
    noise_std = math.sqrt(2 * tau * eps)
    noise = torch.randn_like(x) * noise_std

    grad = grad_fn(x)
    delta = rho * grad / (grad.norm() + lambd)
    grad_sam = grad_fn(x + delta)

    x_new = x - eps * grad_sam + noise

    if debug and step is not None and step % 50 == 0:
        print(f"[SAM-SGLD {step}] ‖g‖={grad.norm().item():.4f}, "
              f"‖g_sam‖={grad_sam.norm().item():.4f}, ‖noise‖={noise.norm().item():.4f}")
    return x_new

def sample_rank1_noise(u, base_std, sigma_dir=1.0):
    z = torch.randn_like(u)
    u_hat = u / (u.norm() + 1e-8)
    z_proj = torch.dot(z, u_hat)
    return base_std * (z + sigma_dir * z_proj * u_hat)

@torch.no_grad()
def sam_sgld_rank_1_step(x, data_grad_fn, prior_prec, eps, tau, rho, lambd, sigma_dir=1.0, debug=False):
    """
    SAM-SGLD with rank-1 directional noise. Adds Gaussian prior via prior_prec * x.
    """
    grad = data_grad_fn(x)
    grad_norm = grad.norm() + lambd

    delta = rho * grad / grad_norm
    grad_sam = data_grad_fn(x + delta)

    # Add prior after data gradient
    grad_sam = grad_sam + prior_prec * x

    noise_std = math.sqrt(2 * tau * eps)
    noise = sample_rank1_noise(grad_sam.detach(), noise_std, sigma_dir)
    x_new = x - eps * grad_sam + noise

    if debug:
        print(f"[SAM-SGLD-R1] ‖g‖={grad.norm().item():.4f}, ‖g_sam‖={grad_sam.norm().item():.4f}, "
              f"‖noise‖={noise.norm().item():.4f}, ‖x‖={x_new.norm().item():.4f}")
    return x_new
#!/usr/bin/env python
# scripts/train.py
import argparse, json, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from bayesian_lora.data.cifar import get_cifar_loaders
from bayesian_lora.models.resnet_cifar import ResNet18CIFAR, ResNet34CIFAR
from bayesian_lora.models.wide_resnet import WRN_28_10_CIFAR

# --- Use *repo* utilities (no re-implementations) ---
from bayesian_lora.utils.params import (
    flatten_params,
    unflatten_params,
    flatten_grads,
)
from bayesian_lora.utils.bn import get_bn_buffers
from bayesian_lora.samplers.utils import cosine_annealed_eps
from bayesian_lora.samplers.sgld import (
    sgld_step,
    asgld_step,
    sam_sgld_step,
    sam_sgld_rank_1_step,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_model(cfg_model, num_classes: int):
    name = cfg_model["name"].lower()
    if name in {"resnet18_cifar", "resnet18"}:
        return ResNet18CIFAR(num_classes=num_classes)
    if name in {"resnet34_cifar", "resnet34"}:
        return ResNet34CIFAR(num_classes=num_classes)
    if name in {"wrn_28_10_cifar", "wrn", "wideresnet"}:
        return WRN_28_10_CIFAR(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {cfg_model['name']}")


def make_grad_fn(model, criterion, batch, device, prior_prec):
    """
    Returns a closure grad_fn(x_flat) -> grad_flat for the *current* batch.
    Uses repo utils: unflatten_params, flatten_grads.
    Adds Gaussian prior gradient: prior_prec * x (weight-decay prior).
    """
    inputs, targets = batch

    def grad_fn(x_flat: torch.Tensor) -> torch.Tensor:
        # load params from flat vector
        unflatten_params(model, x_flat)
        model.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()

        g = flatten_grads(model)
        if prior_prec != 0.0:
            g = g + prior_prec * x_flat
        return g

    return grad_fn


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 123))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Data ----------------
    train_loader, test_loader, num_classes = get_cifar_loaders(
        name=cfg["data"]["name"],
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 4),
        augment=cfg["data"].get("augment", True),
    )

    # ---------------- Model ----------------
    model = build_model(cfg["model"], num_classes=num_classes).to(device)

    # ---------------- (Optional) Pretrain / Finetune ----------------
    epochs = cfg["train"].get("epochs", 0)
    if epochs > 0:
        opt = optim.SGD(
            model.parameters(),
            lr=cfg["train"].get("lr", 0.1),
            momentum=0.9,
            weight_decay=cfg["train"].get("weight_decay", 5e-4),
        )
        criterion = nn.CrossEntropyLoss()
        for ep in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()

    # ---------------- Sampling config ----------------
    sampler = cfg["sampler"]["name"].lower()
    eps = float(cfg["sampler"].get("step_size", 1e-3))  # base step size
    tau = float(cfg["sampler"].get("tau", 1.0))         # temperature
    burn_in = int(cfg["sampler"].get("burn_in", 0))
    thin = int(cfg["sampler"].get("thin", 1))
    num_samples = int(cfg["sampler"].get("num_samples", 0))
    prior_prec = float(cfg["sampler"].get("prior_prec", 0.0))  # gradient of log prior ~ prior_prec * x

    # ASGLD params (only used if sampler == "asgld")
    beta1 = float(cfg["sampler"].get("beta1", 0.9))
    beta2 = float(cfg["sampler"].get("beta2", 0.999))
    a = float(cfg["sampler"].get("a", 1.0))
    lambd = float(cfg["sampler"].get("lambd", 1e-8))

    # SAM params (used for samplers starting with "sam")
    rho = float(cfg["sampler"].get("rho", 0.05))
    sigma_dir = float(cfg["sampler"].get("sigma_dir", 0.0))  # for rank-1 variant
    # optional cosine schedule
    use_cosine = bool(cfg["sampler"].get("cosine_eps", False))
    t_max = int(cfg["sampler"].get("t_max", 0))  # period for cosine annealing

    out_dir = Path(cfg["out"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Sampling loop (flat vector space) ----------------
    criterion = nn.CrossEntropyLoss()
    x_flat = flatten_params(model).to(device)
    # ASGLD state
    m = torch.zeros_like(x_flat)
    v = torch.zeros_like(x_flat)

    it = 0
    saved = 0
    model.train()  # we mutate BN stats during forward on train batches

    # Single pass over many epochs' worth by cycling the loader
    # until we collect num_samples (after burn-in with thinning)
    while saved < num_samples:
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # either constant eps or cosine-annealed
            cur_eps = (
                cosine_annealed_eps(it, t_max, eps_min=eps * 0.1, eps_max=eps)
                if (use_cosine and t_max > 0)
                else eps
            )

            # Build grad_fn for *this* batch (uses utils flatten/unflatten)
            grad_fn = make_grad_fn(model, criterion, (inputs, targets), device, prior_prec)

            if sampler == "sgld":
                # g(x): compute from current x_flat
                g = grad_fn(x_flat)
                x_flat = sgld_step(x_flat, g, eps=cur_eps, tau=tau, debug=False, step=it)

            elif sampler == "asgld":
                g = grad_fn(x_flat)
                x_flat, m, v = asgld_step(
                    x=x_flat, m=m, v=v, grad=g,
                    eps=cur_eps, beta1=beta1, beta2=beta2, a=a, lambd=lambd, tau=tau,
                    step=it, debug=False
                )

            elif sampler in {"sam-sgld", "sam_sgld"}:
                x_flat = sam_sgld_step(
                    x=x_flat, grad_fn=grad_fn,
                    eps=cur_eps, tau=tau, rho=rho, lambd=lambd,
                    debug=False, step=it
                )

            elif sampler in {"sam-sgld-r1", "sam_sgld_rank_1"}:
                # data_grad_fn should not include prior; we add prior inside sam_sgld_rank_1 via prior_prec
                def data_grad_fn(xx):
                    # grad w.r.t. data likelihood only
                    unflatten_params(model, xx)
                    model.zero_grad(set_to_none=True)
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    loss.backward()
                    return flatten_grads(model)

                x_flat = sam_sgld_rank_1_step(
                    x=x_flat,
                    data_grad_fn=data_grad_fn,
                    prior_prec=prior_prec,
                    eps=cur_eps,
                    tau=tau,
                    rho=rho,
                    lambd=lambd,
                    sigma_dir=sigma_dir,
                    debug=False,
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

            # push params back to the model
            unflatten_params(model, x_flat)

            # Save sample after burn-in and thinning
            if it >= burn_in and ((it - burn_in) % thin == 0):
                sample = {
                    "x": x_flat.detach().clone().cpu(),
                    "bn": [b.detach().clone().cpu() for b in get_bn_buffers(model)],
                }
                torch.save(sample, out_dir / f"sample_{saved+1:04d}.pth")
                saved += 1

            it += 1
            if saved >= num_samples:
                break

    # Save manifest
    manifest = {
        "config": cfg,
        "num_samples": num_samples,
        "burn_in": burn_in,
        "thin": thin,
        "tau": tau,
        "prior_prec": prior_prec,
        "beta1": beta1,
        "beta2": beta2,
        "a": a,
        "lambd": lambd,
        "rho": rho,
        "sigma_dir": sigma_dir,
        "eps": eps,
        "cosine_eps": use_cosine,
        "t_max": t_max,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {num_samples} samples to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
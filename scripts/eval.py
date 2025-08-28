#!/usr/bin/env python
# scripts/eval.py
import argparse
from pathlib import Path

import torch
import yaml

from bayesian_lora.data.cifar import get_cifar_loaders
from bayesian_lora.models.resnet_cifar import ResNet18CIFAR, ResNet34CIFAR
from bayesian_lora.models.wide_resnet import WRN_28_10_CIFAR

# --- Use repo evaluation (predict_with_sample, evaluate_* ) ---
from bayesian_lora.eval.evaluate import (
    evaluate_model,
    evaluate_ensemble_bn,
)


def build_model(cfg_model, num_classes: int):
    name = cfg_model["name"].lower()
    if name in {"resnet18_cifar", "resnet18"}:
        return ResNet18CIFAR(num_classes=num_classes)
    if name in {"resnet34_cifar", "resnet34"}:
        return ResNet34CIFAR(num_classes=num_classes)
    if name in {"wrn_28_10_cifar", "wrn", "wideresnet"}:
        return WRN_28_10_CIFAR(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {cfg_model['name']}")


def main(cfg_path: str, samples_dir: str | None, k: int, single: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # data
    _, test_loader, num_classes = get_cifar_loaders(
        name=cfg["data"]["name"],
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 4),
        augment=False,
    )

    # model (weights overwritten per sample by evaluation utils)
    model = build_model(cfg["model"], num_classes=num_classes).to(device)

    # sample paths
    if samples_dir is None:
        samples_dir = cfg["out"]["dir"]
    paths = sorted(Path(samples_dir).glob("sample_*.pth"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No samples found under {samples_dir}")

    if single:
        # Evaluate just the first sample as a single model (flat + BN ignored here)
        s0 = torch.load(paths[0], map_location=device)
        acc, nll = evaluate_model(s0["x"].to(device), model, test_loader, device)
        print(f"Single-sample Acc: {acc:.3%} | NLL: {nll:.4f}")
        return

    # Ensemble of first k samples
    if k is not None and k > 0:
        paths = paths[:k]
    samples = [torch.load(p, map_location=device) for p in paths]  # each {'x': flat, 'bn': ...}

    acc, nll = evaluate_ensemble_bn(model, samples, test_loader, device)
    print(f"Ensemble@{len(samples)} — Acc: {acc:.3%} | NLL: {nll:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--samples_dir", default=None)
    ap.add_argument("--k", type=int, default=20, help="Use first k samples for the ensemble")
    ap.add_argument("--single", action="store_true", help="Evaluate only the first sample (no ensemble)")
    args = ap.parse_args()
    main(args.config, args.samples_dir, args.k, args.single)
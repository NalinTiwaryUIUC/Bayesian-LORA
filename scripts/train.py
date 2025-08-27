#!/usr/bin/env python
import argparse, yaml
def main(cfg_path):
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    # TODO: build loaders (bayesian_lora.data), model (bayesian_lora.models),
    # train/finetune, then run your sampling loop and save outputs to cfg["out"]["dir"].
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args(); main(args.config)

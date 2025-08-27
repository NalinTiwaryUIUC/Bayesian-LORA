#!/usr/bin/env python
import argparse, yaml
def main(cfg_path, ckpt=None):
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    # TODO: load model/samples and compute accuracy/ECE/NLL using bayesian_lora.eval.*
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--checkpoint")
    a = ap.parse_args(); main(a.config, a.checkpoint)

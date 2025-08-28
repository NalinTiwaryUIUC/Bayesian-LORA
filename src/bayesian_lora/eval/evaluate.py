# src/bayesian_lora/eval/evaluate.py
import math
import torch
import torch.nn.functional as F

from bayesian_lora.utils.params import unflatten_params
from bayesian_lora.utils.bn import load_bn_buffers


def predict_with_sample(model, sample, inputs):
    """
    sample: dict with keys:
      - 'x': flat parameter vector (Tensor)
      - 'bn': BN buffers (whatever format your load_bn_buffers expects)
    """
    unflatten_params(model, sample["x"])
    load_bn_buffers(model, sample["bn"])
    model.eval()
    with torch.no_grad():
        return model(inputs)


@torch.no_grad()
def evaluate_model(x, model, test_loader, device):
    """
    Evaluate a single flat-parameter sample x.
    Returns: (acc, avg_nll)
    """
    unflatten_params(model, x)
    model.eval()
    correct, total = 0, 0
    test_nll = 0.0

    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        test_nll += F.cross_entropy(outputs, lbls, reduction="sum").item()
        preds = outputs.argmax(1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)

    acc = correct / total
    avg_nll = test_nll / total
    return acc, avg_nll


@torch.no_grad()
def evaluate_ensemble_bn(model, samples, loader, device):
    """
    samples: list of dicts like {'x': flat_params, 'bn': bn_buffers}
    Returns: (acc, nll) using log-mean-exp for ensemble NLL.
    """
    correct, n_total, nll_sum = 0, 0, 0.0
    S = len(samples)
    assert S > 0, "No samples provided to evaluate_ensemble_bn."
    logS = math.log(S)

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logp_true_list = []
        probs_sum = None

        for sample in samples:
            logits = predict_with_sample(model, sample, inputs)
            logp = torch.log_softmax(logits, dim=1)
            logp_true_list.append(logp.gather(1, targets.unsqueeze(1)).squeeze(1))

            p = logp.exp()
            probs_sum = p if probs_sum is None else probs_sum + p

        # log mean p_true = logsumexp_s log p_true_s - log S
        logp_true_stacked = torch.stack(logp_true_list, dim=0)  # [S, B]
        log_mean_p_true = torch.logsumexp(logp_true_stacked, dim=0) - logS
        nll_sum += (-log_mean_p_true).sum().item()

        probs = probs_sum / S
        correct += probs.argmax(1).eq(targets).sum().item()
        n_total += targets.size(0)

    return correct / n_total, nll_sum / n_total
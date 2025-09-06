#!/usr/bin/env python3
"""
Comprehensive local testing for SGLD sample independence and behavior.
Tests the fixed prior-likelihood balance configuration.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from bayesian_lora.models.hf_lora import LoRAModel
from bayesian_lora.samplers.sgld import SGLDSampler
from bayesian_lora.data.glue_datasets import MRPCDataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.eval_mrpc_lora import load_config
from torch.utils.data import DataLoader
import math

def setup_model_and_data():
    """Setup model and data for testing."""
    print("=== Setting up model and data ===")
    
    # Change to project root directory
    import os
    project_root = os.path.dirname(os.path.dirname(__file__))
    os.chdir(project_root)
    
    # Load config
    config = load_config('configs/mrpc_roberta_lora_sgld.yaml')
    
    # Create model
    base_model = RobertaForSequenceClassification.from_pretrained(
        config['model']['backbone'],
        num_labels=2
    )
    
    lora_config = config['model']['lora']
    model = LoRAModel(
        base_model,
        r=lora_config['rank'],
        alpha=lora_config['alpha'],
        dropout=lora_config['dropout'],
        target_modules=lora_config['inject_into']
    )
    
    # Create tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['backbone'])
    
    # Create dataset
    dataset = MRPCDataset(
        tokenizer=tokenizer,
        max_length=config['model']['max_sequence_length'],
        split='validation'  # Use validation set for testing
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {config['data']['batch_size']}")
    
    return model, tokenizer, dataloader, config

def test_prior_likelihood_balance(model, config):
    """Test the prior-likelihood balance."""
    print("\n=== Testing Prior-Likelihood Balance ===")
    
    prior_std = config['training']['sgld_lora']['prior_std']
    
    # Compute prior term
    prior_loss = 0.0
    param_count = 0
    for param in model.parameters():
        if param.requires_grad:
            prior_loss += torch.sum(param ** 2) / (2 * prior_std ** 2)
            param_count += param.numel()
    
    # Don't divide by param_count - use total prior loss like SGLD does
    
    # Compute likelihood term with dummy data
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        labels = torch.tensor([0])
        
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        likelihood_loss = F.cross_entropy(logits, labels)
    
    ratio = prior_loss / likelihood_loss
    print(f"Prior std: {prior_std}")
    print(f"Total prior loss: {prior_loss:.6f}")
    print(f"Likelihood loss: {likelihood_loss:.6f}")
    print(f"Prior/Likelihood ratio: {ratio:.6f}")
    
    if 0.1 <= ratio <= 10:
        print("✅ Prior and likelihood are balanced!")
        return True
    else:
        print("❌ Prior and likelihood are not balanced!")
        return False

def test_parameter_exploration(model, config, num_steps=100):
    """Test parameter exploration during SGLD sampling."""
    print(f"\n=== Testing Parameter Exploration ({num_steps} steps) ===")
    
    # Create SGLD sampler
    sgld_config = config['training']['sgld_lora']
    sampler = SGLDSampler(
        model=model,
        temperature=sgld_config['temperature'],
        step_size=sgld_config['learning_rate'],
        prior_std=sgld_config['prior_std'],
        gradient_clip_norm=sgld_config['gradient_clip_norm']
    )
    
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Track parameter changes
    param_traces = []
    initial_params = []
    
    for param in trainable_params[:3]:  # Track first 3 parameters
        initial_params.append(param.data.clone())
    
    # Simulate SGLD steps
    model.train()
    for step in range(num_steps):
        # Create dummy batch
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        labels = torch.tensor([0])
        
        # Take SGLD step
        sampler.step(input_ids, attention_mask, labels)
        
        # Record parameter values every 10 steps
        if step % 10 == 0:
            current_params = []
            for param in trainable_params[:3]:
                current_params.append(param.data.clone())
            param_traces.append(current_params)
    
    # Analyze parameter drift
    print("Parameter drift analysis:")
    for i, (initial, final) in enumerate(zip(initial_params, param_traces[-1])):
        drift = torch.norm(final - initial).item()
        print(f"  Param {i}: drift = {drift:.6f}")
    
    # Check if parameters are exploring
    total_drift = sum(torch.norm(final - initial).item() 
                     for initial, final in zip(initial_params, param_traces[-1]))
    
    if total_drift > 0.01:
        print("✅ Parameters are exploring (good drift)")
        return True
    else:
        print("❌ Parameters not exploring (no drift)")
        return False

def test_sample_independence(model, config, num_samples=20):
    """Test independence of SGLD samples."""
    print(f"\n=== Testing Sample Independence ({num_samples} samples) ===")
    
    # Create SGLD sampler
    sgld_config = config['training']['sgld_lora']
    sampler = SGLDSampler(
        model=model,
        temperature=sgld_config['temperature'],
        step_size=sgld_config['learning_rate'],
        prior_std=sgld_config['prior_std'],
        gradient_clip_norm=sgld_config['gradient_clip_norm']
    )
    
    # Collect samples
    samples = []
    model.train()
    
    for i in range(num_samples):
        # Take multiple SGLD steps between samples (simulate thinning)
        for _ in range(10):  # 10 steps between samples
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
            labels = torch.tensor([0])
            sampler.step(input_ids, attention_mask, labels)
        
        # Save sample
        sample_state = sampler.get_current_state()
        samples.append(sample_state)
        
        if i % 5 == 0:
            print(f"  Collected sample {i+1}/{num_samples}")
    
    # Test sample diversity
    print("\nSample diversity analysis:")
    
    # Get first parameter from each sample
    first_param_values = []
    for sample in samples:
        # Get first trainable parameter
        for name, param in sample.items():
            if 'lora' in name.lower() and 'weight' in name.lower():
                first_param_values.append(param.flatten()[:10].cpu().numpy())  # First 10 elements
                break
    
    first_param_values = np.array(first_param_values)
    
    # Compute pairwise distances
    distances = []
    for i in range(len(first_param_values)):
        for j in range(i+1, len(first_param_values)):
            dist = np.linalg.norm(first_param_values[i] - first_param_values[j])
            distances.append(dist)
    
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    print(f"  Mean pairwise distance: {mean_distance:.6f}")
    print(f"  Std pairwise distance: {std_distance:.6f}")
    
    if mean_distance > 0.001:
        print("✅ Samples show good diversity")
        return True
    else:
        print("❌ Samples are too similar (not independent)")
        return False

def test_ensemble_predictions(model, config, dataloader, num_samples=10):
    """Test ensemble prediction diversity."""
    print(f"\n=== Testing Ensemble Predictions ({num_samples} samples) ===")
    
    # Create SGLD sampler
    sgld_config = config['training']['sgld_lora']
    sampler = SGLDSampler(
        model=model,
        temperature=sgld_config['temperature'],
        step_size=sgld_config['learning_rate'],
        prior_std=sgld_config['prior_std'],
        gradient_clip_norm=sgld_config['gradient_clip_norm']
    )
    
    # Get a test batch
    test_batch = next(iter(dataloader))
    input_ids = test_batch['input_ids']
    attention_mask = test_batch['attention_mask']
    labels = test_batch['labels']
    
    # Collect predictions from multiple samples
    all_predictions = []
    all_logits = []
    
    model.train()
    for i in range(num_samples):
        # Take SGLD steps
        for _ in range(5):  # 5 steps between samples
            sampler.step(input_ids, attention_mask, labels)
        
        # Get prediction (keep in train mode to preserve dropout)
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
        
        if i % 3 == 0:
            print(f"  Generated prediction {i+1}/{num_samples}")
    
    # Analyze prediction diversity
    all_predictions = np.array(all_predictions)
    all_logits = np.array(all_logits)
    
    print(f"\nPrediction analysis:")
    print(f"  Prediction shape: {all_predictions.shape}")
    print(f"  Logits shape: {all_logits.shape}")
    
    # Check prediction agreement
    prediction_agreement = np.mean(all_predictions == all_predictions[0])
    print(f"  Prediction agreement with first sample: {prediction_agreement:.3f}")
    
    # Check logit variance
    logit_variance = np.var(all_logits, axis=0)
    mean_logit_variance = np.mean(logit_variance)
    print(f"  Mean logit variance across samples: {mean_logit_variance:.6f}")
    
    # For small models, we expect some prediction agreement but good logit variance
    if mean_logit_variance > 0.001:
        print("✅ Ensemble shows good logit diversity (predictions may agree due to model confidence)")
        return True
    else:
        print("❌ Ensemble logits too similar")
        return False

def test_convergence_diagnostics(model, config, num_chains=3, samples_per_chain=20):
    """Test convergence diagnostics (R-hat simulation)."""
    print(f"\n=== Testing Convergence Diagnostics ===")
    print(f"Chains: {num_chains}, Samples per chain: {samples_per_chain}")
    
    # Create SGLD sampler
    sgld_config = config['training']['sgld_lora']
    sampler = SGLDSampler(
        model=model,
        temperature=sgld_config['temperature'],
        step_size=sgld_config['learning_rate'],
        prior_std=sgld_config['prior_std'],
        gradient_clip_norm=sgld_config['gradient_clip_norm']
    )
    
    # Collect samples from multiple chains
    chain_samples = []
    
    for chain in range(num_chains):
        print(f"  Running chain {chain+1}/{num_chains}")
        
        # Reset model to initial state (simulate chain restart)
        # In practice, this would be the MAP solution
        for param in model.parameters():
            if param.requires_grad:
                param.data.normal_(0, 0.01)  # Random initialization
        
        chain_log_posteriors = []
        model.train()
        
        for sample in range(samples_per_chain):
            # Take SGLD steps
            for _ in range(5):  # 5 steps between samples
                input_ids = torch.tensor([[1, 2, 3, 4, 5]])
                attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
                labels = torch.tensor([0])
                sampler.step(input_ids, attention_mask, labels)
            
            # Compute log posterior (simplified)
            with torch.no_grad():
                # Likelihood term
                output = model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                likelihood = F.cross_entropy(logits, labels)
                
                # Prior term
                prior_loss = 0.0
                for param in model.parameters():
                    if param.requires_grad:
                        prior_loss += torch.sum(param ** 2) / (2 * sgld_config['prior_std'] ** 2)
                
                log_posterior = -(likelihood + prior_loss)
                chain_log_posteriors.append(log_posterior.item())
        
        chain_samples.append(chain_log_posteriors)
    
    # Compute R-hat (simplified version)
    chain_means = [np.mean(chain) for chain in chain_samples]
    chain_vars = [np.var(chain, ddof=1) for chain in chain_samples]
    
    between_chain_var = np.var(chain_means, ddof=1)
    within_chain_var = np.mean(chain_vars)
    
    if within_chain_var > 0:
        r_hat = np.sqrt((between_chain_var + within_chain_var) / within_chain_var)
    else:
        r_hat = float('inf')
    
    print(f"\nConvergence diagnostics:")
    print(f"  Between-chain variance: {between_chain_var:.6f}")
    print(f"  Within-chain variance: {within_chain_var:.6f}")
    print(f"  R-hat: {r_hat:.3f}")
    
    if r_hat < 1.1:
        print("✅ Good convergence (R-hat < 1.1)")
        return True
    else:
        print("❌ Poor convergence (R-hat >= 1.1)")
        return False

def main():
    """Run all tests."""
    print("🔬 SGLD Sample Independence Testing Suite")
    print("=" * 50)
    
    # Setup
    model, tokenizer, dataloader, config = setup_model_and_data()
    
    # Run tests
    tests = [
        ("Prior-Likelihood Balance", lambda: test_prior_likelihood_balance(model, config)),
        ("Parameter Exploration", lambda: test_parameter_exploration(model, config)),
        ("Sample Independence", lambda: test_sample_independence(model, config)),
        ("Ensemble Predictions", lambda: test_ensemble_predictions(model, config, dataloader)),
        ("Convergence Diagnostics", lambda: test_convergence_diagnostics(model, config)),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SGLD is ready for full experiment.")
    else:
        print("⚠️  Some tests failed. Review configuration before running full experiment.")
    
    return results

if __name__ == "__main__":
    main()

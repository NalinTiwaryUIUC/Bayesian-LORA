# src/bayesian_lora/samplers/sgld.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Sampler Classes for Training Scripts
# =============================================================================

class BaseSampler:
    """Base class for all samplers."""
    
    def __init__(self, model, temperature=1.0, step_size=1e-4, noise_scale=1.0):
        self.model = model
        self.temperature = temperature
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.device = next(model.parameters()).device
    
    def get_current_state(self):
        """Get current model state."""
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}
    
    def step(self, data, target):
        """Take one sampling step."""
        raise NotImplementedError


class SGLDSampler(BaseSampler):
    """Stochastic Gradient Langevin Dynamics sampler."""
    
    def __init__(self, model, temperature=1.0, step_size=1e-4, noise_scale=1.0, 
                 prior_std=0.1, gradient_clip_norm=1.0):
        super().__init__(model, temperature, step_size, noise_scale)
        self.prior_std = prior_std
        self.gradient_clip_norm = gradient_clip_norm
    
    def step(self, *args):
        """
        Take one SGLD step.
        Supports both (data, target) and (input_ids, attention_mask, labels) formats.
        """
        self.model.train()
        
        # Handle different input formats
        if len(args) == 2:
            # Traditional format: (data, target)
            data, target = args
            output = self.model(data)
        elif len(args) == 3:
            # LoRA format: (input_ids, attention_mask, labels)
            input_ids, attention_mask, labels = args
            output = self.model(input_ids, attention_mask=attention_mask)
            target = labels
        else:
            raise ValueError("Expected 2 or 3 arguments")
        
        # Handle different output formats
        if hasattr(output, 'logits'):
            # HuggingFace output object (LoRA models)
            logits = output.logits
        else:
            # Direct logits (CIFAR models)
            logits = output
        
        # Compute loss
        loss = F.cross_entropy(logits, target)
        
        # Add prior term (Gaussian prior on LoRA parameters)
        prior_loss = 0.0
        for param in self.model.parameters():
            if param.requires_grad:  # Only LoRA parameters
                prior_loss += torch.sum(param ** 2) / (2 * self.prior_std ** 2)
        
        total_loss = loss + prior_loss
        
        # Backward pass
        self.model.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        
        # SGLD update
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None and param.requires_grad:
                    # SGLD update: θ ← θ - η∇L(θ) + √(2η/τ)ξ
                    # where L(θ) = NLL + prior
                    param.data -= self.step_size * param.grad
                    
                    # Add noise: √(2η/τ)ξ
                    noise_std = math.sqrt(2 * self.temperature * self.step_size)
                    noise = torch.randn_like(param) * noise_std
                    param.data += noise


class ASGLDSampler(BaseSampler):
    """Adaptive SGLD sampler."""
    
    def __init__(self, model, temperature=1.0, step_size=1e-4, noise_scale=1.0, 
                 beta1=0.9, beta2=0.999, a=0.1, lambd=1e-8):
        super().__init__(model, temperature, step_size, noise_scale)
        self.beta1 = beta1
        self.beta2 = beta2
        self.a = a
        self.lambd = lambd
        
        # Initialize moment estimates
        self.m = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        self.v = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
    
    def step(self, data, target):
        self.model.train()
        
        # Forward pass
        output = self.model(data)
        
        # Handle different output formats
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
            
        loss = F.cross_entropy(logits, target)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # ASGLD update
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.data
                    
                    # Update moment estimates
                    self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                    self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad * grad)
                    
                    # Adaptive drift
                    A = self.m[name] / (self.v[name].sqrt() + self.lambd)
                    
                    # Noise
                    noise = torch.randn_like(param) * math.sqrt(2 * self.temperature * self.step_size)
                    
                    # Update
                    param.data -= self.step_size * (grad + self.a * A) + noise


class SAMSGLDSampler(BaseSampler):
    """SAM-SGLD sampler."""
    
    def __init__(self, model, temperature=1.0, step_size=1e-4, noise_scale=1.0, 
                 rho=0.1, lambd=1e-8, beta1=0.9, beta2=0.999):
        super().__init__(model, temperature, step_size, noise_scale)
        self.rho = rho
        self.lambd = lambd
        self.beta1 = beta1
        self.beta2 = beta2
    
    def step(self, data, target):
        self.model.train()
        
        # First forward pass
        output1 = self.model(data)
        
        # Handle different output formats
        if hasattr(output1, 'logits'):
            logits1 = output1.logits
        else:
            logits1 = output1
            
        loss1 = F.cross_entropy(logits1, target)
        
        # Backward pass for first gradient
        self.model.zero_grad()
        loss1.backward()
        
        # Store first gradient
        grad1 = {name: param.grad.clone() if param.grad is not None else None 
                for name, param in self.model.named_parameters()}
        
        # Perturb parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if grad1[name] is not None:
                    grad_norm = grad1[name].norm() + self.lambd
                    delta = self.rho * grad1[name] / grad_norm
                    param.data += delta
        
        # Second forward pass
        output2 = self.model(data)
        
        # Handle different output formats
        if hasattr(output2, 'logits'):
            logits2 = output2.logits
        else:
            logits2 = output2
            
        loss2 = F.cross_entropy(logits2, target)
        
        # Backward pass for second gradient
        self.model.zero_grad()
        loss2.backward()
        
        # SGLD update with SAM gradient
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Add noise
                    noise = torch.randn_like(param) * math.sqrt(2 * self.temperature * self.step_size)
                    param.data -= self.step_size * param.grad + noise
        
        # Restore original parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if grad1[name] is not None:
                    grad_norm = grad1[name].norm() + self.lambd
                    delta = self.rho * grad1[name] / grad_norm
                    param.data -= delta


class SAMSGLDRank1Sampler(BaseSampler):
    """SAM-SGLD Rank-1 sampler."""
    
    def __init__(self, model, temperature=1.0, step_size=1e-4, noise_scale=1.0, 
                 rho=0.1, lambd=1e-8, beta1=0.9, beta2=0.999, sigma_dir=1.0):
        super().__init__(model, temperature, step_size, noise_scale)
        self.rho = rho
        self.lambd = lambd
        self.beta1 = beta1
        self.beta2 = beta2
        self.sigma_dir = sigma_dir
    
    def step(self, data, target):
        self.model.train()
        
        # First forward pass
        output1 = self.model(data)
        
        # Handle different output formats
        if hasattr(output1, 'logits'):
            logits1 = output1.logits
        else:
            logits1 = output1
            
        loss1 = F.cross_entropy(logits1, target)
        
        # Backward pass for first gradient
        self.model.zero_grad()
        loss1.backward()
        
        # Store first gradient
        grad1 = {name: param.grad.clone() if param.grad is not None else None 
                for name, param in self.model.named_parameters()}
        
        # Perturb parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if grad1[name] is not None:
                    grad_norm = grad1[name].norm() + self.lambd
                    delta = self.rho * grad1[name] / grad_norm
                    param.data += delta
        
        # Second forward pass
        output2 = self.model(data)
        
        # Handle different output formats
        if hasattr(output2, 'logits'):
            logits2 = output2.logits
        else:
            logits2 = output2
            
        loss2 = F.cross_entropy(logits2, target)
        
        # Backward pass for second gradient
        self.model.zero_grad()
        loss2.backward()
        
        # SGLD update with SAM gradient and rank-1 noise
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Rank-1 noise
                    noise_std = math.sqrt(2 * self.temperature * self.step_size)
                    z = torch.randn_like(param)
                    u_hat = param.grad / (param.grad.norm() + 1e-8)
                    z_proj = torch.dot(z.flatten(), u_hat.flatten())
                    noise = noise_std * (z + self.sigma_dir * z_proj * u_hat)
                    
                    param.data -= self.step_size * param.grad + noise
        
        # Restore original parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if grad1[name] is not None:
                    grad_norm = grad1[name].norm() + self.lambd
                    delta = self.rho * grad1[name] / grad_norm
                    param.data -= delta
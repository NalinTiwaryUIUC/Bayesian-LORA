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
        """Get current model state (only trainable parameters to save memory)."""
        # Clear GPU cache before saving state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Only save trainable parameters (LoRA parameters)
        return {name: param.clone().detach() for name, param in self.model.named_parameters() 
                if param.requires_grad}
    
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
        self.noise_scale = noise_scale  # Store noise scale for use in step function
        
        # Initialize diagnostic attributes
        self.last_drift_norm = 0.0
        self.last_noise_norm = 0.0
        self.last_step_size = 0.0
    
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
        
        # Add prior term (Gaussian prior on trainable parameters)
        prior_loss = 0.0
        for param in self.model.parameters():
            if param.requires_grad:  # Only trainable parameters
                prior_loss += torch.sum(param ** 2) / (2 * self.prior_std ** 2)
        
        total_loss = loss + prior_loss
        
        # Backward pass
        self.model.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        
        # SGLD update
        total_drift_norm = 0.0
        total_noise_norm = 0.0
        
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None and param.requires_grad:
                    # SGLD update: θ ← θ - η∇L(θ) + √(2η/τ)ξ
                    # where L(θ) = NLL + prior
                    drift = self.step_size * param.grad
                    param.data = param.data - drift
                    
                    # Add noise: √(2η/τ)ξ
                    # Use configurable noise scaling for stability
                    noise_std = math.sqrt(2 * self.step_size / self.temperature) * self.noise_scale
                    noise = torch.randn_like(param) * noise_std
                    param.data = param.data + noise
                    
                    # Track norms for diagnostics
                    total_drift_norm += drift.norm().item() ** 2
                    total_noise_norm += noise.norm().item() ** 2
        
        # Store diagnostic information
        self.last_drift_norm = math.sqrt(total_drift_norm)
        self.last_noise_norm = math.sqrt(total_noise_norm)
        self.last_step_size = self.step_size


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
                    noise = torch.randn_like(param) * math.sqrt(2 * self.step_size / self.temperature) * 0.05
                    
    # Update
                    param.data = param.data - self.step_size * (grad + self.a * A) + noise


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
                    param.data = param.data + delta
        
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
                    noise = torch.randn_like(param) * math.sqrt(2 * self.step_size / self.temperature) * 0.05
                    param.data = param.data - self.step_size * param.grad + noise
        
        # Restore original parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if grad1[name] is not None:
                    grad_norm = grad1[name].norm() + self.lambd
                    delta = self.rho * grad1[name] / grad_norm
                    param.data = param.data - delta


class SAMSGLDRank1Sampler(BaseSampler):
    """SAM-SGLD Rank-1 sampler."""
    
    def __init__(self, model, temperature=1.0, step_size=1e-4, noise_scale=1.0, 
                 rho=0.1, lambd=1e-8, sigma_dir=1.0, gradient_clip_norm=1.0, prior_std=0.1):
        super().__init__(model, temperature, step_size, noise_scale)
        self.rho = rho
        self.lambd = lambd
        self.sigma_dir = sigma_dir
        self.gradient_clip_norm = gradient_clip_norm
        self.prior_std = prior_std
        # Store last step diagnostics
        self.last_drift_norm = 0.0
        self.last_noise_norm = 0.0
        self.last_step_size = 0.0
    
    def step(self, *args):
        """
        Take one SAM-SGLD step.
        Supports both (data, target) and (input_ids, attention_mask, labels) formats.
        """
        self.model.train()
        
        # Handle different input formats
        if len(args) == 2:
            # Traditional format: (data, target)
            data, target = args
            input_ids, attention_mask, labels = None, None, target
            output1 = self.model(data)
        elif len(args) == 3:
            # LoRA format: (input_ids, attention_mask, labels)
            input_ids, attention_mask, labels = args
            data, target = None, labels
            output1 = self.model(input_ids, attention_mask=attention_mask)
        else:
            raise ValueError("Expected 2 or 3 arguments")
        
        # First forward pass
        
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
                    param.data = param.data + delta
        
        # Second forward pass
        if data is not None:
            output2 = self.model(data)
        else:
            output2 = self.model(input_ids, attention_mask=attention_mask)
        
        # Handle different output formats
        if hasattr(output2, 'logits'):
            logits2 = output2.logits
        else:
            logits2 = output2
            
        loss2 = F.cross_entropy(logits2, target)
        
        # Note: Prior term should be evaluated at original parameters, not perturbed ones
        # We'll add it after restoring parameters
        total_loss = loss2
        
        # Backward pass for second gradient
        self.model.zero_grad()
        total_loss.backward()
        
        # Store the SAM gradient (computed at perturbed parameters)
        sam_grad = {name: param.grad.clone() if param.grad is not None else None 
                   for name, param in self.model.named_parameters()}
        
        # Restore original parameters BEFORE SGLD update
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if grad1[name] is not None:
                    grad_norm = grad1[name].norm() + self.lambd
                    delta = self.rho * grad1[name] / grad_norm
                    param.data = param.data - delta
        
        # Add prior term to SAM gradient (evaluated at original parameters)
        prior_grad = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                prior_grad[name] = param / (self.prior_std ** 2)
        
        # Combine SAM gradient with prior gradient
        combined_grad = {}
        for name in sam_grad:
            if sam_grad[name] is not None and name in prior_grad:
                combined_grad[name] = sam_grad[name] + prior_grad[name]
            elif sam_grad[name] is not None:
                combined_grad[name] = sam_grad[name]
            elif name in prior_grad:
                combined_grad[name] = prior_grad[name]
            else:
                combined_grad[name] = None
        
        # Apply gradient clipping to the combined gradient (SAM + prior)
        if self.gradient_clip_norm > 0:
            # Compute total gradient norm
            total_norm = 0
            for grad in combined_grad.values():
                if grad is not None:
                    total_norm += grad.norm().item() ** 2
            total_norm = math.sqrt(total_norm)
            
            # Clip if necessary
            if total_norm > self.gradient_clip_norm:
                clip_coef = self.gradient_clip_norm / (total_norm + 1e-6)
                for name in combined_grad:
                    if combined_grad[name] is not None:
                        combined_grad[name] = combined_grad[name] * clip_coef
        
        # SGLD update with combined gradient and rank-1 noise (applied to original parameters)
        total_drift_norm = 0.0
        total_noise_norm = 0.0
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if combined_grad[name] is not None:
                    # Calculate actual drift (step_size * gradient)
                    drift = self.step_size * combined_grad[name]
                    total_drift_norm += drift.norm().item() ** 2
                    
                    # Rank-1 noise direction based on SAM gradient (corrected)
                    noise_std = math.sqrt(2 * self.step_size / self.temperature) * self.noise_scale
                    z = torch.randn_like(param)
                    u_hat = sam_grad[name] / (sam_grad[name].norm() + self.lambd) if sam_grad[name] is not None else None
                    if u_hat is not None:
                        z_proj = torch.dot(z.flatten(), u_hat.flatten())
                        noise = noise_std * (z + self.sigma_dir * z_proj * u_hat)
                    else:
                        noise = noise_std * z  # Fallback to isotropic noise
                    
                    total_noise_norm += noise.norm().item() ** 2
                    
                    param.data = param.data - drift + noise
        
        # Store diagnostics for logging
        self.last_drift_norm = math.sqrt(total_drift_norm)
        self.last_noise_norm = math.sqrt(total_noise_norm)
        self.last_step_size = self.step_size
        
        # SAM-specific diagnostics
        self.last_sam_perturbation_norm = 0.0
        self.last_sam_gradient_norm = 0.0
        self.last_rank1_noise_contribution = 0.0
        self.last_loss_change = 0.0
        
        # Calculate SAM perturbation norm
        sam_perturbation_norm = 0.0
        for name, param in self.model.named_parameters():
            if grad1[name] is not None:
                grad_norm = grad1[name].norm() + self.lambd
                delta = self.rho * grad1[name] / grad_norm
                sam_perturbation_norm += delta.norm().item() ** 2
        self.last_sam_perturbation_norm = math.sqrt(sam_perturbation_norm)
        
        # Calculate SAM gradient norm
        sam_gradient_norm = 0.0
        for grad in sam_grad.values():
            if grad is not None:
                sam_gradient_norm += grad.norm().item() ** 2
        self.last_sam_gradient_norm = math.sqrt(sam_gradient_norm)
        
        # Calculate rank-1 noise contribution
        rank1_noise_norm = 0.0
        isotropic_noise_norm = 0.0
        for name, param in self.model.named_parameters():
            if combined_grad[name] is not None:
                noise_std = math.sqrt(2 * self.step_size / self.temperature) * self.noise_scale
                z = torch.randn_like(param)
                u_hat = sam_grad[name] / (sam_grad[name].norm() + self.lambd) if sam_grad[name] is not None else None
                if u_hat is not None:
                    z_proj = torch.dot(z.flatten(), u_hat.flatten())
                    rank1_component = self.sigma_dir * z_proj * u_hat
                    rank1_noise_norm += (noise_std * rank1_component).norm().item() ** 2
                    isotropic_noise_norm += (noise_std * z).norm().item() ** 2
                else:
                    isotropic_noise_norm += (noise_std * z).norm().item() ** 2
        
        total_rank1_noise_norm = math.sqrt(rank1_noise_norm)
        total_isotropic_noise_norm = math.sqrt(isotropic_noise_norm)
        if total_rank1_noise_norm + total_isotropic_noise_norm > 0:
            self.last_rank1_noise_contribution = total_rank1_noise_norm / (total_rank1_noise_norm + total_isotropic_noise_norm)
        else:
            self.last_rank1_noise_contribution = 0.0
        
        # Calculate loss change (approximate)
        self.last_loss_change = abs(loss2.item() - loss1.item())
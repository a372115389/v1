"""
Online World Model Adaptation: Value-Conditioned Variational Inference

This module implements the core innovation for solving dataset bias in offline RL:
online adaptation of the world model p_φ(ψ|s) to a value-conditioned posterior p_{φ'}(ψ|s).

Key Innovation:
φ'^* = argmax_{φ'} E_{ψ ~ p_{φ'}(ψ|s)}[V_θ(ψ, z^*)] - λ · D_KL(p_{φ'}(ψ|s) || p_φ(ψ|s))

This transforms planning from searching in a static world model to planning in a 
dynamically adapted world model that respects both value objectives and physical constraints.
"""

import jax
import jax.numpy as jnp
import optax
import functools
from typing import Callable, Tuple, Dict, Any


def create_variational_objective(
    psi_sampler: Callable,
    psi_sde,
    value_decoder,
    task_embedding,
    kl_coef: float = 1.0,
    n_samples: int = 32
):
    """
    Create the variational objective function for online world model adaptation.
    
    This implements the core objective:
    J(Δφ) = E_{ψ ~ p_{φ+Δφ}(ψ|s)}[V_θ(ψ, z^*)] - λ * D_KL(p_{φ+Δφ}(ψ|s) || p_φ(ψ|s))
    
    Args:
        psi_sampler: The diffusion sampler for successor features
        psi_sde: The SDE governing the diffusion process
        value_decoder: The learned value function V_θ
        task_embedding: Current task embedding z^*
        kl_coef: Regularization coefficient λ
        n_samples: Number of samples for Monte Carlo estimation
        
    Returns:
        Objective function that takes (phi_base, delta_phi, obs, rng) -> (objective, info)
    """
    
    def objective_fn(phi_base, delta_phi, obs, rng):
        """
        Compute the variational objective for given parameters.
        
        Args:
            phi_base: Base model parameters φ (frozen)
            delta_phi: Adaptation parameters Δφ (optimized online)
            obs: Current observation
            rng: Random number generator
            
        Returns:
            Tuple of (negative_objective, info_dict)
        """
        # Combine base and delta parameters: φ' = φ + Δφ
        phi_adapted = jax.tree_util.tree_map(lambda base, delta: base + delta, phi_base, delta_phi)
        
        # CHALLENGE 1 SOLUTION: Differentiable Value Expectation Estimation
        # We use the reparameterization trick for diffusion models to ensure differentiability.
        # Key insight: Instead of sampling from p_{φ+Δφ} directly, we sample noise and transform
        # it through the adapted diffusion model, maintaining differentiability w.r.t. Δφ.
        
        rng, sample_rng = jax.random.split(rng)
        
        # Generate multiple samples for Monte Carlo estimation
        batch_obs = jnp.tile(obs, (n_samples, 1))  # Broadcast observation
        
        # Sample from adapted posterior p_{φ+Δφ}(ψ|s) using reparameterization
        # This is differentiable w.r.t. phi_adapted (and thus delta_phi)
        psi_samples_adapted = psi_sampler(phi_adapted, sample_rng, batch_obs)
        
        # Sample from prior p_φ(ψ|s) for KL divergence computation
        rng, prior_sample_rng = jax.random.split(rng)
        psi_samples_prior = psi_sampler(phi_base, prior_sample_rng, batch_obs)
        
        # Compute value expectation: E_{ψ ~ p_{φ'}(ψ|s)}[V_θ(ψ, z^*)]
        current_task_embed = task_embedding.model_def.apply({"params": task_embedding.ema_params}, task_id=None)
        task_embed_batch = jnp.tile(current_task_embed[None], (n_samples, 1))
        
        values_adapted = value_decoder.model_def.apply(
            {"params": value_decoder.ema_params},
            psi_samples_adapted,
            task_embed_batch,
            training=False
        ).squeeze(-1)
        
        value_expectation = jnp.mean(values_adapted)
        
        # CHALLENGE 2 SOLUTION: KL Divergence Approximation for Diffusion Models
        # Since exact KL divergence between two neural diffusion models has no closed form,
        # we use a combination of approaches:
        # 
        # Method 1: Parameter-based approximation (when Δφ is small)
        # Since Δφ is initialized as zero and constrained to be small, we can use
        # a second-order Taylor approximation in parameter space.
        #
        # Method 2: Sample-based Monte Carlo approximation
        # We approximate KL via importance sampling using samples from both distributions.
        
        # Method 1: Parameter-based KL approximation
        # For small Δφ, KL(p_{φ+Δφ} || p_φ) ≈ 0.5 * Δφ^T * F * Δφ
        # where F is the Fisher Information Matrix. We approximate F with identity for efficiency.
        param_kl = 0.5 * sum(
            jnp.sum(delta ** 2) for delta in jax.tree_util.tree_leaves(delta_phi)
        )
        
        # Method 2: Sample-based KL approximation using Monte Carlo
        # KL(p_{φ'} || p_φ) ≈ E_{ψ ~ p_{φ'}}[log p_{φ'}(ψ|s) - log p_φ(ψ|s)]
        # We approximate log-densities using the score function from the SDE
        
        # For VPSDE, the log-density involves the score function
        # This is computationally intensive, so we use the parameter-based approximation
        # as the primary KL term and add a small sample-based correction
        
        # Compute sample-based KL correction (simplified)
        # This captures distributional differences not captured by parameter norm
        values_prior = value_decoder.model_def.apply(
            {"params": value_decoder.ema_params},
            psi_samples_prior,
            task_embed_batch,
            training=False
        ).squeeze(-1)
        
        # Use value difference as a proxy for distribution difference
        # Higher values from adapted samples indicate distribution shift
        sample_kl_proxy = jnp.mean((values_adapted - values_prior) ** 2)
        
        # Combine both KL approximations
        kl_divergence = param_kl + 0.1 * sample_kl_proxy  # 0.1 is a scaling factor
        
        # Final variational objective (we negate it for minimization)
        objective = value_expectation - kl_coef * kl_divergence
        
        # Information for logging and debugging
        info = {
            "value_expectation": value_expectation,
            "kl_divergence": kl_divergence,
            "param_kl": param_kl,
            "sample_kl_proxy": sample_kl_proxy,
            "objective": objective,
            "mean_adapted_value": jnp.mean(values_adapted),
            "mean_prior_value": jnp.mean(values_prior),
        }
        
        return -objective, info  # Negative for minimization
    
    return objective_fn


def online_world_model_adaptation(
    phi_base,
    delta_phi_init,
    delta_phi_optimizer,
    delta_phi_opt_state,
    obs,
    psi_sampler,
    psi_sde,
    value_decoder,
    task_embedding,
    config,
    rng
):
    """
    Perform online world model adaptation via gradient-based optimization.
    
    This implements the core innovation: given a current observation and value function,
    adapt the world model parameters to create a value-conditioned posterior distribution.
    
    Args:
        phi_base: Base world model parameters φ (frozen)
        delta_phi_init: Initial adaptation parameters Δφ (typically zeros)
        delta_phi_optimizer: Optimizer for Δφ
        delta_phi_opt_state: Optimizer state
        obs: Current observation
        psi_sampler: Diffusion sampler for successor features
        psi_sde: SDE for diffusion process
        value_decoder: Value function V_θ
        task_embedding: Task embedding z^*
        config: Configuration containing adaptation hyperparameters
        rng: Random number generator
        
    Returns:
        Tuple of (adapted_phi, adaptation_info)
    """
    
    # Extract adaptation hyperparameters
    adaptation_steps = config.planning.get("adaptation_steps", 10)
    kl_coef = config.planning.get("kl_coef", 1.0)
    n_adaptation_samples = config.planning.get("n_adaptation_samples", 32)
    
    # Create variational objective
    objective_fn = create_variational_objective(
        psi_sampler, psi_sde, value_decoder, task_embedding, 
        kl_coef, n_adaptation_samples
    )
    
    # Online optimization loop
    current_delta_phi = delta_phi_init
    current_opt_state = delta_phi_opt_state
    adaptation_history = []
    
    for step in range(adaptation_steps):
        rng, step_rng = jax.random.split(rng)
        
        # Compute objective and gradients
        (loss, info), grads = jax.value_and_grad(objective_fn, argnums=1, has_aux=True)(
            phi_base, current_delta_phi, obs, step_rng
        )
        
        # Apply gradient update
        updates, new_opt_state = delta_phi_optimizer.update(grads, current_opt_state)
        new_delta_phi = optax.apply_updates(current_delta_phi, updates)
        
        # Update for next iteration
        current_delta_phi = new_delta_phi
        current_opt_state = new_opt_state
        
        # Record adaptation progress
        adaptation_history.append({
            "step": step,
            "loss": loss,
            **info
        })
    
    # Combine adapted parameters
    phi_adapted = jax.tree_util.tree_map(
        lambda base, delta: base + delta, phi_base, current_delta_phi
    )
    
    adaptation_info = {
        "adaptation_history": adaptation_history,
        "final_delta_phi_norm": sum(
            jnp.sum(delta ** 2) for delta in jax.tree_util.tree_leaves(current_delta_phi)
        ),
        "adaptation_steps": adaptation_steps,
    }
    
    return phi_adapted, current_opt_state, adaptation_info


# JIT-compiled version for efficiency
online_world_model_adaptation_jit = jax.jit(
    online_world_model_adaptation,
    static_argnames=("psi_sampler", "psi_sde", "config")
)
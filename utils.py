import functools

import jax
import jax.numpy as jnp
import optax
from models.online_adaptation import online_world_model_adaptation_jit


def batch_add(a, b):
    return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


def flatten_dict(config):
    """Flatten a hierarchical dict to a simple dict."""
    new_dict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            sub_dict = flatten_dict(value)
            for subkey, subvalue in sub_dict.items():
                new_dict[key + "/" + subkey] = subvalue
        elif isinstance(value, tuple):
            new_dict[key] = str(value)
        else:
            new_dict[key] = value
    return new_dict


def clip_by_global_norm(max_norm):
    """Scale gradient updates using their global norm.

    Args:
      max_norm: The maximum global norm for an update.

    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        g_norm = optax.global_norm(updates)
        updates = jax.tree_util.tree_map(lambda t: (t / g_norm) * max_norm, updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def get_planner(planner, guidance_fn_builder, num_samples, num_elites, config=None):
    """
    Create a planner with online world model adaptation capability.
    
    CORE INNOVATION: This planner implements value-conditioned variational inference
    to adapt the world model p_φ(ψ|s) → p_{φ'}(ψ|s) in real-time, enabling planning
    beyond the limitations of offline data distribution.
    
    Args:
        planner: Planning algorithm ("random_shooting", "guided_diffusion", or "online_adaptation")
        guidance_fn_builder: Function that builds guidance from ValueDecoder and TaskEmbedding  
        num_samples: Number of samples for planning
        num_elites: Number of elite samples to average
        
    Returns:
        Planner function that takes (rng, psi, psi_sampler, policy, policy_sampler,
                                    value_decoder, task_embedding, delta_phi, 
                                    delta_phi_optimizer, delta_phi_opt_state, obs)
    """
    @functools.partial(
        jax.jit,
        static_argnames=("psi_sampler", "policy_sampler"),
    )
    def planner_fn(rng, psi, psi_sampler, policy, policy_sampler, value_decoder, 
                   task_embedding, delta_phi, delta_phi_optimizer, delta_phi_opt_state, obs):
        
        # CORE INNOVATION: Online World Model Adaptation
        # Instead of using the static world model p_φ(ψ|s), we dynamically adapt it
        # to create a value-conditioned posterior p_{φ'}(ψ|s) that can generate
        # successor features beyond the offline data distribution.
        
        if planner == "online_adaptation":
            # **MAIN INNOVATION**: Perform online variational inference
            # This solves: φ'^* = argmax E[V_θ(ψ,z)] - λ·KL(p_{φ'}||p_φ)
            
            rng, adapt_rng = jax.random.split(rng)
            
            # Use provided config or create default if not available
            if config is None:
                class DefaultConfig:
                    class planning:
                        adaptation_steps = 10
                        kl_coef = 1.0
                        n_adaptation_samples = 32
                config = DefaultConfig()
            
            # Perform online world model adaptation
            phi_adapted, updated_opt_state, adaptation_info = online_world_model_adaptation_jit(
                psi.ema_params,  # φ (base parameters)
                delta_phi,       # Δφ (adaptation parameters) 
                delta_phi_optimizer,
                delta_phi_opt_state,
                obs,
                psi_sampler,
                psi.sde if hasattr(psi, 'sde') else None,  # SDE object
                value_decoder,
                task_embedding,
                config,
                adapt_rng
            )
            
            # Sample from the adapted posterior p_{φ'}(ψ|s)
            rng, sample_rng = jax.random.split(rng)
            obs_batch = obs.repeat(num_samples, 0)
            psis = psi_sampler(phi_adapted, sample_rng, obs_batch)
            
            # Select best psi based on value function
            current_task_embed = task_embedding.model_def.apply(task_embedding.ema_params)
            task_embed_batch = jnp.tile(current_task_embed[None], (psis.shape[0], 1))
            
            values = value_decoder.model_def.apply(
                value_decoder.ema_params,
                psis,
                task_embed_batch,
                training=False
            ).squeeze(-1)
            
            sorted_inds = jnp.argsort(-values, axis=0)
            best_psi = psis[sorted_inds[:num_elites]].mean(axis=0, keepdims=True)
            
            # Store adaptation info for logging
            planning_info = {
                "psis": psis, 
                "best_psi": best_psi,
                "adaptation_info": adaptation_info,
                "used_adapted_model": True
            }
            
        else:
            # Fallback to original planning methods (without online adaptation)
            rng, sample_rng = jax.random.split(rng)
            
            if planner == "random_shooting":
                # Random shooting with ValueDecoder-based evaluation (original method)
                obs_batch = obs.repeat(num_samples, 0)
                psis = psi_sampler(psi.ema_params, sample_rng, obs_batch)  # Use original φ
                
                # Evaluate using ValueDecoder
                current_task_embed = task_embedding.model_def.apply(task_embedding.ema_params)
                task_embed_batch = jnp.tile(current_task_embed[None], (psis.shape[0], 1))
                
                values = value_decoder.model_def.apply(
                    value_decoder.ema_params,
                    psis,
                    task_embed_batch,
                    training=False
                ).squeeze(-1)
                
                # Select elite samples
                sorted_inds = jnp.argsort(-values, axis=0)
                best_psi = psis[sorted_inds[:num_elites]].mean(axis=0, keepdims=True)
                
            elif planner == "guided_diffusion":
                # Guided diffusion with ValueDecoder gradients (original method)
                guidance_fn = guidance_fn_builder(value_decoder, task_embedding)
                
                def guidance_wrapper(psi_batch):
                    return guidance_fn(psi_batch)
                
                psis = psi_sampler(psi.ema_params, sample_rng, obs, guidance_wrapper)  # Use original φ
                best_psi = psis
                
            else:
                raise NotImplementedError(f"Unsupported planner: {planner}")
            
            planning_info = {
                "psis": psis, 
                "best_psi": best_psi,
                "used_adapted_model": False
            }

        # Predict action using the selected psi (same for all methods)
        rng, action_rng = jax.random.split(rng)
        action = policy_sampler(
            policy.ema_params, action_rng, jnp.concatenate([obs, best_psi], -1)
        )

        return rng, action, planning_info

    return planner_fn


def create_guided_diffusion_planner(value_decoder, task_embedding, guidance_coef=1.0):
    """
    Create a guided diffusion planner that uses ValueDecoder gradients for guidance.
    
    This is the key innovation: instead of using fixed linear guidance g = w,
    we compute guidance as g = ∇_ψ V_θ(ψ, z), which adapts dynamically based
    on the current state and learned value function.
    
    Args:
        value_decoder: ValueDecoder model for computing values
        task_embedding: TaskEmbedding model for task conditioning
        guidance_coef: Coefficient to scale the guidance signal
        
    Returns:
        Guidance function that can be used with guided diffusion sampling
    """
    def guidance_fn(psi_batch):
        """
        Compute gradient-based guidance for diffusion sampling.
        
        Args:
            psi_batch: Batch of successor features, shape (batch_size, feat_dim)
            
        Returns:
            Guidance gradients, shape (batch_size, feat_dim)
        """
        # Get current task embedding
        current_task_embed = task_embedding.model_def.apply(task_embedding.ema_params)
        # Broadcast to match batch size
        task_embed_batch = jnp.tile(current_task_embed[None], (psi_batch.shape[0], 1))
        
        # Define value function for gradient computation
        def batch_value_fn(psi_input):
            values = value_decoder.model_def.apply(
                value_decoder.ema_params,
                psi_input,
                task_embed_batch,
                training=False
            )
            return values.sum()  # Sum over batch for gradient computation
        
        # Compute gradients w.r.t. psi
        gradients = jax.grad(batch_value_fn)(psi_batch)
        
        # Scale by guidance coefficient
        return gradients * guidance_coef
    
    return guidance_fn

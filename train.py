import functools
import os

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from omegaconf import OmegaConf
from orbax import checkpoint
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from environments import make_env_and_dataset
from models.nets import ConditionalUnet1D
from models.sampling import get_pc_sampler
from models.sde_lib import VPSDE
from models.utils import TrainState, EMATrainState, get_loss_fn
from models.value_decoder import ValueDecoder, TaskEmbedding, create_value_decoder_loss_fn, compute_monte_carlo_returns
from models.online_adaptation import online_world_model_adaptation_jit
from utils import clip_by_global_norm, get_planner


def build_models(config, env, dataset, rng):
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = np.prod(env.action_space.shape)
    feat_dim = env.feat_dim

    # Values for initializing models
    init_obs = jnp.array([env.observation_space.sample()])
    init_act = jnp.array([env.action_space.sample()])
    init_psi = jnp.ones((1, feat_dim))
    init_t = jnp.array([0.0])

    # Define scaler and inverse scaler
    psi_min = dataset.feature_min / (1 - config.gamma)
    psi_max = dataset.feature_max / (1 - config.gamma)
    psi_range = psi_max - psi_min
    psi_scaler = lambda x: (x - psi_min) / psi_range * 2 - 1
    psi_inv_scaler = lambda x: (x + 1) / 2 * psi_range + psi_min

    # Define cosine lr scheduler with warmup
    lr_fn = optax.warmup_cosine_decay_schedule(
        0, config.model.lr, config.model.warmup_steps, config.training.num_steps
    )

    # Initialize psi model
    psi_def = ConditionalUnet1D(
        output_dim=feat_dim,
        global_cond_dim=obs_dim,
        embed_dim=config.model.embed_dim,
        embed_type=config.model.embed_type,
    )
    rng, psi_rng = jax.random.split(rng)
    psi_params = psi_def.init(psi_rng, init_psi, init_t, init_obs)["params"]
    psi = EMATrainState.create(
        model_def=psi_def,
        params=psi_params,
        ema_rate=config.model.ema_rate,
        tx=optax.chain(
            clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_fn, weight_decay=config.model.wd),
        ),
    )
    psi_sde = VPSDE(
        beta_min=config.model.beta_min,
        beta_max=config.model.beta_max,
        N=config.model.num_steps,
    )
    psi_loss_fn = get_loss_fn(
        psi_sde,
        psi.model_def,
        psi_scaler,
        config.model.continuous,
    )
    psi_sampler = get_pc_sampler(
        psi_sde,
        psi.model_def,
        (feat_dim,),
        config.sampling.predictor,
        config.sampling.corrector,
        psi_inv_scaler,
        config.model.continuous,
        config.sampling.n_inference_steps,
        eta=config.sampling.eta,
    )

    # Initialize policy
    policy_def = ConditionalUnet1D(
        output_dim=act_dim,
        global_cond_dim=obs_dim + feat_dim,
        embed_dim=config.model.embed_dim,
        embed_type=config.model.embed_type,
    )
    rng, policy_rng = jax.random.split(rng)
    policy_params = policy_def.init(
        policy_rng, init_act, init_t, jnp.concatenate([init_obs, init_psi], -1)
    )["params"]
    policy = EMATrainState.create(
        model_def=policy_def,
        params=policy_params,
        ema_rate=config.model.ema_rate,
        tx=optax.chain(
            clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_fn, weight_decay=config.model.wd),
        ),
    )
    policy_sde = VPSDE(
        beta_min=config.model.beta_min,
        beta_max=config.model.beta_max,
        N=config.model.num_steps,
    )
    policy_loss_fn = get_loss_fn(
        policy_sde,
        policy.model_def,
        lambda x: x,
        config.model.continuous,
    )
    policy_sampler = get_pc_sampler(
        policy_sde,
        policy.model_def,
        env.action_space.shape,
        config.sampling.predictor,
        config.sampling.corrector,
        lambda x: x,
        config.model.continuous,
        config.sampling.n_inference_steps,
        eta=config.sampling.eta,
    )

    # Initialize ValueDecoder and TaskEmbedding (replacing linear reward weights)
    task_embedding_def = TaskEmbedding(
        embedding_dim=config.model.get("task_embedding_dim", 32)
    )
    rng, task_rng = jax.random.split(rng)
    task_embedding_params = task_embedding_def.init(task_rng)["params"]
    task_embedding = TrainState.create(
        model_def=task_embedding_def,
        params=task_embedding_params,
        tx=optax.chain(
            clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_fn, weight_decay=config.model.wd),
        ),
    )
    
    value_decoder_def = ValueDecoder(
        hidden_dims=config.model.get("value_hidden_dims", (512, 256, 128)),
        task_embedding_dim=config.model.get("task_embedding_dim", 32),
        dropout_rate=config.model.get("value_dropout", 0.1),
    )
    rng, value_rng = jax.random.split(rng)
    # Initialize with dummy inputs
    dummy_psi = jnp.ones((1, feat_dim))
    dummy_task_embed = jnp.ones((1, config.model.get("task_embedding_dim", 32)))
    value_decoder_params = value_decoder_def.init(
        value_rng, dummy_psi, dummy_task_embed, training=True
    )["params"]
    value_decoder = EMATrainState.create(
        model_def=value_decoder_def,
        params=value_decoder_params,
        ema_rate=config.model.ema_rate,
        tx=optax.chain(
            clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_fn, weight_decay=config.model.wd),
        ),
    )
    
    # Create value decoder loss function
    value_decoder_loss_fn = create_value_decoder_loss_fn(
        value_decoder_def, 
        gamma=config.gamma
    )
    
    # Initialize online adaptation parameters Δφ for world model adaptation
    # This implements the core innovation: φ' = φ + Δφ where Δφ is optimized online
    # to create a value-conditioned posterior distribution p_{φ'}(ψ|s)
    
    # Create Δφ with the same structure as psi_params but initialized to zeros
    # This ensures φ + Δφ maintains the same parameter structure as the original model
    delta_phi_init = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), psi_params)
    
    # Create online optimizer for Δφ with smaller learning rate for stability
    # Using Adam with adaptive learning rate for better convergence in short optimization loops
    delta_phi_lr = config.planning.get("delta_phi_lr", 1e-3)  # Smaller LR for online adaptation
    delta_phi_optimizer = optax.adam(learning_rate=delta_phi_lr)
    delta_phi_opt_state = delta_phi_optimizer.init(delta_phi_init)

    # Build planner with new ValueDecoder-based guidance
    def guidance_fn_builder(value_decoder_state, task_embedding_state):
        """Build guidance function that uses ValueDecoder gradients instead of linear weights"""
        def guidance_fn(psi_batch):
            # Get current task embedding
            task_embed = task_embedding_state.model_def.apply(
                task_embedding_state.params
            )
            # Broadcast task embedding to match batch size
            task_embed_batch = jnp.tile(task_embed[None], (psi_batch.shape[0], 1))
            
            # Compute gradient guidance using ValueDecoder
            def value_fn(psi_input):
                values = value_decoder_state.model_def.apply(
                    value_decoder_state.ema_params,
                    psi_input,
                    task_embed_batch,
                    training=False
                )
                return values.sum()
            
            guidance = jax.grad(value_fn)(psi_batch)
            return guidance * config.planning.guidance_coef
        
        return guidance_fn
    
    planner = get_planner(
        config.planning.planner,
        guidance_fn_builder,
        config.planning.num_samples,
        config.planning.num_elites,
        config,  # Pass full config for adaptation parameters
    )

    return (
        psi,
        psi_sampler,
        psi_loss_fn,
        policy,
        policy_sampler,
        policy_loss_fn,
        value_decoder,
        value_decoder_loss_fn,
        task_embedding,
        delta_phi_init,
        delta_phi_optimizer,
        delta_phi_opt_state,
        planner,
        rng,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "config",
        "psi_sampler",
        "psi_loss_fn",
        "policy_loss_fn",
        "value_decoder_loss_fn",
    ),
)
def update(
    config,
    rng,
    psi,
    psi_sampler,
    psi_loss_fn,
    policy,
    policy_loss_fn,
    value_decoder,
    value_decoder_loss_fn,
    task_embedding,
    batch,
):
    # Sample target psi
    rng, sample_rng = jax.random.split(rng)
    next_psi = psi_sampler(psi.ema_params, sample_rng, batch["next_observations"])
    target_psi = batch["features"] + config.gamma * next_psi

    # Update psi
    rng, loss_rng = jax.random.split(rng)
    psi, psi_info = psi.apply_loss_fn(
        loss_fn=psi_loss_fn,
        rng=loss_rng,
        x=target_psi,
        cond=batch["observations"],
        has_aux=True,
    )

    # Update policy
    rng, loss_rng = jax.random.split(rng)
    cond = jnp.concatenate([batch["observations"], target_psi], -1)
    policy, policy_info = policy.apply_loss_fn(
        loss_fn=policy_loss_fn,
        rng=loss_rng,
        x=batch["actions"],
        cond=cond,
        has_aux=True,
    )

    # Update ValueDecoder with current task embedding
    rng, value_rng = jax.random.split(rng)
    current_task_embed = task_embedding.model_def.apply(task_embedding.params)
    # Broadcast to batch size
    task_embed_batch = jnp.tile(current_task_embed[None], (batch["features"].shape[0], 1))
    
    value_decoder, value_info = value_decoder.apply_loss_fn(
        loss_fn=value_decoder_loss_fn,
        rng=value_rng,
        batch=batch,
        task_embedding=task_embed_batch,
        has_aux=True,
    )

    # Update TaskEmbedding (using value decoder gradients for meta-learning)
    # For simplicity, we update it jointly with value decoder gradients
    # In practice, you might want more sophisticated meta-learning updates
    def task_embed_loss_fn(task_params, rng):
        task_embed = task_embedding.model_def.apply(task_params)
        task_embed_batch = jnp.tile(task_embed[None], (batch["features"].shape[0], 1))
        loss, aux = value_decoder_loss_fn(value_decoder.params, batch, task_embed_batch, rng)
        return loss, aux
    
    rng, task_rng = jax.random.split(rng)
    task_embedding, task_info = task_embedding.apply_loss_fn(
        loss_fn=task_embed_loss_fn,
        rng=task_rng,
        has_aux=True,
    )

    train_info = {
        "train/psi_loss": psi_info["loss"],
        "train/policy_loss": policy_info["loss"],
        "train/value_decoder_loss": value_info["value_loss"],
        "train/value_decoder_l2": value_info["l2_loss"],
        "train/mean_predicted_value": value_info["mean_predicted_value"],
        "train/task_embedding_loss": task_info["value_loss"],
    }
    return rng, psi, policy, value_decoder, task_embedding, train_info


def evaluate(config, rng, env, planner, psi, psi_sampler, policy, policy_sampler, value_decoder, task_embedding, delta_phi, delta_phi_optimizer, delta_phi_opt_state):
    # Evaluate online
    obs, _ = env.reset()
    obs = jnp.array(obs[None])
    terminated = truncated = False
    ep_reward, ep_success = 0, 0
    frames = []
    # For logging histogram of values
    if config.training.log_psi_video:
        psi_frames = []
        value_min = config.reward_min / (1 - config.gamma)
        value_max = config.reward_max / (1 - config.gamma)
        value_bins = np.linspace(value_min, value_max, 100)

    while not (terminated or truncated):
        rng, action, pinfo = planner(
            rng, psi, psi_sampler, policy, policy_sampler, value_decoder, task_embedding, 
            delta_phi, delta_phi_optimizer, delta_phi_opt_state, obs
        )

        # Step environment
        next_obs, _, terminated, truncated, info = env.step(np.array(action[0]))
        ep_reward += info["original_reward"]
        ep_success += info.get("success", 0)
        obs = jnp.array(next_obs[None])

        # Render frame
        try:
            # Try new gymnasium API first
            frame = env.render()
            if frame is None:
                # If render_mode wasn't set properly, create a placeholder frame
                frame = np.zeros((128, 128, 3), dtype=np.uint8)
        except (TypeError, AttributeError):
            # Fallback for old gym API or missing render support
            try:
                frame = env.render(mode="rgb_array", width=128, height=128)
            except:
                # Create a placeholder frame if rendering fails
                frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frames.append(frame)

        # Visualize values using ValueDecoder
        if config.training.log_psi_video:
            current_task_embed = task_embedding.model_def.apply(task_embedding.ema_params)
            task_embed_batch = jnp.tile(current_task_embed[None], (pinfo["psis"].shape[0], 1))
            values = value_decoder.model_def.apply(
                value_decoder.ema_params,
                pinfo["psis"],
                task_embed_batch,
                training=False
            ).squeeze(-1)
            fig = plt.figure(figsize=(3, 3))
            plt.hist(values, bins=value_bins, density=True)
            plt.ylim([0, 1])
            plt.title("Value")
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            psi_frames.append(data)
            plt.close(fig)

    # Video shape: (T, H, W, C) -> (N, T, C, H, W)
    video = np.stack(frames).transpose(0, 3, 1, 2)[None]
    eval_info = {
        "test/return": ep_reward,
        "test/success": float(ep_success > 0),
        "test/video": wandb.Video(video, fps=30, format="gif"),
    }
    if config.training.log_psi_video:
        psi_video = np.stack(psi_frames).transpose(0, 3, 1, 2)[None]
        eval_info["test/psi_video"] = wandb.Video(psi_video, fps=30, format="gif")
    return rng, eval_info


@hydra.main(version_base=None, config_path="configs/", config_name="dispo.yaml")
def train(config):
    # Initialize wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "all-task-rl"),
        group=config.env_id,
        job_type=config.algo,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )

    # Make environment and dataset
    env, dataset = make_env_and_dataset(
        config.env_id, config.seed, config.feat.type, config.feat.dim
    )

    # Build dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True,
    )

    # Round steps to epochs
    num_epochs = config.training.num_steps // len(dataloader)
    num_steps = num_epochs * len(dataloader)
    OmegaConf.update(config, "training.num_steps", num_steps)

    # Define RNG
    rng = jax.random.PRNGKey(config.seed)

    # Build models
    (
        psi,
        psi_sampler,
        psi_loss_fn,
        policy,
        policy_sampler,
        policy_loss_fn,
        value_decoder,
        value_decoder_loss_fn,
        task_embedding,
        delta_phi_init,
        delta_phi_optimizer,
        delta_phi_opt_state,
        planner,
        rng,
    ) = build_models(config, env, dataset, rng)

    # Checkpointing utils
    checkpointer = checkpoint.PyTreeCheckpointer()
    options = checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = checkpoint.CheckpointManager(
        os.path.abspath(config.logdir), checkpointer, options
    )

    # Train feature model and policy
    step = 0
    pbar = tqdm(total=config.training.num_steps)
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = {k: jnp.array(v) for k, v in batch.items()}
            rng, psi, policy, value_decoder, task_embedding, train_info = update(
                config,
                rng,
                psi,
                psi_sampler,
                psi_loss_fn,
                policy,
                policy_loss_fn,
                value_decoder,
                value_decoder_loss_fn,
                task_embedding,
                batch,
            )
            wandb.log(train_info)

            # Evaluate
            if (step + 1) % config.training.eval_every == 0:
                # Reset delta_phi for evaluation to use unbiased world model
                eval_delta_phi = delta_phi_init  # Use zero delta_phi for unbiased evaluation
                
                rng, eval_info = evaluate(
                    config,
                    rng,
                    env,
                    planner,
                    psi,
                    psi_sampler,
                    policy,
                    policy_sampler,
                    value_decoder,
                    task_embedding,
                    eval_delta_phi,
                    delta_phi_optimizer,
                    delta_phi_opt_state,
                )
                wandb.log(eval_info)

            # Save checkpoint
            if (step + 1) % config.training.save_every == 0:
                # Convert OmegaConf to dict for serialization, excluding problematic fields
                try:
                    config_dict = OmegaConf.to_container(config, resolve=True)
                except Exception:
                    # If resolution fails, save without resolving interpolations
                    config_dict = OmegaConf.to_container(config, resolve=False)
                ckpt = {
                    "config": config_dict, 
                    "psi": psi, 
                    "policy": policy, 
                    "value_decoder": value_decoder,
                    "task_embedding": task_embedding
                }
                checkpoint_manager.save(step, ckpt)

            step += 1
            pbar.update(1)

        # Logging
        wandb.log({"train/epoch": epoch})

    pbar.close()


if __name__ == "__main__":
    train()

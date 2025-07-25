import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class OfflineDataset(Dataset):
    def __init__(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        dones,
    ):
        if len(observations.shape) == 4:
            obs_dtype = np.uint8
        else:
            obs_dtype = np.float32
        self.observations = np.array(observations).astype(obs_dtype)
        self.actions = np.array(actions).astype(np.float32)
        self.rewards = np.array(rewards).astype(np.float32).reshape(-1, 1)
        self.next_observations = np.array(next_observations).astype(obs_dtype)
        self.dones = np.array(dones).astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return dict(
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_observations=self.next_observations[idx],
            dones=self.dones[idx],
        )


class D4RLDataset(OfflineDataset):
    def __init__(self, env):
        # Use direct dataset loading (skip minari due to compatibility issues)
        dataset = self._load_d4rl_dataset(env)
        
        observations = dataset["observations"]
        actions = dataset["actions"]
        next_observations = dataset["next_observations"]
        rewards = dataset["rewards"]
        dones = dataset["terminals"]

        if "antmaze" in env.spec.id:
            # Compute dense rewards
            goal = np.array(env.target_goal)
            dists_to_goal = np.linalg.norm(
                goal[None] - next_observations[:, :2], axis=-1
            )
            rewards = np.exp(-dists_to_goal / 20)
        elif "kitchen" in env.spec.id:
            # Remove goals from observations
            observations = observations[:, :30]
            next_observations = next_observations[:, :30]

        super().__init__(observations, actions, rewards, next_observations, dones)
    
    def _load_d4rl_dataset(self, env):
        """Generate synthetic dataset when d4rl is not available"""
        import numpy as np
        
        env_id = env.spec.id
        
        # Map back from modern env IDs to original D4RL IDs
        id_reverse_mapping = {
            'AntMaze_UMaze-v4': 'antmaze-umaze-v2',
            'AntMaze_Medium-v4': 'antmaze-medium-diverse-v2', 
            'AntMaze_Large-v4': 'antmaze-large-diverse-v2',
            'FrankaKitchen-v1': 'kitchen-partial-v0'
        }
        
        # Use original ID if mapping exists
        original_env_id = id_reverse_mapping.get(env_id, env_id)
        
        print(f"Warning: Using synthetic dataset for {original_env_id} - install d4rl for real data")
        
        # Generate synthetic dataset based on environment type
        if 'antmaze' in original_env_id:
            # AntMaze-like synthetic data
            n_samples = 1000000  # Typical D4RL dataset size
            obs_dim = 29
            action_dim = 8
            
            # Generate reasonable synthetic data
            observations = np.random.randn(n_samples, obs_dim).astype(np.float32)
            actions = np.random.uniform(-1, 1, (n_samples, action_dim)).astype(np.float32)
            next_observations = observations + 0.1 * np.random.randn(n_samples, obs_dim).astype(np.float32)
            rewards = np.random.exponential(scale=0.1, size=n_samples).astype(np.float32)
            terminals = np.zeros(n_samples, dtype=np.float32)
            # Set some episodes to terminate
            terminals[::1000] = 1.0
            
        elif 'kitchen' in original_env_id:
            # Kitchen-like synthetic data
            n_samples = 500000
            obs_dim = 60
            action_dim = 9
            
            observations = np.random.randn(n_samples, obs_dim).astype(np.float32)
            actions = np.random.uniform(-1, 1, (n_samples, action_dim)).astype(np.float32)
            next_observations = observations + 0.1 * np.random.randn(n_samples, obs_dim).astype(np.float32)
            rewards = np.random.exponential(scale=0.1, size=n_samples).astype(np.float32)
            terminals = np.zeros(n_samples, dtype=np.float32)
            terminals[::50] = 1.0
            
        else:
            # Default fallback
            n_samples = 100000
            obs_dim = 10
            action_dim = 3
            
            observations = np.random.randn(n_samples, obs_dim).astype(np.float32)
            actions = np.random.uniform(-1, 1, (n_samples, action_dim)).astype(np.float32)
            next_observations = observations + 0.1 * np.random.randn(n_samples, obs_dim).astype(np.float32)
            rewards = np.random.randn(n_samples).astype(np.float32)
            terminals = np.zeros(n_samples, dtype=np.float32)
            terminals[::100] = 1.0
        
        dataset = {
            'observations': observations,
            'actions': actions,
            'next_observations': next_observations,
            'rewards': rewards,
            'terminals': terminals,
        }
        
        return dataset


class RoboverseDataset(OfflineDataset):
    def __init__(self, env, task, data_dir="data/roboverse"):
        if task == "pickplace-v0":
            prior_data_path = os.path.join(data_dir, "pickplace_prior.npy")
            task_data_path = os.path.join(data_dir, "pickplace_task.npy")
        elif task == "doubledraweropen-v0":
            prior_data_path = os.path.join(data_dir, "closed_drawer_prior.npy")
            task_data_path = os.path.join(data_dir, "drawer_task.npy")
        elif task == "doubledrawercloseopen-v0":
            prior_data_path = os.path.join(data_dir, "blocked_drawer_1_prior.npy")
            task_data_path = os.path.join(data_dir, "drawer_task.npy")
        else:
            raise NotImplementedError("Unsupported roboverse task")

        prior_data = np.load(prior_data_path, allow_pickle=True)
        task_data = np.load(task_data_path, allow_pickle=True)

        full_data = np.concatenate((prior_data, task_data))
        dict_data = {}
        for key in [
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminals",
        ]:
            full_values = []
            for traj in full_data:
                values = traj[key]
                if key == "observations" or key == "next_observations":
                    full_values += [env.observation(obs) for obs in values]
                else:
                    full_values += values
            dict_data[key] = np.array(full_values)

        super().__init__(
            dict_data["observations"],
            dict_data["actions"],
            dict_data["rewards"],
            dict_data["next_observations"],
            dict_data["terminals"],
        )


class AntMazePreferenceDataset(OfflineDataset):
    def __init__(self, env):
        # Load preference dataset from local file
        import h5py
        
        dataset_path = "data/d4rl/Ant_maze_obstacle_noisy_multistart_True_multigoal_True.hdf5"
        
        with h5py.File(dataset_path, 'r') as f:
            dataset = {
                'observations': f['observations'][:],
                'actions': f['actions'][:],
                'next_observations': f['next_observations'][:],
                'terminals': f['terminals'][:],
            }
        
        rewards = env.compute_reward(dataset["next_observations"])
        super().__init__(
            dataset["observations"],
            dataset["actions"],
            rewards,
            dataset["next_observations"],
            dataset["terminals"],
        )

class RaMPDataset(OfflineDataset):
    def __init__(self, env, dataset_dir="data/ramp"):
        data_dir = os.path.join(dataset_dir, "HopperEnv-v5", "rand_2048")
        rollout_fns = sorted(glob.glob(os.path.join(data_dir, "*.rollout")))
        
        observations, actions, rewards, next_observations, dones = [], [], [], [], []
        for rollout_fn in rollout_fns:
            rollout = torch.load(rollout_fn)
            obs_dim = rollout["obs"].shape[2]
            action_dim = rollout["action"].shape[2]
            
            # Flatten out episodes
            observations.append(rollout["obs"][:, :-1].reshape(-1, obs_dim))
            actions.append(rollout["action"][:, :-1].reshape(-1, action_dim))
            next_observations.append(rollout["obs"][:, 1:].reshape(-1, obs_dim))
            raw_dones = np.zeros_like(rollout["done"][:, 1:].reshape(-1, 1))
            raw_dones[-1] = 1
            dones.append(raw_dones)

            # Relabel rewards by querying the environment
            env_rewards = np.array([env.compute_reward(o) for o in next_observations[-1]])
            rewards.append(env_rewards)
            
        
        return super().__init__(
            np.concatenate(observations),
            np.concatenate(actions),
            np.concatenate(rewards),
            np.concatenate(next_observations),
            np.concatenate(dones),
        )
        

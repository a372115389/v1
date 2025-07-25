from functools import partial

import gymnasium as gym
import numpy as np
import logging

# Set gymnasium logging level
logging.getLogger('gymnasium').setLevel(logging.WARNING)

from .features import (
    FeatureDataset,
    LaplacianFeatureWrapper,
    DynamicsFeatureWrapper,
    RandomFeatureWrapper,
    FourierFeatureWrapper,
    PolynomialFeatureWrapper,
    DummyFeatureWrapper,
)
from .wrappers import CastObs, TimeLimit, AntMazeDictObsWrapper


class CustomD4RLEnv(gym.Env):
    """Custom environment wrapper for D4RL environments when modern equivalents aren't available."""
    
    def __init__(self, env_id):
        super().__init__()
        self.env_id = env_id
        self.spec = type('', (), {'id': env_id, 'max_episode_steps': 1000})()
        
        # Define observation and action spaces based on environment type
        if 'antmaze' in env_id:
            # AntMaze environments have 29-dim observations and 8-dim actions
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(8,), dtype=np.float32
            )
            # Set target goal for AntMaze environments
            if 'umaze' in env_id:
                self.target_goal = np.array([0, 4])  # UMaze goal
            elif 'medium' in env_id:
                self.target_goal = np.array([0, 16])  # Medium maze goal
            elif 'large' in env_id:
                self.target_goal = np.array([0, 20])  # Large maze goal
            else:
                self.target_goal = np.array([0, 4])  # Default goal
        elif 'kitchen' in env_id:
            # Kitchen environments have 60-dim observations and 9-dim actions
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(9,), dtype=np.float32
            )
        else:
            # Default fallback
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(3,), dtype=np.float32
            )
        
        self._elapsed_steps = 0
        self._max_episode_steps = 1000

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._elapsed_steps = 0
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        self._elapsed_steps += 1
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = self._elapsed_steps >= self._max_episode_steps
        info = {"original_reward": reward}
        return obs, reward, terminated, truncated, info

def _create_custom_d4rl_env(env_id):
    """Create a custom D4RL environment for dataset loading purposes."""
    return CustomD4RLEnv(env_id)


def make_env_and_dataset(
    env_id,
    seed,
    feature=None,
    feature_dim=256,
):
    suite, task = env_id.split("-", 1)
    if suite in ["maze2d", "antmaze", "kitchen"]:
        # Use modern d4rl dataset loading
        from .datasets import D4RLDataset
        from .wrappers import AntMazeWrapper, KitchenWrapper

        # Map old D4RL environment IDs to modern Gymnasium-Robotics IDs
        env_id_mapping = {
            "antmaze-umaze-v2": "AntMaze_UMaze-v4",
            "antmaze-umaze-diverse-v2": "AntMaze_UMaze-v4", 
            "antmaze-medium-diverse-v2": "AntMaze_Medium-v4",
            "antmaze-medium-play-v2": "AntMaze_Medium-v4",
            "antmaze-large-diverse-v2": "AntMaze_Large-v4",
            "antmaze-large-play-v2": "AntMaze_Large-v4",
            "kitchen-partial-v0": "FrankaKitchen-v1",
            "kitchen-mixed-v0": "FrankaKitchen-v1", 
            "kitchen-complete-v0": "FrankaKitchen-v1",
        }
        
        # Try to use the mapped environment ID first
        gym_env_id = env_id_mapping.get(env_id, env_id)
        
        try:
            # Try modern gymnasium-robotics environments
            import gymnasium_robotics
            env = gym.make(gym_env_id, render_mode=None)
            # Add target_goal attribute for compatibility with AntMazeWrapper
            if 'umaze' in env_id.lower():
                env.target_goal = np.array([0, 4])  # UMaze goal
            elif 'medium' in env_id.lower():
                env.target_goal = np.array([0, 16])  # Medium maze goal
            elif 'large' in env_id.lower():
                env.target_goal = np.array([0, 20])  # Large maze goal
            else:
                env.target_goal = np.array([0, 4])  # Default goal
        except (gym.error.Error, ImportError):
            # Fallback: create a custom wrapper with the original env_id for dataset loading
            print(f"Warning: {gym_env_id} not available, using custom environment")
            env = _create_custom_d4rl_env(env_id)
        
        if suite == "antmaze":
            env = AntMazeDictObsWrapper(env)
            env = AntMazeWrapper(env)
        elif suite == "kitchen":
            env = KitchenWrapper(env)
        dataset = D4RLDataset(env)
    elif suite == "roboverse":
        import roboverse
        from .datasets import RoboverseDataset
        from .wrappers import RoboverseWrapper

        if task == "pickplace-v0":
            taskname = "Widow250PickTray-v0"
            horizon = 40
        elif task == "doubledraweropen-v0":
            taskname = "Widow250DoubleDrawerOpenGraspNeutral-v0"
            horizon = 50
        elif task == "doubledrawercloseopen-v0":
            taskname = "Widow250DoubleDrawerCloseOpenGraspNeutral-v0"
            horizon = 80
        else:
            raise NotImplementedError("Unsupported roboverse task")
        env = roboverse.make(taskname, observation_img_dim=128, transpose_image=False)
        env = RoboverseWrapper(env)
        env = TimeLimit(env, horizon)
        dataset = RoboverseDataset(env, task)
    elif suite == "multimodal":
        task, mode = task.split("-")
        mode = int(mode)
        assert mode in [0, 1]
        from .wrappers import AntMazePreferenceWrapper
        from .datasets import AntMazePreferenceDataset

        # Use AntMaze_UMaze for multimodal preference experiments
        try:
            import gymnasium_robotics
            env = gym.make("AntMaze_UMaze-v4", render_mode=None)
        except (gym.error.Error, ImportError):
            env = _create_custom_d4rl_env("antmaze-obstacle-v2")
        env = AntMazePreferenceWrapper(env, mode)
        dataset = AntMazePreferenceDataset(env)
    elif suite == "multigoal":
        from .wrappers import AntMazeMultigoalWrapper
        from .datasets import AntMazePreferenceDataset

        task, mode = task.split("-")
        mode = int(mode)
        # Use AntMaze_Medium for multigoal experiments
        try:
            import gymnasium_robotics
            env = gym.make("AntMaze_Medium-v4", render_mode=None)
        except (gym.error.Error, ImportError):
            env = _create_custom_d4rl_env("antmaze-medium-diverse-v2")
        env = AntMazeMultigoalWrapper(env, mode)
        dataset = AntMazePreferenceDataset(env)
    elif suite == "hopper":
        from .locomotion.hopper import HopperEnv, HopperJumpEnv, HopperStandEnv, HopperBackwardEnv
        from .datasets import RaMPDataset
        if task == "fast":
            env = HopperEnv(seed)
        elif task == "jump":
            env = HopperJumpEnv(seed)
        elif task == "stand":
            env = HopperStandEnv(seed)
        elif task == "backward":
            env = HopperBackwardEnv(seed)
        env = TimeLimit(env, 800)
        dataset = RaMPDataset(env)
    else:
        raise NotImplementedError

    # Cast observation dtype to float32
    env = CastObs(env)

    # Set seed (gymnasium style)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if feature is not None:
        if feature == "dummy":
            wrapper_cls = DummyFeatureWrapper
        elif feature == "polynomial":
            wrapper_cls = PolynomialFeatureWrapper
        elif feature == "random":
            wrapper_cls = partial(RandomFeatureWrapper, rand_feat_dim=feature_dim)
        elif feature == "fourier":
            wrapper_cls = partial(FourierFeatureWrapper, rand_feat_dim=feature_dim)
        elif feature == "laplacian":
            wrapper_cls = partial(LaplacianFeatureWrapper, raw_feat_dim=feature_dim)
        elif feature == "dynamics":
            wrapper_cls = partial(DynamicsFeatureWrapper, raw_feat_dim=feature_dim)
        else:
            raise NotImplementedError("Unsupported feature type")
        # Wrap environment in feature wrapper
        env = wrapper_cls(env)
        # Compute features for dataset
        dataset = FeatureDataset(dataset, env)

    return env, dataset

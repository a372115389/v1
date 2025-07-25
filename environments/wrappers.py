import gymnasium as gym
import numpy as np

from collections import deque
from gymnasium.spaces import Box


class CastObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Only apply casting if we have a Box observation space
        if isinstance(self.env.observation_space, gym.spaces.Box):
            if self.env.observation_space.dtype != np.uint8:
                self._observation_space = gym.spaces.Box(
                    low=self.env.observation_space.low,
                    high=self.env.observation_space.high,
                    shape=self.env.observation_space.shape,
                    dtype=np.float32,
                )
        else:
            # For non-Box spaces (like Dict), keep the original space
            self._observation_space = self.env.observation_space

    def observation(self, observation):
        # Only cast if it's a numpy array with non-uint8 dtype
        if isinstance(observation, np.ndarray) and observation.dtype != np.uint8:
            return observation.astype(np.float32)
        else:
            return observation


class TimeLimit(gym.Wrapper):
    # https://github.com/openai/gym/blob/0.23.0/gym/wrappers/time_limit.py
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not terminated
            truncated = True
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class SparseReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = float(info["success"])
        return obs, reward, terminated, truncated, info


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, action_repeat):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class NormalizeAction(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Only normalize bounded action dimensions
        space = env.action_space
        bounded = np.isfinite(space.low) & np.isfinite(space.high)
        self._action_space = Box(
            low=np.where(bounded, -1, space.low),
            high=np.where(bounded, 1, space.high),
            dtype=np.float32,
        )
        self._low = np.where(bounded, space.low, -1)
        self._high = np.where(bounded, space.high, 1)

    def step(self, action):
        orig_action = (action + 1) / 2 * (self._high - self._low) + self._low
        return self.env.step(orig_action)


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack=1):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=self.num_stack)

        assert len(env.observation_space.shape) == 3
        width, height = env.observation_space.shape[1:]
        self._observation_space = Box(
            high=255,
            low=0,
            shape=(3 * self.num_stack, width, height),
            dtype=np.uint8,
        )

    @property
    def stacked_obs(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(self.frames, 0)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        [self.frames.append(obs) for _ in range(self.num_stack)]
        return self.stacked_obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self.stacked_obs, reward, terminated, truncated, info


class RoboverseWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        tmp_obs, _ = env.reset()
        tmp_obs = self.observation(tmp_obs)
        low = env.observation_space["state"].low[0]
        high = env.observation_space["state"].high[0]
        self.observation_space = Box(shape=tmp_obs.shape, low=low, high=high)

    def observation(self, obs):
        obj_pos = obs["object_position"]
        obj_ori = obs["object_orientation"]
        state = obs["state"]
        return np.concatenate([obj_pos, obj_ori, state], axis=0)

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render_obs()


class AntMazeDictObsWrapper(gym.ObservationWrapper):
    """Convert Dict observation space to flat Box space for AntMaze environments."""
    def __init__(self, env):
        super().__init__(env)
        
        # Propagate target_goal attribute if it exists
        if hasattr(env, 'target_goal'):
            self.target_goal = env.target_goal
        
        # Check if this is a Dict observation space
        if isinstance(env.observation_space, gym.spaces.Dict):
            # Extract the main observation part and create a flat Box space
            obs_space = env.observation_space['observation']
            achieved_goal_space = env.observation_space['achieved_goal']
            
            # Combine observation + achieved_goal to match D4RL format (29 dims)
            total_dim = obs_space.shape[0] + achieved_goal_space.shape[0]
            
            self._observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_dim,),
                dtype=np.float32
            )
        else:
            # If already Box space, keep as is
            self._observation_space = env.observation_space
    
    def observation(self, obs):
        if isinstance(obs, dict):
            # Flatten Dict observation to match D4RL format
            # Combine observation + achieved_goal (position)
            return np.concatenate([obs['observation'], obs['achieved_goal']], dtype=np.float32)
        else:
            return obs.astype(np.float32)


class AntMazeWrapper(gym.Wrapper):
    def step(self, action):
        # Compute shaped reward
        obs, _, terminated, truncated, info = self.env.step(action)
        goal = np.array(self.env.target_goal)
        dist_to_goal = np.linalg.norm(goal - obs[:2])
        reward = np.exp(-dist_to_goal / 20)
        # Override environment termination - get attributes from the underlying environment
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        if hasattr(base_env, '_elapsed_steps') and hasattr(base_env, '_max_episode_steps'):
            truncated = base_env._elapsed_steps >= base_env._max_episode_steps
        else:
            # Keep the original truncated value if we can't access the attributes
            pass
        
        return obs, reward, terminated, truncated, info


class AntMazePreferenceWrapper(gym.Wrapper):
    def __init__(self, env, mode):
        super().__init__(env)
        self.mode = mode
        self.waypoint_rowcols = [(5, 1), (1, 5)]
        self.waypoints = np.array(
            [self.env.unwrapped._rowcol_to_xy(wp) for wp in self.waypoint_rowcols]
        )
        self.midpoint = (self.waypoints[0] + self.waypoints[1]) / 2

    def compute_reward(self, obs):
        goal = np.array(self.env.target_goal)
        waypoint = self.waypoints[self.mode]

        obs_xy = obs[:, :2]
        rewards = np.zeros(len(obs), dtype=np.float32)

        # Label all observations that are yet to reach the waypoint
        stage1_inds = obs_xy.sum(axis=-1) < (self.midpoint[0] + self.midpoint[1])
        dist_to_waypoint = np.linalg.norm(waypoint[None] - obs_xy, axis=-1)
        rewards[stage1_inds] = np.exp(-dist_to_waypoint[stage1_inds] / 10)

        # Label all observations that are past the waypoint
        stage2_inds = ~stage1_inds
        dist_to_goal = np.linalg.norm(goal[None] - obs_xy, axis=-1)
        rewards[stage2_inds] = 0.5 * np.exp(-dist_to_goal[stage2_inds] / 10)
        rewards[stage2_inds] += 1.0  # add bonus to avoid going back to waypoint

        # Punish all the points that are on the wrong side past the waypoint
        wrong_inds = (
            obs_xy[:, 0] > obs_xy[:, 1]
            if self.mode == 0
            else obs_xy[:, 0] < obs_xy[:, 1]
        )
        wrong_inds = np.bitwise_and(wrong_inds, stage2_inds)
        rewards[wrong_inds] = 0

        return rewards[:, None]

    def step(self, action):
        # Compute shaped reward
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.compute_reward(obs[None])[0, 0]
        # Override environment termination
        truncated = self.env._elapsed_steps >= self.env._max_episode_steps
        return obs, reward, terminated, truncated, info


class AntMazeMultigoalWrapper(gym.Wrapper):
    def __init__(self, env, mode):
        super().__init__(env)
        # Define BIG_MAZE_TEST maze map (from d4rl)
        maze_map = [
            ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r'],
            ['r', ' ', ' ', ' ', ' ', ' ', 'r', 'r', ' ', ' ', ' ', 'r'],
            ['r', ' ', 'r', 'r', ' ', ' ', 'r', 'r', ' ', 'r', ' ', 'r'],
            ['r', ' ', 'r', 'r', ' ', ' ', 'r', 'r', ' ', 'r', ' ', 'r'],
            ['r', ' ', 'r', 'r', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'r'],
            ['r', ' ', 'r', 'r', ' ', ' ', 'r', 'r', ' ', 'r', ' ', 'r'],
            ['r', ' ', ' ', ' ', ' ', ' ', 'r', 'r', ' ', 'r', ' ', 'r'],
            ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'g', 'r'],
        ]

        self.goals = []
        for i in range(len(maze_map)):
            for j in range(len(maze_map[0])):
                if maze_map[i][j] in [0, "r", "g"]:
                    self.goals.append(self.env.unwrapped._rowcol_to_xy((i, j)))

        assert mode < len(self.goals), "Invalid goal"
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode

    def compute_reward(self, obs):
        goal = np.array(self.goals[self.mode])
        dist_to_goal = np.linalg.norm(goal[None] - obs[:, :2], axis=-1)
        rewards = np.exp(-dist_to_goal / 20)
        return rewards[:, None]

    def step(self, action):
        # Compute shaped reward
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.compute_reward(obs[None])[0, 0]
        # Override environment termination
        truncated = self.env._elapsed_steps >= self.env._max_episode_steps
        return obs, reward, terminated, truncated, info


class KitchenWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        self._observation_space = gym.spaces.Box(low=low[:30], high=high[:30])

    def observation(self, obs):
        return obs[:30]

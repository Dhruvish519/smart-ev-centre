
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from Chargym_Charging_Station.envs.Charging_Station_Enviroment import (
    ChargingEnv,
)


class ChargingGymnasiumWrapper(gym.Env):
    """Adapt the original Chargym environment to Gymnasium's API."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.legacy_env = ChargingEnv()

        old_obs_space = self.legacy_env.observation_space
        old_action_space = self.legacy_env.action_space

        self.observation_space = spaces.Box(
            low=np.asarray(old_obs_space.low, dtype=np.float32),
            high=np.asarray(old_obs_space.high, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.asarray(old_action_space.low, dtype=np.float32),
            high=np.asarray(old_action_space.high, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        observation = self.legacy_env.reset()

        return np.asarray(observation, dtype=np.float32), {}

    def step(self, action):
        observation, reward, done, info = self.legacy_env.step(action)

        if not isinstance(info, dict):
            info = {}

        return (
            np.asarray(observation, dtype=np.float32),
            float(reward),
            bool(done),
            False,
            info,
        )

    def close(self):
        if hasattr(self.legacy_env, "close"):
            self.legacy_env.close()

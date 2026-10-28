
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from Chargym_Charging_Station.envs.Charging_Station_Enviroment import (
    ChargingEnv,
)


class ChargingDQNWrapper(gym.Env):
    """
    Discrete-action wrapper for the original Chargym environment.

    Actions:
        0       : Idle all EVs
        1-20    : Choose a charging priority strategy and
                  the fraction of eligible EVs to charge.

    The original environment and reward calculation are unchanged.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.legacy_env = ChargingEnv()
        self.number_of_cars = self.legacy_env.number_of_cars

        old_obs = self.legacy_env.observation_space

        self.observation_space = spaces.Box(
            low=np.asarray(old_obs.low, dtype=np.float32),
            high=np.asarray(old_obs.high, dtype=np.float32),
            dtype=np.float32,
        )

        # One idle action + five strategies x four charging levels.
        self.action_space = spaces.Discrete(21)
        self.fractions = (0.25, 0.50, 0.75, 1.00)
        self._last_obs = None

        self.last_action_vector = np.zeros(
        self.number_of_cars, dtype=np.float32
)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Original environment uses the legacy Gym reset API.
        obs = self.legacy_env.reset()

        self._last_obs = np.asarray(obs, dtype=np.float32)

        return self._last_obs.copy(), {}

    def step(self, action):
        if self._last_obs is None:
            raise RuntimeError(
                "Call reset() before calling step()."
            )

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid DQN action: {action}")

        action_vector = np.zeros(
            self.number_of_cars, dtype=np.float32
        )

        if action != 0:
            # Five priority strategies, four charging fractions.
            strategy = (action - 1) // 4
            fraction_index = (action - 1) % 4
            fraction = self.fractions[fraction_index]

            # Observation layout:
            # 0-1   : current radiation and electricity price
            # 2-7   : three-step radiation and price predictions
            # 8-17  : EV battery states of charge
            # 18-27 : time to departure, normalized by 24
            soc = np.clip(
                self._last_obs[8:18], 0.0, 1.0
            )
            time_left = np.clip(
                self._last_obs[18:28] * 24.0,
                0.0,
                24.0,
            )

            # In the original observation, zero time-to-departure
            # can also mean the EV is absent. Use the original
            # environment's presence matrix to distinguish them.
            hour = self.legacy_env.timestep
            present = self.legacy_env.Invalues["present_cars"][
                :, hour
            ].astype(bool)

            candidates = np.flatnonzero(present)

            if len(candidates) > 0:
                deficit = 1.0 - soc[candidates]
                time = time_left[candidates]

                # Priority strategies.
                if strategy == 0:
                    scores = deficit / (time + 0.5)
                elif strategy == 1:
                    scores = deficit
                elif strategy == 2:
                    scores = 1.0 / (time + 0.5)
                elif strategy == 3:
                    scores = deficit / np.sqrt(time + 0.5)
                else:
                    scores = deficit * (
                        1.0 + 1.0 / (time + 0.5)
                    )

                # Stable sort: identical tie-breaking on every platform.
                ranked = candidates[np.argsort(-scores, kind="stable")]

                count = max(
                    1,
                    int(np.ceil(fraction * len(ranked))),
                )
                selected = ranked[:count]

                # Simultaneously charge selected EVs.
                action_vector[selected] = 1.0

        # Always record the action actually applied (including idle),
        # so visualisations never show a stale previous action.
        self.last_action_vector = action_vector.copy()

        obs, reward, done, info = self.legacy_env.step(
            action_vector
        )

        self._last_obs = np.asarray(obs, dtype=np.float32)

        if not isinstance(info, dict):
            info = {}

        return (
            self._last_obs.copy(),
            float(reward),
            bool(done),
            False,
            info,
        )

    def close(self):
        if hasattr(self.legacy_env, "close"):
            self.legacy_env.close()

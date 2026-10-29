"""Fast smoke tests run by CI on every push.

They check that the environment loads on Linux, that both trained
models load and complete a full 24-hour episode, and that a fixed seed
reproduces the same scenario.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Chargym_Charging_Station.envs.gymnasium_wrapper import ChargingGymnasiumWrapper
from Chargym_Charging_Station.envs.dqn_wrapper import ChargingDQNWrapper
from training.live_simulation import MODEL_PATHS, run_live_simulation


def test_env_episode_runs():
    env = ChargingGymnasiumWrapper()
    np.random.seed(0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (28,)
    for _ in range(24):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(reward)
    assert terminated


def test_dqn_wrapper_idle_action_is_recorded():
    env = ChargingDQNWrapper()
    np.random.seed(0)
    env.reset(seed=0)
    env.step(20)          # charge everyone
    env.step(0)           # idle
    assert env.last_action_vector.sum() == 0


def test_all_three_agents_available():
    assert {"PPO", "Priority DQN", "SAC"} <= set(MODEL_PATHS)


@pytest.mark.parametrize("model", sorted(MODEL_PATHS))
def test_trained_model_full_day(model):
    summary, hourly, ev = run_live_simulation(model, seed=1000)
    assert summary["hours_completed"] == 24
    assert len(ev) == 24 * 10
    assert np.isfinite(summary["total_reward"])


def test_seed_is_reproducible():
    a, _, _ = run_live_simulation("PPO", seed=1234)
    b, _, _ = run_live_simulation("PPO", seed=1234)
    assert a["total_reward"] == pytest.approx(b["total_reward"])

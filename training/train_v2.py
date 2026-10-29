"""Improved training protocol (v2).

Compared with train_ppo.py / train_dqn.py this script:
  * gives every agent (PPO, Priority DQN, SAC) the SAME budget (default 100k steps),
  * trains several independent random seeds,
  * evaluates on fixed *validation* days during training, giving a
    learning curve (reports/learning_curves.csv),
  * keeps the best checkpoint on validation (model selection), and
  * never touches the *test* days (seeds 1000-1029) used for the final
    comparison, so test results are unbiased.

Usage:
    python training/train_v2.py --algo ppo --seed 0
    python training/train_v2.py --algo dqn --seed 0 --steps 100000
    python training/train_v2.py --algo sac --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from Chargym_Charging_Station.envs.dqn_wrapper import ChargingDQNWrapper
from Chargym_Charging_Station.envs.gymnasium_wrapper import ChargingGymnasiumWrapper

VALIDATION_SEEDS = list(range(500, 520))  # disjoint from test seeds 1000-1029
MODEL_DIR = ROOT / "models" / "v2"
NAMES = {"ppo": "PPO", "dqn": "Priority DQN", "sac": "SAC"}
CURVE_DIR = ROOT / "reports" / "learning_curves"


def make_env(algo):
    # PPO and SAC control continuous per-bay rates; DQN uses priority actions.
    return ChargingDQNWrapper() if algo == "dqn" else ChargingGymnasiumWrapper()


def evaluate(model, algo, seeds):
    """Mean deterministic episode reward over fixed scenario seeds."""
    # The simulator draws scenarios from NumPy's global RNG, which also
    # drives training. Save and restore it so evaluation does not make
    # the training scenarios repeat.
    rng_state = np.random.get_state()
    env = make_env(algo)
    rewards = []
    try:
        for seed in seeds:
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)
            total = 0.0
            for _ in range(24):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            rewards.append(total)
    finally:
        env.close()
        np.random.set_state(rng_state)
    return float(np.mean(rewards)), float(np.std(rewards))


class ValidationCallback(BaseCallback):
    def __init__(self, algo, seed, eval_freq, save_path):
        super().__init__()
        self.algo, self.seed = algo, seed
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best = -np.inf
        self.rows = []

    def _record(self):
        mean, std = evaluate(self.model, self.algo, VALIDATION_SEEDS)
        self.rows.append({
            "algo": NAMES[self.algo],
            "seed": self.seed,
            "timesteps": self.num_timesteps,
            "val_mean_reward": mean,
            "val_std_reward": std,
        })
        if mean > self.best:
            self.best = mean
            self.model.save(str(self.save_path))
        print(f"[{self.algo} s{self.seed}] {self.num_timesteps:>7} steps  "
              f"val reward {mean:8.3f}  best {self.best:8.3f}", flush=True)

    def _on_training_start(self):
        self._record()  # untrained policy = starting point of the curve

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            self._record()
        return True


def build(algo, env, seed):
    if algo == "ppo":
        return PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.0,
            seed=seed, device="cpu", verbose=0,
        )
    if algo == "sac":
        # Off-policy, maximum-entropy actor-critic for continuous actions.
        # n_steps=8: the departure penalty arrives hours after the charging
        # decision that caused it. With standard 1-step targets SAC learned to
        # shuffle energy between cars (V2G) to avoid grid cost and never
        # connected that to the later penalty (reports/ablation/). 8-step
        # returns propagate the penalty back to the decision much faster.
        return SAC(
            "MlpPolicy", env,
            learning_rate=3e-4, buffer_size=100_000, learning_starts=2_000,
            batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
            gradient_steps=1, ent_coef="auto", n_steps=8,
            seed=seed, device="cpu", verbose=0,
        )
    return DQN(
        "MlpPolicy", env,
        learning_rate=5e-4, buffer_size=50_000, learning_starts=2_000,
        batch_size=64, gamma=0.99, train_freq=4, target_update_interval=1_000,
        exploration_fraction=0.3, exploration_final_eps=0.05,
        seed=seed, device="cpu", verbose=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn", "sac"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    args = parser.parse_args()

    torch.set_num_threads(1)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)

    name = f"{args.algo}_seed{args.seed}"
    model = build(args.algo, make_env(args.algo), args.seed)
    callback = ValidationCallback(args.algo, args.seed, args.eval_freq,
                                  MODEL_DIR / f"{name}_best")
    model.learn(total_timesteps=args.steps, callback=callback)

    pd.DataFrame(callback.rows).to_csv(CURVE_DIR / f"{name}.csv", index=False)
    print(f"Done. Best validation reward {callback.best:.3f} -> "
          f"{MODEL_DIR / (name + '_best.zip')}")


if __name__ == "__main__":
    main()

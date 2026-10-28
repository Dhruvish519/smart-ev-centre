
import sys
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Chargym_Charging_Station.envs.dqn_wrapper import (
    ChargingDQNWrapper,
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dqn_priority"
    / "charging_dqn_priority_final.zip"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "dqn_priority"
    / "evaluation.csv"
)

EPISODES = 20


def evaluate(env, model=None):
    rewards = []

    for episode in range(EPISODES):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(
                    obs, deterministic=True
                )

            obs, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)

        rewards.append(total_reward)
        print(
            f"Episode {episode + 1:02d}: "
            f"reward = {total_reward:.2f}"
        )

    return np.asarray(rewards, dtype=float)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = DQN.load(str(MODEL_PATH), device="cpu")
    env = ChargingDQNWrapper()

    try:
        print("\n=== Priority DQN evaluation ===")
        dqn_rewards = evaluate(env, model)

        print("\n=== Random baseline evaluation ===")
        random_rewards = evaluate(env)

        print("\n=== Summary ===")
        print(
            f"Priority DQN mean: {dqn_rewards.mean():.2f} "
            f"(std: {dqn_rewards.std():.2f})"
        )
        print(
            f"Random baseline mean: {random_rewards.mean():.2f} "
            f"(std: {random_rewards.std():.2f})"
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True, exist_ok=True
        )

        with OUTPUT_PATH.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                ["episode", "priority_dqn_reward", "random_reward"]
            )

            for i in range(EPISODES):
                writer.writerow(
                    [
                        i + 1,
                        dqn_rewards[i],
                        random_rewards[i],
                    ]
                )

        print(f"\nResults saved to: {OUTPUT_PATH}")

    finally:
        env.close()


if __name__ == "__main__":
    main()

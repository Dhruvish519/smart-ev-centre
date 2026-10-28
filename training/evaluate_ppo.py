
import sys
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Chargym_Charging_Station.envs.gymnasium_wrapper import (
    ChargingGymnasiumWrapper,
)

MODEL_PATH = PROJECT_ROOT / "models" / "ppo" / "charging_ppo_final.zip"
OUTPUT_PATH = PROJECT_ROOT / "logs" / "ppo" / "evaluation.csv"
EPISODES = 20


def evaluate_policy(env, model=None):
    episode_returns = []

    for episode in range(EPISODES):
        obs, _ = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

        episode_returns.append(total_reward)

        print(
            f"Episode {episode + 1:02d}/{EPISODES}: "
            f"reward = {total_reward:.2f}"
        )

    return np.asarray(episode_returns)


def main():
    env = ChargingGymnasiumWrapper()
    model = PPO.load(str(MODEL_PATH), device="cpu")

    print("\nEvaluating PPO...")
    ppo_returns = evaluate_policy(env, model)

    print("\nEvaluating random-action baseline...")
    random_returns = evaluate_policy(env)

    print("\n===== Evaluation Summary =====")
    print(
        f"PPO mean reward: {ppo_returns.mean():.2f} "
        f"+/- {ppo_returns.std():.2f}"
    )
    print(
        f"Random mean reward: {random_returns.mean():.2f} "
        f"+/- {random_returns.std():.2f}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "ppo_reward", "random_reward"])

        for i, (ppo_reward, random_reward) in enumerate(
            zip(ppo_returns, random_returns), start=1
        ):
            writer.writerow([i, ppo_reward, random_reward])

    print(f"\nResults saved to: {OUTPUT_PATH}")
    env.close()


if __name__ == "__main__":
    main()

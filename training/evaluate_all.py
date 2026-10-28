
import sys
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, DQN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Chargym_Charging_Station.envs.gymnasium_wrapper import (
    ChargingGymnasiumWrapper,
)
from Chargym_Charging_Station.envs.dqn_wrapper import (
    ChargingDQNWrapper,
)

EPISODES = 20
OUTPUT_PATH = PROJECT_ROOT / "logs" / "comparison" / "evaluation.csv"


def evaluate(env, model=None, episodes=EPISODES):
    rewards = []

    for episode in range(episodes):
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

        rewards.append(total_reward)
        print(f"Episode {episode + 1:02d}: reward = {total_reward:.2f}")

    return np.asarray(rewards)


def report(name, rewards):
    print(
        f"{name}: mean = {rewards.mean():.2f}, "
        f"std = {rewards.std():.2f}"
    )


def main():
    ppo_env = ChargingGymnasiumWrapper()
    dqn_env = ChargingDQNWrapper()

    ppo_model = PPO.load(
        str(PROJECT_ROOT / "models" / "ppo" / "charging_ppo_final.zip"),
        device="cpu",
    )
    dqn_model = DQN.load(
        str(PROJECT_ROOT / "models" / "dqn" / "charging_dqn_final.zip"),
        device="cpu",
    )

    print("\n=== PPO evaluation ===")
    ppo_rewards = evaluate(ppo_env, ppo_model)
    report("PPO", ppo_rewards)

    print("\n=== PPO random baseline ===")
    ppo_random = evaluate(ppo_env)
    report("PPO random baseline", ppo_random)

    print("\n=== DQN evaluation ===")
    dqn_rewards = evaluate(dqn_env, dqn_model)
    report("DQN", dqn_rewards)

    print("\n=== DQN random baseline ===")
    dqn_random = evaluate(dqn_env)
    report("DQN random baseline", dqn_random)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "episode",
            "ppo_reward",
            "ppo_random_reward",
            "dqn_reward",
            "dqn_random_reward",
        ])

        for i in range(EPISODES):
            writer.writerow([
                i + 1,
                ppo_rewards[i],
                ppo_random[i],
                dqn_rewards[i],
                dqn_random[i],
            ])

    print(f"\nResults saved to: {OUTPUT_PATH}")

    ppo_env.close()
    dqn_env.close()


if __name__ == "__main__":
    main()

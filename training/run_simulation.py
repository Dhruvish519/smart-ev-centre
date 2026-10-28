
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Chargym_Charging_Station.envs.gymnasium_wrapper import (
    ChargingGymnasiumWrapper,
)

MODEL_PATH = PROJECT_ROOT / "models" / "ppo" / "charging_ppo_final.zip"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    env = ChargingGymnasiumWrapper()
    model = PPO.load(str(MODEL_PATH), device="cpu")

    obs, info = env.reset()
    total_reward = 0.0
    rows = []

    for hour in range(24):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        rows.append({
            "hour": hour,
            "reward": float(reward),
            "action_mean": float(np.mean(action)),
            "action_min": float(np.min(action)),
            "action_max": float(np.max(action)),
        })

        print(
            f"Hour {hour + 1:02d}/24 | "
            f"Reward: {reward:8.3f} | "
            f"Mean action: {np.mean(action):6.3f}"
        )

        if terminated or truncated:
            break

    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(exist_ok=True)

    import pandas as pd

    results_path = output_dir / "simulation_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)

    print(f"\nEpisode reward: {total_reward:.3f}")
    print(f"Results saved to: {results_path}")

    env.close()


if __name__ == "__main__":
    main()

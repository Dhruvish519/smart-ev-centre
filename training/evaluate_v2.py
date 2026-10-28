"""Test the v2 models (every training seed) on the 30 held-out test days.

    python training/evaluate_v2.py

Writes reports/v2_comparison.csv and copies the best seed of each agent
(chosen on VALIDATION reward, never on test) to models/v2/<algo>_final.zip,
which the dashboard uses.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN, PPO

from training.compare_models import NUM_SCENARIOS, START_SEED
from training.train_v2 import make_env

MODEL_DIR = ROOT / "models" / "v2"
CURVE_DIR = ROOT / "reports" / "learning_curves"
OUT = ROOT / "reports" / "v2_comparison.csv"
NAMES = {"ppo": "PPO", "dqn": "Priority DQN"}


def run_day(model, algo, seed):
    env = make_env(algo)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(24):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break
    legacy = env.legacy_env
    return total, float(np.nansum(legacy.Grid_Evol)), float(np.nansum(legacy.Penalty_Evol))


def main():
    rows = []
    for algo, cls in (("ppo", PPO), ("dqn", DQN)):
        best_val, best_path = -np.inf, None
        for path in sorted(MODEL_DIR.glob(f"{algo}_seed*_best.zip")):
            train_seed = int(path.stem.split("seed")[1].split("_")[0])
            curve = pd.read_csv(CURVE_DIR / f"{algo}_seed{train_seed}.csv")
            val = curve["val_mean_reward"].max()
            if val > best_val:
                best_val, best_path = val, path
            model = cls.load(str(path), device="cpu")
            for i in range(NUM_SCENARIOS):
                seed = START_SEED + i
                total, grid, pen = run_day(model, algo, seed)
                rows.append({"scenario_seed": seed, "model": NAMES[algo],
                             "train_seed": train_seed, "total_reward": total,
                             "grid_energy": grid, "departure_penalty": pen})
        if best_path is not None:
            shutil.copy(best_path, MODEL_DIR / f"{algo}_final.zip")
            print(f"{NAMES[algo]}: deployed {best_path.name} (validation {best_val:.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    per_seed = df.groupby(["model", "train_seed"])["total_reward"].mean()
    print(per_seed.round(3).to_string())
    print(df.groupby("model")[["total_reward", "grid_energy", "departure_penalty"]]
          .agg(["mean", "std"]).round(2).to_string())


if __name__ == "__main__":
    main()

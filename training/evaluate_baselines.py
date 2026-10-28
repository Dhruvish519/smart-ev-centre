"""Evaluate simple rule-based baselines on the same 30 scenarios as the RL models.

Baselines give context to the RL results: a learned policy is only
interesting if it beats sensible hand-written rules.

    python training/evaluate_baselines.py

Writes reports/baseline_comparison.csv (same columns as model_comparison.csv).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Chargym_Charging_Station.envs.gymnasium_wrapper import ChargingGymnasiumWrapper

NUM_SCENARIOS = 30
START_SEED = 1000  # identical to training/compare_models.py
OUTPUT_PATH = ROOT / "reports" / "baseline_comparison.csv"


def charge_all_max(obs):
    """Charge every connected EV at full power, every hour."""
    return np.ones(10, dtype=np.float32)


def do_nothing(obs):
    """Never charge (lower bound: zero grid cost, maximum penalty)."""
    return np.zeros(10, dtype=np.float32)


def chargym_rbc(obs):
    """Rule-based controller from the original Chargym paper.

    Charge at full power if the EV leaves within ~3 hours, otherwise
    charge in proportion to current/next-hour solar radiation.
    """
    action = np.zeros(10, dtype=np.float32)
    for car in range(10):
        t_leave = obs[18 + car]
        if t_leave == 0:
            action[car] = 0.0
        elif t_leave < 0.16667:
            action[car] = 1.0
        else:
            action[car] = (obs[0] + obs[2]) / 2
    return action


def cheap_hours(obs):
    """Charge fully when price is low (normalised price 0.5) or departure is near."""
    price_is_low = obs[1] <= 0.6
    action = np.zeros(10, dtype=np.float32)
    for car in range(10):
        t_leave = obs[18 + car]
        if t_leave > 0 and (price_is_low or t_leave < 0.16667):
            action[car] = 1.0
    return action


BASELINES = {
    "Charge-all (max power)": charge_all_max,
    "Rule-based (Chargym RBC)": chargym_rbc,
    "Cheap-hours rule": cheap_hours,
    "No charging": do_nothing,
}


def run(policy, seed):
    env = ChargingGymnasiumWrapper()
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    steps = 0
    for _ in range(24):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += reward
        steps += 1
        if terminated or truncated:
            break
    legacy = env.legacy_env
    return total, float(np.nansum(legacy.Grid_Evol)), float(np.nansum(legacy.Penalty_Evol)), steps


def main():
    rows = []
    for i in range(NUM_SCENARIOS):
        seed = START_SEED + i
        for name, policy in BASELINES.items():
            total, grid, penalty, steps = run(policy, seed)
            rows.append({
                "scenario_seed": seed, "model": name, "total_reward": total,
                "grid_energy": grid, "departure_penalty": penalty,
                "steps_completed": steps,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(df.groupby("model")[["total_reward", "grid_energy", "departure_penalty"]]
          .agg(["mean", "std"]).round(2).to_string())
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

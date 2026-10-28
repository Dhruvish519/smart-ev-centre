
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "reports" / "simulation_results.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Results not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    hours = df["hour"] + 1

    # Chart 1: hourly reward
    plt.figure(figsize=(11, 5))
    plt.plot(hours, df["reward"], marker="o")
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Hour of day")
    plt.ylabel("Reward")
    plt.title("PPO Charging Scheduler: Hourly Reward")
    plt.xticks(range(1, 25))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ppo_hourly_reward.png", dpi=200)
    plt.close()

    # Chart 2: average action output
    plt.figure(figsize=(11, 5))
    plt.plot(hours, df["action_mean"], marker="o")
    plt.xlabel("Hour of day")
    plt.ylabel("Mean action output")
    plt.title("PPO Charging Scheduler: Mean Action Output")
    plt.xticks(range(1, 25))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ppo_hourly_actions.png", dpi=200)
    plt.close()

    print("Saved charts:")
    print(FIGURE_DIR / "ppo_hourly_reward.png")
    print(FIGURE_DIR / "ppo_hourly_actions.png")


if __name__ == "__main__":
    main()

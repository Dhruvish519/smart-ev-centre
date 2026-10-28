
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_CSV = (
    PROJECT_ROOT / "logs" / "comparison" / "evaluation.csv"
)
PRIORITY_CSV = (
    PROJECT_ROOT / "logs" / "dqn_priority" / "evaluation.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def load_rewards(path, columns):
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {path}")

    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    return {
        name: np.array([float(row[column]) for row in rows])
        for name, column in columns.items()
    }


def main():
    original = load_rewards(
        ORIGINAL_CSV,
        {
            "PPO": "ppo_reward",
            "PPO random": "ppo_random_reward",
            "Original DQN": "dqn_reward",
            "Original DQN random": "dqn_random_reward",
        },
    )

    priority = load_rewards(
        PRIORITY_CSV,
        {
            "Priority DQN": "priority_dqn_reward",
            "Priority DQN random": "random_reward",
        },
    )

    results = {**original, **priority}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: mean reward with standard-deviation error bars.
    names = list(results.keys())
    means = [results[name].mean() for name in names]
    stds = [results[name].std() for name in names]

    plt.figure(figsize=(10, 6))
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.axhline(0, linewidth=0.8)
    plt.ylabel("Mean episode reward")
    plt.title("Charging Scheduler: Evaluation Rewards")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mean_reward_comparison.png", dpi=200)
    plt.close()

    # Figure 2: reward distribution across evaluation episodes.
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [results[name] for name in names],
        tick_labels=names,
    )
    plt.ylabel("Episode reward")
    plt.title("Episode Reward Distribution")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "episode_reward_distribution.png",
        dpi=200,
    )
    plt.close()

    print("Graphs saved to:")
    print(OUTPUT_DIR / "mean_reward_comparison.png")
    print(OUTPUT_DIR / "episode_reward_distribution.png")

    print("\nMean reward summary:")
    for name, rewards in results.items():
        print(
            f"{name:24s} mean={rewards.mean():7.2f}, "
            f"std={rewards.std():.2f}"
        )


if __name__ == "__main__":
    main()



from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "reports" / "model_comparison.csv"
OUTPUT_DIR = ROOT / "reports" / "comparison_graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Comparison CSV not found: {INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "scenario_seed",
    "model",
    "total_reward",
    "grid_energy",
    "departure_penalty",
]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

summary = (
    df.groupby("model")
    .agg(
        mean_reward=("total_reward", "mean"),
        mean_grid_energy=("grid_energy", "mean"),
        mean_departure_penalty=("departure_penalty", "mean"),
    )
    .reindex(["PPO", "Priority DQN"])
)

if summary.isna().any().any():
    raise ValueError(
        "Expected both PPO and Priority DQN results in the CSV."
    )

# --------------------------------------------------
# GRAPH 1: MEAN REWARD
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    summary.index,
    summary["mean_reward"],
)

ax.set_title("PPO vs Priority DQN: Mean Reward")
ax.set_ylabel("Mean episode reward")
ax.set_xlabel("Model")
ax.axhline(0, linewidth=0.8)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "mean_reward.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# --------------------------------------------------
# GRAPH 2: GRID ENERGY
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    summary.index,
    summary["mean_grid_energy"],
)

ax.set_title("PPO vs Priority DQN: Mean Grid Energy")
ax.set_ylabel("Mean grid energy (environment units)")
ax.set_xlabel("Model")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "grid_energy.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# --------------------------------------------------
# GRAPH 3: DEPARTURE PENALTY
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    summary.index,
    summary["mean_departure_penalty"],
)

ax.set_title("PPO vs Priority DQN: Mean Departure Penalty")
ax.set_ylabel("Mean departure penalty")
ax.set_xlabel("Model")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "departure_penalty.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# --------------------------------------------------
# GRAPH 4: REWARD ACROSS ALL SCENARIOS
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

for model_name in ["PPO", "Priority DQN"]:
    model_df = (
        df[df["model"] == model_name]
        .sort_values("scenario_seed")
    )

    ax.plot(
        model_df["scenario_seed"],
        model_df["total_reward"],
        marker="o",
        markersize=3,
        label=model_name,
    )

ax.set_title("Episode Reward Across 30 Scenarios")
ax.set_xlabel("Scenario seed")
ax.set_ylabel("Episode reward")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "reward_across_scenarios.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# --------------------------------------------------
# SAVE SUMMARY TABLE
# --------------------------------------------------

summary.to_csv(OUTPUT_DIR / "comparison_summary.csv")

print("\nAll comparison graphs generated successfully!")
print(f"Output folder: {OUTPUT_DIR}")
print("\nGenerated files:")
for file_path in sorted(OUTPUT_DIR.iterdir()):
    print(f"- {file_path.name}")

print("\nSummary:")
print(summary.round(3).to_string())


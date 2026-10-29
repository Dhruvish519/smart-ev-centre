"""Comparison graphs of PPO, Priority DQN and SAC (PNG, for slides/report).

    python training/plot_algorithm_comparison.py

Reads reports/v2_comparison.csv, reports/baseline_comparison.csv and
reports/learning_curves/*.csv; writes reports/algorithm_comparison/*.png
and summary.csv.
"""
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = REPORTS / "algorithm_comparison"
ALGOS = ["Priority DQN", "PPO", "SAC"]
COLORS = {"PPO": "#4C5BD4", "Priority DQN": "#12A08A", "SAC": "#C2410C"}
RULE = "#A7B1C2"
INK, MUTED, LINE = "#16233A", "#5A6880", "#D9DFE8"

plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": LINE, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "semibold",
    "axes.titlesize": 14, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("saved", OUT / name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v2 = pd.read_csv(REPORTS / "v2_comparison.csv")
    base = pd.read_csv(REPORTS / "baseline_comparison.csv")
    algos = [a for a in ALGOS if a in set(v2["model"])]
    cols = ["total_reward", "grid_energy", "departure_penalty"]

    per_day = v2.groupby(["model", "scenario_seed"], as_index=False)[cols].mean()
    per_seed = v2.groupby(["model", "train_seed"])[cols].mean()
    allp = pd.concat([per_day, base], ignore_index=True)
    summary = (allp.groupby("model")
               .agg(mean_reward=("total_reward", "mean"), std_reward=("total_reward", "std"),
                    grid_energy=("grid_energy", "mean"), departure_penalty=("departure_penalty", "mean"))
               .sort_values("mean_reward", ascending=False))
    summary["seed_std_reward"] = per_seed["total_reward"].groupby("model").std()
    summary.round(3).to_csv(OUT / "summary.csv")

    # 1. Leaderboard -------------------------------------------------
    s = summary.sort_values("mean_reward")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(s.index, s["mean_reward"], xerr=s["std_reward"], capsize=3,
            color=[COLORS.get(m, RULE) for m in s.index], error_kw=dict(ecolor=MUTED, lw=1))
    for y, v in enumerate(s["mean_reward"]):
        ax.text(-0.6, y, f"{v:.2f}", va="center", ha="right", color="white", fontweight="bold")
    ax.set_xlabel("Mean reward per test day (closer to 0 is better)")
    ax.set_title("Leaderboard: RL algorithms vs rule-based baselines", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "1_leaderboard.png")

    # 2. Metric bars (error = std across training seeds) ------------
    agg = per_seed.groupby("model").agg(["mean", "std"])
    titles = [("total_reward", "Reward per day", "higher is better"),
              ("grid_energy", "Grid energy (kWh/day)", "lower is cheaper"),
              ("departure_penalty", "Departure penalty", "lower is better")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (c, t, note) in zip(axes, titles):
        m = [agg.loc[a, (c, "mean")] for a in algos]
        e = [agg.loc[a, (c, "std")] for a in algos]
        bars = ax.bar(algos, m, yerr=e, capsize=4, color=[COLORS[a] for a in algos],
                      error_kw=dict(ecolor=MUTED, lw=1))
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in m], padding=3, color=INK)
        ax.set_title(t, loc="left")
        ax.set_ylabel(note, color=MUTED)
        if c != "total_reward":
            ax.set_ylim(0, max(np.add(m, e)) * 1.15)
        ax.grid(axis="x", visible=False)
        if c == "total_reward":
            ax.set_ylim(min(m) * 1.25, 0)  # rewards are negative: shorter bar is better
    fig.suptitle("Algorithm comparison on 30 test days (error bars: std over 3 training seeds)",
                 x=0.01, ha="left", fontsize=13, color=MUTED)
    fig.tight_layout()
    save(fig, "2_metrics_by_algorithm.png")

    # 3. Reward distribution -----------------------------------------
    best_rule = summary.drop(index=algos)["mean_reward"].idxmax()
    names = algos + [best_rule]
    data = [allp.loc[allp["model"] == m, "total_reward"].values for m in names]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                    medianprops=dict(color=INK, lw=1.5))
    rng = np.random.default_rng(0)
    for i, (m, d) in enumerate(zip(names, data), start=1):
        c = COLORS.get(m, RULE)
        bp["boxes"][i - 1].set(facecolor=c + "30", edgecolor=c)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=14, color=c, alpha=.8, zorder=3)
    ax.set_xticks(range(1, len(names) + 1), [*algos, f"{best_rule}\n(best rule)"])
    ax.set_ylabel("Episode reward")
    ax.set_title("Reward on each of the 30 test days", loc="left")
    ax.grid(axis="x", visible=False)
    save(fig, "3_reward_distribution.png")

    # 4. Learning curves ---------------------------------------------
    curves = pd.concat([pd.read_csv(p) for p in sorted((REPORTS / "learning_curves").glob("*.csv"))])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for a in algos:
        g = curves[curves["algo"] == a].groupby("timesteps")["val_mean_reward"].agg(["mean", "min", "max"])
        ax.fill_between(g.index, g["min"], g["max"], color=COLORS[a], alpha=.15, lw=0)
        ax.plot(g.index, g["mean"], color=COLORS[a], lw=2.4, label=a)
    bv = REPORTS / "baseline_validation.csv"
    if bv.exists():
        rule = pd.read_csv(bv).set_index("model")["val_mean_reward"].drop("No charging", errors="ignore")
        ax.axhline(rule.max(), color=MUTED, ls="--", lw=1.3)
        ax.text(ax.get_xlim()[1], rule.max(), f"best rule ({rule.idxmax()})  ", ha="right",
                va="bottom", color=MUTED, fontsize=11)
    ax.set_xlabel("Training steps (1 day = 24 steps)")
    ax.set_ylabel("Validation reward")
    ax.set_title("Learning curves (mean of 3 seeds, band = range)", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    save(fig, "4_learning_curves.png")

    # 5. Trade-off ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, r in summary.iterrows():
        learned = m in algos
        ax.scatter(r["grid_energy"], r["departure_penalty"], s=170 if learned else 80,
                   color=COLORS.get(m, RULE), edgecolor="white", lw=1.5, zorder=3)
        if learned or m == "No charging":
            off, ha = ((-10, -4), "right") if m == "SAC" else ((0, 10), "center")
            if m == "No charging":
                off, ha = (10, -4), "left"
            ax.annotate(m, (r["grid_energy"], r["departure_penalty"]), xytext=off,
                        textcoords="offset points", color=INK, ha=ha)
    rules = summary.drop(index=algos + ["No charging"])
    ax.annotate("Rule-based policies", (rules["grid_energy"].max(), rules["departure_penalty"].max()),
                xytext=(0, -18), textcoords="offset points", ha="right", color=MUTED)
    ax.set_xlim(left=-10); ax.set_ylim(bottom=-5)
    ax.set_xlabel("Grid energy bought per day (kWh)")
    ax.set_ylabel("Departure penalty per day")
    ax.set_title("Cost vs comfort trade-off (bottom-left is ideal)", loc="left")
    save(fig, "5_cost_comfort_tradeoff.png")

    # 6. Head-to-head matrix ----------------------------------------
    piv = per_day.pivot(index="scenario_seed", columns="model", values="total_reward")[algos]
    n = len(algos)
    win = np.full((n, n), np.nan)
    fig, ax = plt.subplots(figsize=(1.9 * n + 2, 1.3 * n + 1))
    for i, r in enumerate(algos):
        for j, c in enumerate(algos):
            if i == j:
                continue
            w = int((piv[r] > piv[c]).sum())
            p = stats.ttest_rel(piv[r], piv[c]).pvalue
            win[i, j] = w / len(piv)
            ax.text(j, i, f"{w}/{len(piv)}\np={p:.1e}" if p < 1e-3 else f"{w}/{len(piv)}\np={p:.3f}",
                    ha="center", va="center", color=INK, fontsize=12)
    ax.imshow(win, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "w", ["#FDE7EA", "#FFFFFF", "#BDE8DE"]), vmin=0, vmax=1)
    ax.set_xticks(range(n), [f"vs {a}" for a in algos]); ax.set_yticks(range(n), algos)
    ax.xaxis.tick_top(); ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title("Days won (row vs column) and paired t-test p-value", loc="left", pad=30)
    save(fig, "6_head_to_head.png")

    # 7. SAC ablation: 1-step vs 8-step returns ----------------------
    abl = REPORTS / "ablation" / "sac_1step_curves.csv"
    if abl.exists() and "SAC" in set(curves["algo"]):
        fig, ax = plt.subplots(figsize=(10, 4.6))
        one = pd.read_csv(abl)
        nstep = curves[curves["algo"] == "SAC"]
        for df, label, colour, style in ((one, "SAC, standard 1-step targets", MUTED, "--"),
                                          (nstep, "SAC, 8-step returns (used)", COLORS["SAC"], "-")):
            g = df.groupby("timesteps")["val_mean_reward"].agg(["mean", "min", "max"])
            ax.fill_between(g.index, g["min"], g["max"], color=colour, alpha=.15, lw=0)
            ax.plot(g.index, g["mean"], color=colour, lw=2.4, ls=style, label=label)
        if bv.exists():
            ax.axhline(rule.max(), color=MUTED, ls=":", lw=1.2)
            ax.text(ax.get_xlim()[1], rule.max(), f"best rule  ", ha="right", va="bottom",
                    color=MUTED, fontsize=11)
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Validation reward")
        ax.set_title("Why SAC needs n-step returns on this task", loc="left")
        ax.legend(frameon=False, loc="lower right")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        save(fig, "7_sac_ablation.png")

    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()

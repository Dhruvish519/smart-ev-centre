import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui import theme as T

REPORTS = ROOT / "reports"
V1 = REPORTS / "model_comparison.csv"
V2 = REPORTS / "v2_comparison.csv"
BASE = REPORTS / "baseline_comparison.csv"
CURVES = REPORTS / "learning_curves"
GRAPHS = REPORTS / "algorithm_comparison"
ALGO_ORDER = ["Priority DQN", "PPO", "SAC"]
CHART = {"displayModeBar": False}


@st.cache_data
def load():
    base = pd.read_csv(BASE) if BASE.exists() else pd.DataFrame()
    v1 = pd.read_csv(V1) if V1.exists() else pd.DataFrame()
    v2 = pd.read_csv(V2) if V2.exists() else pd.DataFrame()
    curves = (pd.concat([pd.read_csv(p) for p in sorted(CURVES.glob("*.csv"))])
              if CURVES.exists() and any(CURVES.glob("*.csv")) else pd.DataFrame())
    return base, v1, v2, curves


def rgba(hex_colour, alpha):
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


base, v1, v2, curves = load()

T.header(
    "How well do the agents do?",
    "Three reinforcement-learning algorithms and four hand-written rules, all "
    "tested on the same 30 days (seeds 1000–1029). None of these days were used "
    "for training or for choosing checkpoints.",
)

if v2.empty and v1.empty:
    st.info("No evaluation results yet. Run training/evaluate_v2.py first.")
    st.stop()

use_v2 = not v2.empty
if use_v2:
    # Mean over training seeds for each test day.
    rl = (v2.groupby(["model", "scenario_seed"], as_index=False)
          [["total_reward", "grid_energy", "departure_penalty"]].mean())
    st.caption("Learned agents: 100k training steps each, 3 independent training "
               "seeds, best checkpoint chosen on separate validation days.")
else:
    rl = v1
algos = [a for a in ALGO_ORDER if a in set(rl["model"])]
all_df = pd.concat([rl, base], ignore_index=True)
summary = (all_df.groupby("model")
           .agg(reward=("total_reward", "mean"), sd=("total_reward", "std"),
                grid=("grid_energy", "mean"), penalty=("departure_penalty", "mean"))
           .sort_values("reward"))

# ------------------------------------------------------------------
T.section("Leaderboard", "Mean reward per day (closer to zero is better). "
          "Bars show ±1 standard deviation across the 30 test days.")
fig = go.Figure(go.Bar(
    x=summary["reward"], y=summary.index, orientation="h",
    error_x=dict(type="data", array=summary["sd"], color=T.MUTED, thickness=1.2),
    marker_color=[T.MODEL_COLORS.get(m, T.RULE) for m in summary.index],
    text=[f"{v:.2f}" for v in summary["reward"]],
    textposition="inside", insidetextanchor="start",
    textfont=dict(color="white", size=14),
    hovertemplate="%{y}: %{x:.2f}<extra></extra>",
))
fig.update_xaxes(title="Mean episode reward")
st.plotly_chart(T.style(fig, height=360, legend=False), use_container_width=True, config=CHART)
st.markdown('<p class="ev-note">Coloured bars are learned agents, grey bars are hand-written rules.</p>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------
T.section("Algorithm comparison by metric",
          "Each bar is the mean over the 30 test days. Error bars show how much the "
          "result varies between the 3 training seeds, which tells you how reliable "
          "each algorithm is to train.")
if use_v2:
    per_seed = (v2.groupby(["model", "train_seed"])
                [["total_reward", "grid_energy", "departure_penalty"]].mean())
    agg = per_seed.groupby("model").agg(["mean", "std"])
else:
    agg = rl.groupby("model")[["total_reward", "grid_energy", "departure_penalty"]].agg(["mean", "std"])
    agg.loc[:, (slice(None), "std")] = 0
metrics = [("total_reward", "Reward per day", "higher is better"),
           ("grid_energy", "Grid energy (kWh/day)", "lower is cheaper"),
           ("departure_penalty", "Departure penalty", "lower is better")]
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.09,
                    subplot_titles=[f"{t}<br><sup>{n}</sup>" for _, t, n in metrics])
for i, (col, _, _) in enumerate(metrics, start=1):
    fig.add_trace(go.Bar(
        x=algos, y=[agg.loc[a, (col, "mean")] for a in algos],
        error_y=dict(type="data", array=[agg.loc[a, (col, "std")] for a in algos],
                     color=T.MUTED, thickness=1.2),
        marker_color=[T.MODEL_COLORS[a] for a in algos],
        text=[f"{agg.loc[a, (col, 'mean')]:.1f}" for a in algos],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ), row=1, col=i)
fig.update_annotations(font=dict(family=T.FONT, size=15, color=T.INK))
T.style(fig, height=400, legend=False).update_layout(margin=dict(t=70))
st.plotly_chart(fig, use_container_width=True, config=CHART)

# ------------------------------------------------------------------
left, right = st.columns(2, gap="large")
with left:
    T.section("Reward on every test day",
              "Each dot is one day. The box shows the middle half of days; "
              "the line inside is the median.")
    best_rule = summary.drop(index=algos, errors="ignore")["reward"].idxmax()
    fig = go.Figure()
    for m in algos + [best_rule]:
        vals = all_df.loc[all_df["model"] == m, "total_reward"]
        c = T.MODEL_COLORS.get(m, T.RULE)
        fig.add_trace(go.Box(
            y=vals, name=m if m in algos else f"{m}<br>(best rule)",
            marker_color=c, fillcolor=rgba(c, .18), line=dict(color=c),
            boxpoints="all", jitter=0.4, pointpos=0, marker_size=5,
            hovertemplate="%{y:.2f}<extra></extra>",
        ))
    fig.update_yaxes(title="Episode reward")
    st.plotly_chart(T.style(fig, height=400, legend=False), use_container_width=True, config=CHART)

with right:
    T.section("The cost–comfort trade-off",
              "Grid energy bought versus the penalty for cars leaving under-charged. "
              "Bottom-left is ideal.")
    fig = go.Figure()
    label_pos = {"No charging": "middle right", "Priority DQN": "top center",
                 "PPO": "top center", "SAC": "middle left"}
    for m, r in summary.iterrows():
        learned = m in algos
        fig.add_trace(go.Scatter(
            x=[r["grid"]], y=[r["penalty"]],
            mode="markers+text" if learned or m == "No charging" else "markers",
            text=[m], textposition=label_pos.get(m, "top center"),
            marker=dict(size=16 if learned else 11, color=T.MODEL_COLORS.get(m, T.RULE),
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"{m}<br>Grid %{{x:.0f}} kWh<br>Penalty %{{y:.2f}}<extra></extra>",
        ))
    rules = summary.drop(index=algos + ["No charging"], errors="ignore")
    if not rules.empty:
        fig.add_annotation(x=rules["grid"].max(), y=rules["penalty"].max(),
                           text="Rule-based policies", showarrow=False,
                           xanchor="right", yshift=-18, font=dict(color=T.MUTED))
    fig.update_xaxes(title="Grid energy per day (kWh)", range=[-15, summary["grid"].max() * 1.12])
    fig.update_yaxes(title="Departure penalty per day", range=[-7, summary["penalty"].max() * 1.1])
    st.plotly_chart(T.style(fig, height=400, legend=False), use_container_width=True, config=CHART)

# ------------------------------------------------------------------
if not curves.empty:
    T.section("Learning curves",
              "Reward on 20 separate validation days, measured every 5,000 training steps. "
              "Lines are the mean over training seeds; the band shows the range.")
    fig = go.Figure()
    for algo in [a for a in ALGO_ORDER if a in set(curves["algo"])]:
        g = curves[curves["algo"] == algo]
        a = g.groupby("timesteps")["val_mean_reward"].agg(["mean", "min", "max"]).reset_index()
        col = T.MODEL_COLORS.get(algo, T.INK)
        fig.add_trace(go.Scatter(x=list(a.timesteps) + list(a.timesteps[::-1]),
                                 y=list(a["max"]) + list(a["min"][::-1]),
                                 fill="toself", fillcolor=rgba(col, .15), line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=a.timesteps, y=a["mean"], name=algo, mode="lines",
                                 line=dict(color=col, width=2.5),
                                 hovertemplate=f"{algo}<br>%{{x:,}} steps<br>%{{y:.2f}}<extra></extra>"))
    bv = REPORTS / "baseline_validation.csv"
    if bv.exists():
        rule = pd.read_csv(bv).set_index("model")["val_mean_reward"].drop("No charging", errors="ignore")
        fig.add_hline(y=rule.max(), line=dict(color=T.MUTED, dash="dash", width=1.5),
                      annotation_text=f"Best rule ({rule.idxmax()})",
                      annotation_position="bottom right",
                      annotation_font=dict(color=T.MUTED))
    fig.update_xaxes(title="Training steps (1 day = 24 steps)")
    fig.update_yaxes(title="Validation reward")
    st.plotly_chart(T.style(fig, height=380), use_container_width=True, config=CHART)

# ------------------------------------------------------------------
ABL = REPORTS / "ablation" / "sac_1step_curves.csv"
if ABL.exists() and not curves.empty and "SAC" in set(curves["algo"]):
    T.section("Why SAC needs n-step returns",
              "Standard SAC learned to move energy between parked cars to dodge the grid "
              "bill, and only paid for it hours later when those cars left under-charged. "
              "Learning from 8 hours of consequences per update fixes this.")
    fig = go.Figure()
    for df, name, col, dash in ((pd.read_csv(ABL), "SAC, standard 1-step targets", T.MUTED, "dash"),
                                (curves[curves["algo"] == "SAC"], "SAC, 8-step returns (used)",
                                 T.MODEL_COLORS["SAC"], "solid")):
        a = df.groupby("timesteps")["val_mean_reward"].agg(["mean", "min", "max"]).reset_index()
        fig.add_trace(go.Scatter(x=list(a.timesteps) + list(a.timesteps[::-1]),
                                 y=list(a["max"]) + list(a["min"][::-1]), fill="toself",
                                 fillcolor=rgba(col, .15), line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=a.timesteps, y=a["mean"], name=name, mode="lines",
                                 line=dict(color=col, width=2.5, dash=dash)))
    fig.update_xaxes(title="Training steps")
    fig.update_yaxes(title="Validation reward")
    st.plotly_chart(T.style(fig, height=340), use_container_width=True, config=CHART)

# ------------------------------------------------------------------
if len(algos) >= 2:
    T.section("Head-to-head",
              "Row versus column: on how many of the 30 test days the row algorithm "
              "earned the higher reward, with the paired t-test p-value.")
    piv = rl.pivot(index="scenario_seed", columns="model", values="total_reward")[algos].dropna()
    z, text = [], []
    for r in algos:
        zr, tr = [], []
        for c in algos:
            if r == c:
                zr.append(np.nan); tr.append("")
                continue
            wins = int((piv[r] > piv[c]).sum())
            p = stats.ttest_rel(piv[r], piv[c]).pvalue
            zr.append(wins / len(piv))
            tr.append(f"<b>{wins}/{len(piv)}</b><br>p = {p:.1e}" if p < 1e-3 else f"<b>{wins}/{len(piv)}</b><br>p = {p:.3f}")
        z.append(zr); text.append(tr)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"vs {a}" for a in algos], y=algos, text=text, texttemplate="%{text}",
        textfont=dict(family=T.FONT, size=15), zmin=0, zmax=1,
        colorscale=[[0, "#FDE7EA"], [0.5, "#FFFFFF"], [1, "#BDE8DE"]],
        showscale=False, xgap=3, ygap=3, hoverinfo="skip",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    st.plotly_chart(T.style(fig, height=110 + 70 * len(algos), legend=False),
                    use_container_width=True, config=CHART)
    st.markdown('<p class="ev-note">Green cells: the row algorithm wins on most days. '
                "p &lt; 0.05 means the difference is unlikely to be chance.</p>",
                unsafe_allow_html=True)

# ------------------------------------------------------------------
if use_v2 and not v1.empty:
    T.section("Effect of better training",
              "Original models (10k / 20k steps, one seed) against the retrained ones "
              "(100k steps each, best checkpoint chosen on validation days).")
    m1 = v1.groupby("model")["total_reward"].mean()
    m2 = rl.groupby("model")["total_reward"].mean()
    table = pd.DataFrame({"Original": m1, "Retrained": m2}).dropna()
    table["Change"] = table["Retrained"] - table["Original"]
    st.dataframe(table.round(2), use_container_width=True)

pngs = sorted(GRAPHS.glob("*.png")) if GRAPHS.exists() else []
if pngs:
    with st.expander("Comparison graphs as images (for slides)"):
        cols = st.columns(2)
        for i, png in enumerate(pngs):
            with cols[i % 2]:
                st.image(str(png), use_container_width=True)
                st.download_button(f"Download {png.name}", png.read_bytes(), png.name,
                                   "image/png", key=f"dl_{png.stem}")

with st.expander("Full table"):
    st.dataframe(summary.rename(columns={
        "reward": "Mean reward", "sd": "Std", "grid": "Grid kWh / day",
        "penalty": "Penalty / day"}).round(2), use_container_width=True)

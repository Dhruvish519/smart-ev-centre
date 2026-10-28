import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
RL = ("PPO", "Priority DQN")


@st.cache_data
def load():
    base = pd.read_csv(BASE) if BASE.exists() else pd.DataFrame()
    v1 = pd.read_csv(V1) if V1.exists() else pd.DataFrame()
    v2 = pd.read_csv(V2) if V2.exists() else pd.DataFrame()
    curves = (pd.concat([pd.read_csv(p) for p in sorted(CURVES.glob("*.csv"))])
              if CURVES.exists() and any(CURVES.glob("*.csv")) else pd.DataFrame())
    return base, v1, v2, curves


base, v1, v2, curves = load()

T.header(
    "How well do the agents do?",
    "Every policy below faces the same 30 test days (seeds 1000–1029). "
    "None of these days were used for training or for choosing checkpoints.",
)

if v2.empty and v1.empty:
    st.info("No evaluation results yet. Run training/compare_models.py first.")
    st.stop()

use_v2 = not v2.empty
if use_v2:
    # Average over training seeds for each test day.
    rl = (v2.groupby(["model", "scenario_seed"], as_index=False)
          [["total_reward", "grid_energy", "departure_penalty"]].mean())
    st.caption("Learned agents: 100k training steps each, averaged over 3 independent training seeds.")
else:
    rl = v1
all_df = pd.concat([rl, base], ignore_index=True)

summary = (all_df.groupby("model")
           .agg(reward=("total_reward", "mean"), sd=("total_reward", "std"),
                grid=("grid_energy", "mean"), penalty=("departure_penalty", "mean"))
           .sort_values("reward"))

# ------------------------------------------------------------------
T.section("Leaderboard", "Mean reward per day (closer to zero is better). "
          "Bars show ±1 standard deviation across the 30 days.")
colors = [T.MODEL_COLORS.get(m, T.RULE) for m in summary.index]
fig = go.Figure(go.Bar(
    x=summary["reward"], y=summary.index, orientation="h",
    error_x=dict(type="data", array=summary["sd"], color=T.MUTED, thickness=1.2),
    marker_color=colors, text=[f"{v:.2f}" for v in summary["reward"]],
    textposition="inside", insidetextanchor="start",
    textfont=dict(color="white", size=14),
    hovertemplate="%{y}: %{x:.2f}<extra></extra>",
))
fig.update_xaxes(title="Mean episode reward")
st.plotly_chart(T.style(fig, height=330, legend=False), use_container_width=True,
                config={"displayModeBar": False})
st.markdown('<p class="ev-note">Coloured bars are learned agents, grey bars are hand-written rules.</p>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------
left, right = st.columns(2, gap="large")
with left:
    T.section("The cost–comfort trade-off",
              "Grid energy bought versus penalty for cars leaving under-charged. "
              "Bottom-left is ideal.")
    fig = go.Figure()
    positions = {"PPO": "top center", "Priority DQN": "bottom center",
                 "No charging": "middle right"}
    for m, r in summary.iterrows():
        labelled = m in positions
        fig.add_trace(go.Scatter(
            x=[r["grid"]], y=[r["penalty"]],
            mode="markers+text" if labelled else "markers", name=m,
            text=[m], textposition=positions.get(m, "top center"),
            marker=dict(size=16 if m in RL else 11,
                        color=T.MODEL_COLORS.get(m, T.RULE),
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"{m}<br>Grid %{{x:.0f}} kWh<br>Penalty %{{y:.2f}}<extra></extra>",
        ))
    rules = summary.loc[[m for m in summary.index if m not in positions]]
    if not rules.empty:
        fig.add_annotation(x=rules["grid"].mean(), y=rules["penalty"].max(),
                           text="Rule-based policies", showarrow=False, yshift=18,
                           font=dict(color=T.MUTED))
    fig.update_xaxes(title="Grid energy per day (kWh)", rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.update_yaxes(title="Departure penalty per day")
    st.plotly_chart(T.style(fig, height=380, legend=False), use_container_width=True,
                    config={"displayModeBar": False})

with right:
    T.section("PPO vs Priority DQN, day by day",
              "Each dot is one test day. Dots above the line are days PPO did better.")
    piv = rl[rl["model"].isin(RL)].pivot(index="scenario_seed", columns="model",
                                         values="total_reward").dropna()
    if set(RL) <= set(piv.columns):
        lo, hi = piv.values.min() - 1, piv.values.max() + 1
        fig = go.Figure([
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                       line=dict(color=T.LINE, dash="dash"), hoverinfo="skip"),
            go.Scatter(x=piv["Priority DQN"], y=piv["PPO"], mode="markers",
                       marker=dict(size=10, color=T.GRID, opacity=.8,
                                   line=dict(color="white", width=1)),
                       text=piv.index,
                       hovertemplate="Day %{text}<br>DQN %{x:.2f}<br>PPO %{y:.2f}<extra></extra>"),
        ])
        fig.update_xaxes(title="Priority DQN reward", range=[lo, hi])
        fig.update_yaxes(title="PPO reward", range=[lo, hi])
        st.plotly_chart(T.style(fig, height=380, legend=False), use_container_width=True,
                        config={"displayModeBar": False})
        a, b = piv["PPO"].values, piv["Priority DQN"].values
        p = stats.ttest_rel(a, b).pvalue
        wins = int((a > b).sum())
        better = "PPO" if a.mean() > b.mean() else "Priority DQN"
        st.markdown(
            f'<p class="ev-note">PPO is better on <b>{wins} of {len(a)}</b> days. '
            f"Mean difference {a.mean() - b.mean():+.2f} in favour of {better}; "
            f"paired t-test p = {p:.3g}"
            f"{' (significant at 5%)' if p < 0.05 else ' (not significant at 5%)'}.</p>",
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------------
if not curves.empty:
    T.section("Learning curves",
              "Reward on 20 separate validation days, measured every 5,000 training steps. "
              "Lines are the mean over training seeds; the band shows the range.")
    fig = go.Figure()
    for algo, g in curves.groupby("algo"):
        agg = g.groupby("timesteps")["val_mean_reward"].agg(["mean", "min", "max"]).reset_index()
        col = T.MODEL_COLORS.get(algo, T.INK)
        rgba = f"rgba({int(col[1:3], 16)},{int(col[3:5], 16)},{int(col[5:7], 16)},0.15)"
        fig.add_trace(go.Scatter(x=list(agg.timesteps) + list(agg.timesteps[::-1]),
                                 y=list(agg["max"]) + list(agg["min"][::-1]),
                                 fill="toself", fillcolor=rgba, line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=agg.timesteps, y=agg["mean"], name=algo,
                                 mode="lines", line=dict(color=col, width=2.5),
                                 hovertemplate=f"{algo}<br>%{{x:,}} steps<br>%{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title="Training steps (1 day = 24 steps)")
    fig.update_yaxes(title="Validation reward")
    st.plotly_chart(T.style(fig, height=360), use_container_width=True,
                    config={"displayModeBar": False})

# ------------------------------------------------------------------
if use_v2 and not v1.empty:
    T.section("Effect of better training",
              "Original models (10k / 20k steps, one seed) against the retrained ones "
              "(100k steps each, best checkpoint chosen on validation days).")
    m1 = v1.groupby("model")["total_reward"].mean()
    m2 = rl.groupby("model")["total_reward"].mean()
    table = pd.DataFrame({"Original": m1, "Retrained": m2})
    table["Change"] = table["Retrained"] - table["Original"]
    st.dataframe(table.round(2), use_container_width=True)

with st.expander("Full table"):
    st.dataframe(summary.rename(columns={
        "reward": "Mean reward", "sd": "Std", "grid": "Grid kWh / day",
        "penalty": "Penalty / day"}).round(2), use_container_width=True)

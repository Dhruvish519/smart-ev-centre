
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from training.live_simulation import run_live_simulation

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

st.set_page_config(
    page_title="Smart EV Charging Scheduler",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Smart EV Charging Scheduler")
st.caption(
    "Interactive EV charging simulation using trained reinforcement "
    "learning models"
)

# --------------------------------------------------
# Run the actual trained model
# --------------------------------------------------
st.header("Run simulation")

sel_col, seed_col = st.columns([2, 1])
with sel_col:
    model_name = st.selectbox(
        "Choose a trained model",
        ["PPO", "Priority DQN"],
    )
with seed_col:
    seed = st.number_input(
        "Scenario seed",
        min_value=0,
        value=1000,
        step=1,
        help="Same seed = same EV arrivals/departures, so models can be "
        "compared fairly on an identical day.",
    )

if st.button("▶ Run 24-Hour Simulation", type="primary", use_container_width=True):
    try:
        with st.spinner(f"Running the trained {model_name} model..."):
            summary, hourly_df, ev_df = run_live_simulation(model_name, int(seed))

        st.session_state["live_summary"] = summary
        st.session_state["live_hourly"] = hourly_df
        st.session_state["live_ev"] = ev_df
        st.session_state["live_model"] = model_name
        st.session_state["playback_hour"] = 1

        st.success(f"{model_name} simulation completed.")

    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
        st.exception(exc)

summary = st.session_state.get("live_summary")
hourly_df = st.session_state.get("live_hourly")
ev_df = st.session_state.get("live_ev")

# --------------------------------------------------
# Display results from the most recent actual run
# --------------------------------------------------
if summary is not None and hourly_df is not None and ev_df is not None:

    max_hour = int(hourly_df["hour"].max())

    # Keep the playback position within the available simulation.
    if "playback_hour" not in st.session_state:
        st.session_state["playback_hour"] = 1

    st.session_state["playback_hour"] = min(
        max(1, int(st.session_state["playback_hour"])),
        max_hour,
    )

    st.divider()
    st.header(f"Simulation results — {summary['model']}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Hours completed", summary["hours_completed"])
    c2.metric("Total episode reward", f"{summary['total_reward']:.3f}")
    c3.metric("Total grid energy", f"{summary['total_grid_energy']:.3f}")
    c4.metric(
        "Departure penalty",
        f"{summary['total_departure_penalty']:.3f}",
    )

    st.caption(
        "Rewards are the negative of the environment's step costs. "
        "Grid-energy units follow the original environment."
    )

    # --------------------------------------------------
    # EV battery-level overview
    # --------------------------------------------------
    st.divider()
    st.header("EV battery levels over 24 hours")

    fig, ax = plt.subplots(figsize=(12, 5))

    for ev_name, group in ev_df.groupby("ev"):
        # Exclude absent periods from the plotted battery line.
        visible = group[group["present"]]
        if not visible.empty:
            ax.plot(
                visible["hour"],
                visible["battery_after_pct"],
                marker=".",
                linewidth=1.5,
                label=ev_name,
            )

    ax.set_xlabel("Simulation hour")
    ax.set_ylabel("Battery level (%)")
    ax.set_xticks(range(1, max_hour + 1))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------
    # Hour-by-hour playback controls
    # --------------------------------------------------
    st.divider()
    st.header("🔋 Hour-by-hour EV playback")

    st.write(
        "Use the controls to inspect the actual state of each EV "
        "during a particular simulation hour."
    )

    control_left, control_middle, control_right = st.columns(
        [1, 3, 1]
    )

    with control_left:
        if st.button(
            "◀ Previous",
            disabled=st.session_state["playback_hour"] <= 1,
            use_container_width=True,
        ):
            st.session_state["playback_hour"] -= 1

    with control_right:
        if st.button(
            "Next ▶",
            disabled=st.session_state["playback_hour"] >= max_hour,
            use_container_width=True,
        ):
            st.session_state["playback_hour"] += 1

    with control_middle:
        st.slider(
            "Select simulation hour",
            min_value=1,
            max_value=max_hour,
            key="playback_hour",
        )

    selected_hour = int(st.session_state["playback_hour"])

    hour_data = hourly_df[
        hourly_df["hour"] == selected_hour
    ]

    if not hour_data.empty:
        row = hour_data.iloc[0]

        h1, h2, h3 = st.columns(3)
        h1.metric("Simulation hour", f"{selected_hour} / {max_hour}")
        h2.metric("Hour reward", f"{row['reward']:.3f}")
        h3.metric("Hour grid energy", f"{row['grid_energy']:.3f}")

    # Show the actual state of every EV at the selected hour.
    selected_ev = ev_df[
        ev_df["hour"] == selected_hour
    ].copy()

    charging_count = int((selected_ev["status"] == "Charging").sum())
    discharging_count = int(
        (selected_ev["status"] == "Discharging").sum()
    )
    idle_count = int((selected_ev["status"] == "Idle").sum())
    absent_count = int((selected_ev["status"] == "Absent").sum())

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Charging", charging_count)
    s2.metric("Discharging", discharging_count)
    s3.metric("Idle", idle_count)
    s4.metric("Absent", absent_count)

    st.subheader(f"EV status at hour {selected_hour}")

    for _, ev in selected_ev.iterrows():
        with st.container(border=True):
            left, middle, right = st.columns([2, 3, 2])

            left.markdown(f"**{ev['ev']}**")
            left.write(str(ev["status"]))

            before = float(ev["battery_before_pct"])
            after = float(ev["battery_after_pct"])

            with middle:
                st.caption(
                    f"Battery: {before:.1f}% → {after:.1f}%"
                )
                st.progress(
                    int(round(max(0, min(100, after))))
                )

            if bool(ev["present"]):
                departure_text = (
                    f"{ev['departure_in_hours']:.0f} hour(s)"
                    if ev["departure_in_hours"] > 0
                    else "Not available"
                )
            else:
                departure_text = "Not present"

            right.caption("Time until departure")
            right.write(departure_text)
            right.caption(f"Model action: {ev['action']:.3f}")

    # Detailed table for the selected hour.
    st.subheader("Detailed EV data")

    display_ev = selected_ev[
        [
            "ev",
            "present",
            "battery_before_pct",
            "battery_after_pct",
            "departure_in_hours",
            "action",
            "status",
        ]
    ].rename(
        columns={
            "ev": "EV",
            "present": "Present",
            "battery_before_pct": "Battery before (%)",
            "battery_after_pct": "Battery after (%)",
            "departure_in_hours": "Hours until departure",
            "action": "Model action",
            "status": "Charging status",
        }
    )

    st.dataframe(
        display_ev.round(3),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------
    # Rewards and hourly data
    # --------------------------------------------------
    st.divider()
    st.header("Hourly performance")

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        hourly_df["hour"],
        hourly_df["reward"],
        marker="o",
        label="Reward",
    )

    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Simulation hour")
    ax.set_ylabel("Reward")
    ax.set_xticks(range(1, max_hour + 1))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Hourly results table")
    st.dataframe(
        hourly_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    download_left, download_right = st.columns(2)

    with download_left:
        st.download_button(
            "Download hourly results CSV",
            data=hourly_df.to_csv(index=False).encode("utf-8"),
            file_name="live_hourly_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download_right:
        st.download_button(
            "Download EV-level results CSV",
            data=ev_df.to_csv(index=False).encode("utf-8"),
            file_name="live_ev_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info(
        "Choose a model above and click Run 24-Hour Simulation "
        "to generate results."
    )

# --------------------------------------------------
# PPO vs Priority DQN: 30-scenario evaluation
# --------------------------------------------------
st.divider()
st.header("PPO vs Priority DQN — 30-Scenario Evaluation")
st.caption(
    "These results are loaded from the saved evaluation CSV and graphs. "
    "They are separate from the single simulation run above."
)

COMPARISON_DIR = REPORTS / "comparison_graphs"
COMPARISON_CSV = REPORTS / "model_comparison.csv"

if COMPARISON_CSV.exists():
    try:
        comparison_df = pd.read_csv(COMPARISON_CSV)
        required_columns = {
            "scenario_seed", "model", "total_reward",
            "grid_energy", "departure_penalty"
        }

        if not required_columns.issubset(comparison_df.columns):
            st.warning(
                "The comparison CSV is missing expected columns. "
                "Run training/compare_models.py again to regenerate it."
            )
        else:
            comparison_summary = (
                comparison_df.groupby("model")
                .agg(
                    mean_reward=("total_reward", "mean"),
                    std_reward=("total_reward", "std"),
                    mean_grid_energy=("grid_energy", "mean"),
                    mean_departure_penalty=("departure_penalty", "mean"),
                    scenarios=("scenario_seed", "nunique"),
                )
                .reindex(["PPO", "Priority DQN"])
                .dropna(how="all")
            )

            if not comparison_summary.empty:
                st.subheader("Summary metrics")
                metric_cols = st.columns(4)

                if "PPO" in comparison_summary.index:
                    ppo_row = comparison_summary.loc["PPO"]
                    metric_cols[0].metric(
                        "PPO mean reward", f"{ppo_row['mean_reward']:.3f}"
                    )
                    metric_cols[1].metric(
                        "PPO grid energy", f"{ppo_row['mean_grid_energy']:.3f}"
                    )

                if "Priority DQN" in comparison_summary.index:
                    dqn_row = comparison_summary.loc["Priority DQN"]
                    metric_cols[2].metric(
                        "Priority DQN mean reward",
                        f"{dqn_row['mean_reward']:.3f}"
                    )
                    metric_cols[3].metric(
                        "Priority DQN grid energy",
                        f"{dqn_row['mean_grid_energy']:.3f}"
                    )

                st.subheader("Model comparison table")
                summary_display = comparison_summary.rename(
                    columns={
                        "mean_reward": "Mean reward",
                        "std_reward": "Reward standard deviation",
                        "mean_grid_energy": "Mean grid energy",
                        "mean_departure_penalty": "Mean departure penalty",
                        "scenarios": "Scenarios",
                    }
                )
                st.dataframe(
                    summary_display.round(3),
                    use_container_width=True,
                )

                st.subheader("Comparison graphs")
                graph_specs = [
                    ("mean_reward.png", "Mean episode reward"),
                    ("grid_energy.png", "Mean grid energy"),
                    ("departure_penalty.png", "Mean departure penalty"),
                    ("reward_across_scenarios.png", "Reward across evaluation scenarios"),
                ]

                for start in range(0, len(graph_specs), 2):
                    graph_cols = st.columns(2)
                    for col, (filename, caption) in zip(
                        graph_cols, graph_specs[start:start + 2]
                    ):
                        graph_path = COMPARISON_DIR / filename
                        with col:
                            if graph_path.exists():
                                st.image(
                                    str(graph_path),
                                    caption=caption,
                                    use_container_width=True,
                                )
                            else:
                                st.info(
                                    f"Graph not found: {filename}. "
                                    "Run training/plot_comparison.py to create it."
                                )

                st.subheader("Individual scenario results")
                st.dataframe(
                    comparison_df.sort_values(
                        ["scenario_seed", "model"]
                    ).round(3),
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download 30-scenario comparison CSV",
                    data=comparison_df.to_csv(index=False).encode("utf-8"),
                    file_name="model_comparison.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.info(
                    "Interpretation: a mean reward closer to zero is better "
                    "in this environment. Grid energy and departure penalty "
                    "are separate metrics, so lower grid energy alone does not "
                    "necessarily mean better charging outcomes."
                )
            else:
                st.warning(
                    "The comparison CSV does not contain PPO and Priority DQN results."
                )
    except Exception as exc:
        st.error(f"Could not load comparison results: {exc}")
else:
    st.info(
        "No 30-scenario comparison results found yet. Run "
        "training/compare_models.py first."
    )

# --------------------------------------------------
# RL models vs rule-based baselines (same 30 scenarios)
# --------------------------------------------------
BASELINE_CSV = REPORTS / "baseline_comparison.csv"

if COMPARISON_CSV.exists() and BASELINE_CSV.exists():
    st.divider()
    st.header("RL models vs rule-based baselines")
    st.caption(
        "All policies are evaluated on the same 30 seeded scenarios "
        "(seeds 1000–1029). Baselines: training/evaluate_baselines.py."
    )

    all_results = pd.concat(
        [pd.read_csv(COMPARISON_CSV), pd.read_csv(BASELINE_CSV)],
        ignore_index=True,
    )
    bench = (
        all_results.groupby("model")
        .agg(
            mean_reward=("total_reward", "mean"),
            std_reward=("total_reward", "std"),
            mean_grid_energy=("grid_energy", "mean"),
            mean_departure_penalty=("departure_penalty", "mean"),
        )
        .sort_values("mean_reward", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 3.8))
    colors = [
        "#16a34a" if name in ("PPO", "Priority DQN") else "#9ca3af"
        for name in bench.index
    ]
    ax.barh(bench.index[::-1], bench["mean_reward"][::-1],
            xerr=bench["std_reward"][::-1], color=colors[::-1], capsize=3)
    ax.set_xlabel("Mean episode reward (closer to 0 is better)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        bench.rename(columns={
            "mean_reward": "Mean reward",
            "std_reward": "Std",
            "mean_grid_energy": "Mean grid energy",
            "mean_departure_penalty": "Mean departure penalty",
        }).round(2),
        use_container_width=True,
    )
    st.info(
        "Green bars are learned policies, grey bars are hand-written rules. "
        "PPO beats every baseline; Priority DQN uses far less grid energy "
        "but pays a larger penalty for under-charged departures."
    )

# --------------------------------------------------
# Previously saved experiment graphs
# --------------------------------------------------
st.divider()
st.header("Previously saved evaluation graphs")

saved_figures = [
    ("mean_reward_comparison.png", "PPO, DQN and baseline comparison"),
    ("episode_reward_distribution.png", "Episode reward distribution"),
    ("ppo_hourly_reward.png", "PPO hourly reward"),
    ("ppo_hourly_actions.png", "PPO hourly actions"),
]

found_figure = False

for filename, caption in saved_figures:
    image_path = FIGURES / filename

    if image_path.exists():
        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True,
        )
        found_figure = True

if not found_figure:
    st.caption("No saved evaluation graphs found yet.")

st.divider()
st.caption("Smart EV Charging Scheduler | PPO and DQN research project")

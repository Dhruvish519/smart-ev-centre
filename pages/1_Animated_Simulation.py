from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Project imports
# This file belongs in: pages/1_Animated_Simulation.py
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.live_simulation import run_live_simulation

st.set_page_config(
    page_title="Animated EV Simulation",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Animated EV Charging Simulation")
st.caption(
    "Run a trained reinforcement-learning model and animate its "
    "24-hour EV charging decisions."
)

st.info(
    "This page animates recorded results from an actual model run. "
    "It does not control physical chargers or simulate a live hardware connection."
)

# ------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------
defaults = {
    "anim_summary": None,
    "anim_hourly": None,
    "anim_ev": None,
    "anim_model": "PPO",
    "anim_hour": 1,
    "anim_playing": False,
    "anim_speed": 1,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ------------------------------------------------------------
# Run simulation controls
# ------------------------------------------------------------
st.header("1. Run a model simulation")

with st.form("animation_run_form"):
    selected_model = st.selectbox(
        "Choose trained model",
        ["PPO", "Priority DQN"],
        index=0 if st.session_state["anim_model"] == "PPO" else 1,
    )
    seed = st.number_input(
        "Scenario seed (same seed = same EV arrivals for every model)",
        min_value=0,
        value=1000,
        step=1,
    )
    submitted = st.form_submit_button(
        "▶ Run new 24-hour simulation",
        type="primary",
        use_container_width=True,
    )

if submitted:
    st.session_state["anim_playing"] = False
    try:
        with st.spinner(f"Running the trained {selected_model} model..."):
            summary, hourly_df, ev_df = run_live_simulation(selected_model, int(seed))

        # Validate expected columns so errors are clear.
        required_hourly = {"hour", "reward", "grid_energy"}
        required_ev = {
            "ev", "hour", "present", "battery_before_pct",
            "battery_after_pct", "departure_in_hours", "action", "status"
        }

        missing_hourly = required_hourly - set(hourly_df.columns)
        missing_ev = required_ev - set(ev_df.columns)

        if missing_hourly or missing_ev:
            raise ValueError(
                f"Unexpected simulation output columns. "
                f"Missing hourly: {sorted(missing_hourly)}; "
                f"missing EV: {sorted(missing_ev)}"
            )

        st.session_state["anim_summary"] = summary
        st.session_state["anim_hourly"] = hourly_df.copy()
        st.session_state["anim_ev"] = ev_df.copy()
        st.session_state["anim_model"] = selected_model
        st.session_state["anim_hour"] = int(hourly_df["hour"].min())
        st.session_state["anim_playing"] = False
        st.success(f"{selected_model} simulation completed. Ready to play.")
    except Exception as exc:
        st.error(f"Could not run simulation: {exc}")
        st.exception(exc)

summary = st.session_state["anim_summary"]
hourly_df = st.session_state["anim_hourly"]
ev_df = st.session_state["anim_ev"]

if summary is None or hourly_df is None or ev_df is None:
    st.warning("Run a simulation above to load the animation.")
    st.stop()

hourly_df = hourly_df.sort_values("hour").reset_index(drop=True)
ev_df = ev_df.sort_values(["hour", "ev"]).reset_index(drop=True)

hours = sorted(hourly_df["hour"].dropna().astype(int).unique().tolist())
if not hours:
    st.error("No hourly results were returned by the simulation.")
    st.stop()

min_hour, max_hour = min(hours), max(hours)
current_hour = int(st.session_state["anim_hour"])
if current_hour not in hours:
    current_hour = min_hour
    st.session_state["anim_hour"] = current_hour

# ------------------------------------------------------------
# Playback controls
# ------------------------------------------------------------
st.divider()
st.header("2. Simulation playback")

control_cols = st.columns([1, 1, 1, 2, 2])
with control_cols[0]:
    if st.button("⏮ Reset", use_container_width=True):
        st.session_state["anim_hour"] = min_hour
        st.session_state["anim_playing"] = False
        st.rerun()

with control_cols[1]:
    if st.button("◀ Previous", use_container_width=True, disabled=current_hour <= min_hour):
        st.session_state["anim_hour"] = max(min_hour, current_hour - 1)
        st.session_state["anim_playing"] = False
        st.rerun()

with control_cols[2]:
    if st.button("Next ▶", use_container_width=True, disabled=current_hour >= max_hour):
        st.session_state["anim_hour"] = min(max_hour, current_hour + 1)
        st.session_state["anim_playing"] = False
        st.rerun()

with control_cols[3]:
    speed_label = st.selectbox(
        "Playback speed",
        ["1 hour / 2 seconds", "1 hour / 1 second", "1 hour / 0.5 seconds"],
        index=1,
        key="anim_speed_label",
    )
    run_every = {
        "1 hour / 2 seconds": "2s",
        "1 hour / 1 second": "1s",
        "1 hour / 0.5 seconds": "500ms",
    }[speed_label]

with control_cols[4]:
    if st.session_state["anim_playing"]:
        if st.button("⏸ Pause", type="primary", use_container_width=True):
            st.session_state["anim_playing"] = False
            st.rerun()
    else:
        if st.button(
            "▶ Play animation",
            type="primary",
            use_container_width=True,
            disabled=current_hour >= max_hour,
        ):
            st.session_state["anim_playing"] = True
            st.rerun()

chosen_hour = st.slider(
    "Jump to simulation hour",
    min_value=min_hour,
    max_value=max_hour,
    value=int(st.session_state["anim_hour"]),
    key="anim_hour_slider",
)
if chosen_hour != st.session_state["anim_hour"]:
    st.session_state["anim_hour"] = int(chosen_hour)
    st.session_state["anim_playing"] = False
    st.rerun()

# Streamlit's fragment auto-runs only the animation area while playing.
# Requires a Streamlit version that supports st.fragment(run_every=...).
@st.fragment(run_every=run_every if st.session_state["anim_playing"] else None)
def render_animation_frame():
    frame_hour = int(st.session_state["anim_hour"])

    if st.session_state["anim_playing"]:
        if frame_hour >= max_hour:
            st.session_state["anim_playing"] = False
        else:
            frame_hour += 1
            st.session_state["anim_hour"] = frame_hour

    row_matches = hourly_df[hourly_df["hour"].astype(int) == frame_hour]
    if row_matches.empty:
        st.warning(f"No hourly metrics found for hour {frame_hour}.")
        return

    hour_row = row_matches.iloc[0]
    ev_now = ev_df[ev_df["hour"].astype(int) == frame_hour].copy()

    st.markdown(f"### 🕒 Hour {frame_hour} of {max_hour} — {st.session_state['anim_model']}")

    # Top-line metrics
    present_count = int(ev_now["present"].fillna(False).astype(bool).sum())
    charging_count = int((ev_now["status"] == "Charging").sum())
    discharging_count = int((ev_now["status"] == "Discharging").sum())
    idle_count = int((ev_now["status"] == "Idle").sum())
    absent_count = int((ev_now["status"] == "Absent").sum())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("EVs present", present_count)
    m2.metric("Charging", charging_count)
    m3.metric("Discharging", discharging_count)
    m4.metric("Grid energy", f"{float(hour_row['grid_energy']):.2f}")
    m5.metric("Hour reward", f"{float(hour_row['reward']):.3f}")

    # Battery chart up to the selected hour
    st.subheader("Battery level progression")
    history = ev_df[ev_df["hour"].astype(int) <= frame_hour]
    fig, ax = plt.subplots(figsize=(12, 4.5))

    for ev_name, group in history.groupby("ev"):
        visible = group[group["present"].fillna(False).astype(bool)]
        if not visible.empty:
            ax.plot(
                visible["hour"],
                visible["battery_after_pct"],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=str(ev_name),
            )

    ax.set_xlim(min_hour, max_hour)
    ax.set_ylim(0, 100)
    ax.set_xticks(hours)
    ax.set_xlabel("Simulation hour")
    ax.set_ylabel("Battery level (%)")
    ax.grid(True, alpha=0.25)
    ax.axvline(frame_hour, linestyle="--", linewidth=1)
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # EV status tiles
    st.subheader(f"EV station status at hour {frame_hour}")
    tile_cols = st.columns(2)
    for index, (_, ev) in enumerate(ev_now.iterrows()):
        with tile_cols[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{ev['ev']}**")
                st.write(f"Status: **{ev['status']}**")

                present = bool(ev["present"])
                after = ev["battery_after_pct"]

                if present and pd.notna(after):
                    st.progress(int(max(0, min(100, round(float(after))))))
                    st.caption(
                        f"Battery: {float(ev['battery_before_pct']):.1f}% → "
                        f"{float(after):.1f}%"
                    )
                    departure = ev["departure_in_hours"]
                    if pd.notna(departure):
                        st.caption(f"Hours until departure: {float(departure):.0f}")
                    st.caption(f"Model action: {float(ev['action']):.3f}")
                else:
                    st.caption("Vehicle is not present at this hour.")

    st.subheader("Hourly EV data")
    show_cols = [
        "ev", "present", "battery_before_pct", "battery_after_pct",
        "departure_in_hours", "action", "status"
    ]
    st.dataframe(
        ev_now[show_cols].rename(columns={
            "ev": "EV",
            "present": "Present",
            "battery_before_pct": "Battery before (%)",
            "battery_after_pct": "Battery after (%)",
            "departure_in_hours": "Hours until departure",
            "action": "Model action",
            "status": "Status",
        }).round(3),
        use_container_width=True,
        hide_index=True,
    )

    # Keep the slider synchronized with auto-playback on the next app rerun.
    st.caption(
        "Playback shows the saved output of this model run. "
        "Use Pause or the hour controls to inspect individual decisions."
    )

render_animation_frame()

# ------------------------------------------------------------
# Download current run data
# ------------------------------------------------------------
st.divider()
st.header("3. Export simulation results")
d1, d2 = st.columns(2)

with d1:
    st.download_button(
        "Download hourly simulation CSV",
        data=hourly_df.to_csv(index=False).encode("utf-8"),
        file_name="animated_simulation_hourly.csv",
        mime="text/csv",
        use_container_width=True,
    )

with d2:
    st.download_button(
        "Download EV-level simulation CSV",
        data=ev_df.to_csv(index=False).encode("utf-8"),
        file_name="animated_simulation_ev_details.csv",
        mime="text/csv",
        use_container_width=True,
    )

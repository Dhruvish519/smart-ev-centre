from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Charging Station View",
    page_icon="🔌",
    layout="wide",
)

st.title("🔌 Smart EV Charging Station")
st.caption("Inspect the charging-station state and model decisions at any hour of the simulation.")
st.info(
    "This is a visualization of the saved simulation results. It does not connect to real chargers or vehicles. "
    "First run a simulation on the Animated Simulation page."
)

# The animation page stores its latest completed run in these session-state keys.
hourly_df = st.session_state.get("anim_hourly")
ev_df = st.session_state.get("anim_ev")
model_name = st.session_state.get("anim_model", "Model not selected")
current_hour = st.session_state.get("anim_hour", None)

if not isinstance(hourly_df, pd.DataFrame) or not isinstance(ev_df, pd.DataFrame):
    st.warning("No simulation results are available yet.")
    st.markdown("Open **Animated Simulation**, select PPO or Priority DQN, and click **Run new 24-hour simulation**. Then return to this page.")
    st.stop()

required_ev = {
    "ev", "hour", "present", "battery_before_pct", "battery_after_pct",
    "departure_in_hours", "action", "status"
}
required_hourly = {"hour", "reward", "grid_energy"}
if required_ev - set(ev_df.columns) or required_hourly - set(hourly_df.columns):
    st.error("The current simulation data does not contain the columns required for this view. Run a fresh simulation from Animated Simulation.")
    st.stop()

hourly_df = hourly_df.copy().sort_values("hour")
ev_df = ev_df.copy().sort_values(["hour", "ev"])
hours = sorted(pd.to_numeric(hourly_df["hour"], errors="coerce").dropna().astype(int).unique().tolist())
if not hours:
    st.error("The simulation contains no valid hourly records.")
    st.stop()

# Use a separate widget key so this page does not conflict with the animation page's slider.
default_hour = int(current_hour) if current_hour is not None and int(current_hour) in hours else hours[0]
selected_hour = st.select_slider(
    "Select simulation hour",
    options=hours,
    value=default_hour,
    format_func=lambda h: f"Hour {h}",
    key="station_view_hour",
)

hour_rows = hourly_df[pd.to_numeric(hourly_df["hour"], errors="coerce") == selected_hour]
ev_now = ev_df[pd.to_numeric(ev_df["hour"], errors="coerce") == selected_hour].copy()
if hour_rows.empty or ev_now.empty:
    st.warning(f"No vehicle records are available for hour {selected_hour}.")
    st.stop()

hr = hour_rows.iloc[0]
present_mask = ev_now["present"].fillna(False).astype(bool)
present_count = int(present_mask.sum())
charging_count = int((ev_now["status"].astype(str).str.lower() == "charging").sum())
idle_count = int((ev_now["status"].astype(str).str.lower() == "idle").sum())
absent_count = int((ev_now["status"].astype(str).str.lower() == "absent").sum())

st.subheader(f"Station overview · {model_name} · Hour {selected_hour}")
metrics = st.columns(5)
metrics[0].metric("EVs present", present_count)
metrics[1].metric("Charging", charging_count)
metrics[2].metric("Idle", idle_count)
metrics[3].metric("Grid energy", f"{float(hr['grid_energy']):.2f}")
metrics[4].metric("Hour reward", f"{float(hr['reward']):.3f}")

st.divider()
st.subheader("Charging bays")
st.caption("Each bay shows the vehicle's state at the selected hour. Battery bars show battery-after values from the model run.")

# Render EV bays in a two-column grid for readability.
rows = list(ev_now.iterrows())
for start in range(0, len(rows), 2):
    cols = st.columns(2)
    for offset, (_, ev) in enumerate(rows[start:start + 2]):
        with cols[offset]:
            status = str(ev.get("status", "Unknown"))
            is_present = bool(ev.get("present", False))
            battery = pd.to_numeric(pd.Series([ev.get("battery_after_pct")]), errors="coerce").iloc[0]
            before = pd.to_numeric(pd.Series([ev.get("battery_before_pct")]), errors="coerce").iloc[0]
            action = pd.to_numeric(pd.Series([ev.get("action")]), errors="coerce").iloc[0]
            departure = pd.to_numeric(pd.Series([ev.get("departure_in_hours")]), errors="coerce").iloc[0]

            with st.container(border=True):
                header = st.columns([3, 2])
                header[0].markdown(f"### 🚗 {ev['ev']}")
                header[1].markdown(f"**{status}**")
                if is_present and pd.notna(battery):
                    safe_battery = max(0, min(100, int(round(float(battery)))))
                    st.progress(safe_battery, text=f"Battery {float(battery):.1f}%")
                    if pd.notna(before):
                        st.caption(f"Battery change: {float(before):.1f}% → {float(battery):.1f}%")
                    detail_cols = st.columns(2)
                    detail_cols[0].metric("Model action", f"{float(action):.3f}" if pd.notna(action) else "N/A")
                    detail_cols[1].metric("Departure in", f"{float(departure):.0f} h" if pd.notna(departure) else "N/A")
                else:
                    st.progress(0, text="Vehicle absent")
                    st.caption("This EV is not present in the station at the selected hour.")

st.divider()
st.subheader("Battery levels at this hour")
plot_df = ev_now.copy()
plot_df["battery_after_pct"] = pd.to_numeric(plot_df["battery_after_pct"], errors="coerce")
plot_df["present_label"] = plot_df["present"].fillna(False).astype(bool).map({True: "Present", False: "Absent"})
fig, ax = plt.subplots(figsize=(11, 4.2))
colors = ["tab:blue" if present else "lightgray" for present in plot_df["present"].fillna(False).astype(bool)]
values = plot_df["battery_after_pct"].fillna(0).clip(0, 100)
ax.bar(plot_df["ev"].astype(str), values, color=colors)
ax.set_ylim(0, 100)
ax.set_ylabel("Battery after (%)")
ax.set_xlabel("Electric vehicle")
ax.set_title(f"EV battery levels — hour {selected_hour}")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Model decisions table")
show_cols = ["ev", "present", "status", "battery_before_pct", "battery_after_pct", "departure_in_hours", "action"]
labels = {
    "ev": "EV", "present": "Present", "status": "Status",
    "battery_before_pct": "Battery before (%)", "battery_after_pct": "Battery after (%)",
    "departure_in_hours": "Hours until departure", "action": "Model action",
}
st.dataframe(ev_now[show_cols].rename(columns=labels).round(3), use_container_width=True, hide_index=True)

st.download_button(
    "Download selected-hour EV data (CSV)",
    data=ev_now[show_cols].to_csv(index=False).encode("utf-8"),
    file_name=f"charging_station_hour_{selected_hour}.csv",
    mime="text/csv",
    use_container_width=True,
)

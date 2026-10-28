import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.live_simulation import MODEL_PATHS, run_live_simulation
from ui import theme as T

MODELS = list(MODEL_PATHS)


@st.cache_data(show_spinner=False)
def simulate(model, seed):
    return run_live_simulation(model, seed)


def departures(ev):
    """Battery level of each car in its last hour at the station."""
    leaving = ev[ev["present"] & (ev["departure_in_hours"] == 1)]
    return leaving["battery_after_pct"]


T.header(
    "Smart EV Charging Scheduler",
    "A reinforcement-learning agent decides, every hour, which cars at a "
    "10-bay station to charge. It tries to buy as little grid electricity "
    "as possible while still sending every car away with a full battery.",
)

c1, c2, c3 = st.columns([2.2, 1, 1.1], vertical_alignment="bottom")
with c1:
    model = st.segmented_control("Agent", MODELS, default=MODELS[0],
                                 key="sim_model") or MODELS[0]
with c2:
    seed = int(st.number_input(
        "Day (scenario seed)", min_value=0, value=1000, step=1,
        help="Each seed is a different day of arrivals and departures. "
             "The same seed gives both agents exactly the same day.",
    ))
with c3:
    st.button("Run this day", type="primary", use_container_width=True,
              key="run_day")

try:
    with st.spinner(f"{model} is running the station for 24 hours…"):
        runs = {m: simulate(m, seed) for m in MODELS}
except Exception as exc:  # surface model/environment problems clearly
    st.error(f"The simulation could not run: {exc}")
    st.stop()

summary, hourly, ev = runs[model]
other = [m for m in MODELS if m != model][0]
o_summary, _, o_ev = runs[other]

dep = departures(ev)
full = int((dep >= 99.5).sum())
o_full = int((departures(o_ev) >= 99.5).sum())

st.write("")
T.stat_row([
    ("Total reward", f"{summary['total_reward']:.2f}", T.INK,
     f"{other}: {o_summary['total_reward']:.2f} on the same day"),
    ("Bought from grid", f"{summary['total_grid_energy']:.0f} kWh", T.GRID,
     f"{other}: {o_summary['total_grid_energy']:.0f} kWh"),
    ("Cars leaving full", f"{full} / {len(dep)}", T.BATTERY,
     f"{other}: {o_full} / {len(departures(o_ev))}"),
    ("Departure penalty", f"{summary['total_departure_penalty']:.2f}", T.PENALTY,
     f"{other}: {o_summary['total_departure_penalty']:.2f}"),
])
st.markdown(
    '<p class="ev-note">Reward = −(grid cost + penalty for under-charged '
    "departures). Closer to zero is better.</p>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
T.section("The day at a glance",
          "Grid purchases against free solar energy. Peak-price hours are shaded: "
          "a good agent shifts charging into cheap or sunny hours.")
st.plotly_chart(T.day_strip(hourly), use_container_width=True,
                config={"displayModeBar": False})

# ------------------------------------------------------------------
T.section("The station, hour by hour",
          "Each cell is a charging bay. Press play to watch the agent's "
          "decisions, or drag the slider to any hour.")

if "hour" not in st.session_state:
    st.session_state.hour = 1
if "playing" not in st.session_state:
    st.session_state.playing = False

b1, b2, _ = st.columns([1, 1, 4])
with b1:
    if st.button("⏸ Pause" if st.session_state.playing else "▶ Play the day",
                 use_container_width=True):
        if not st.session_state.playing and st.session_state.hour >= 24:
            st.session_state.hour = 1
        st.session_state.playing = not st.session_state.playing
        st.rerun()
with b2:
    if st.button("↺ Back to midnight", use_container_width=True):
        st.session_state.hour = 1
        st.session_state.playing = False
        st.rerun()


@st.fragment(run_every=0.9 if st.session_state.playing else None)
def station():
    if st.session_state.playing:
        if st.session_state.hour >= 24:
            st.session_state.playing = False
            st.rerun()
        else:
            st.session_state.hour += 1

    st.slider("Hour", 1, 24, key="hour", label_visibility="collapsed",
              format="Step %d of 24")
    h = st.session_state.hour
    row = hourly[hourly["hour"] == h].iloc[0]
    ev_now = ev[ev["hour"] == h]
    peak = row["price"] > hourly["price"].min() + 1e-9

    T.stat_row([
        ("cars plugged in", f"{int(row['evs_present'])}", T.INK,
         f"{h - 1:02d}:00 – {h % 24:02d}:00"),
        ("price per kWh", f"{row['price']:.2f}", T.GRID,
         "peak tariff" if peak else "off-peak tariff"),
        ("kWh of solar", f"{row['solar_kwh']:.1f}", T.SOLAR, None),
        ("kWh from grid", f"{row['grid_energy']:.1f}", T.GRID, None),
        ("reward this hour", f"{row['reward']:.2f}",
         T.PENALTY if row["departure_penalty"] > 0 else T.INK,
         f"incl. penalty {row['departure_penalty']:.2f}" if row["departure_penalty"] > 0 else "no penalty"),
    ], compact=True)
    st.write("")
    st.markdown(T.bays_html(ev_now), unsafe_allow_html=True)


station()

# ------------------------------------------------------------------
T.section("Battery levels across the day",
          "Rows are bays, columns are hours. Blank cells mean the bay was empty.")
st.plotly_chart(T.soc_heatmap(ev), use_container_width=True,
                config={"displayModeBar": False})

with st.expander("Raw data for this day"):
    t1, t2 = st.tabs(["Hourly", "Per car"])
    with t1:
        st.dataframe(hourly.round(3), use_container_width=True, hide_index=True)
        st.download_button("Download hourly CSV", hourly.to_csv(index=False),
                           f"{model}_seed{seed}_hourly.csv", "text/csv")
    with t2:
        st.dataframe(ev.round(2), use_container_width=True, hide_index=True)
        st.download_button("Download per-car CSV", ev.to_csv(index=False),
                           f"{model}_seed{seed}_cars.csv", "text/csv")

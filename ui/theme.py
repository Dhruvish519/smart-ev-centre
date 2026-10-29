"""Shared visual language for the Streamlit app.

Colours carry meaning everywhere in the app:
    amber  = solar energy        indigo    = grid energy (costs money)
    teal   = battery charge      raspberry = departure penalty
"""
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

INK = "#16233A"
MUTED = "#5A6880"
LINE = "#D9DFE8"
PAPER = "#F3F5F8"
SURFACE = "#FFFFFF"
GRID = "#4C5BD4"
SOLAR = "#F0A202"
BATTERY = "#12A08A"
PENALTY = "#D1495B"
RULE = "#A7B1C2"

MODEL_COLORS = {"PPO": GRID, "Priority DQN": BATTERY, "SAC": "#C2410C"}

FONT = "Barlow, system-ui, sans-serif"


def setup_page(title, icon="⚡"):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    # Blank lines would end the HTML block in Markdown, so strip them.
    css = "\n".join(line for line in _CSS.splitlines() if line.strip())
    st.html(css)


def header(title, subtitle):
    st.markdown(
        f'<div class="ev-header"><h1>{escape(title)}</h1>'
        f"<p>{escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def section(title, note=None):
    note_html = f"<p>{escape(note)}</p>" if note else ""
    st.markdown(
        f'<div class="ev-section"><h2>{escape(title)}</h2>{note_html}</div>',
        unsafe_allow_html=True,
    )


def stat_row(items, compact=False):
    """items: list of (label, value, colour, hint)."""
    cells = "".join(
        f'<div class="ev-stat" style="--c:{c}"><span class="v">{escape(v)}</span>'
        f'<span class="l">{escape(l)}</span>'
        + (f'<span class="h">{escape(h)}</span>' if h else "")
        + "</div>"
        for l, v, c, h in items
    )
    cls = "ev-stats compact" if compact else "ev-stats"
    st.markdown(f'<div class="{cls}">{cells}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# Plotly
# ------------------------------------------------------------------
def style(fig, height=320, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=14, color=INK),
        hoverlabel=dict(font_family=FONT, bgcolor=SURFACE),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, ticks="outside",
                     tickcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    return fig


def day_strip(hourly, current_hour=None):
    """Grid energy bars + solar area over a shaded tariff background."""
    fig = go.Figure()
    hours = hourly["hour"] - 1  # clock hour at the start of each step

    if "price" in hourly:
        expensive = hourly["price"] > hourly["price"].min() + 1e-9
        for h, exp in zip(hours, expensive):
            if exp:
                fig.add_vrect(x0=h - 0.5, x1=h + 0.5, fillcolor=GRID,
                              opacity=0.06, line_width=0, layer="below")

    if "solar_kwh" in hourly:
        fig.add_trace(go.Scatter(
            x=hours, y=hourly["solar_kwh"], name="Solar available (kWh)",
            mode="lines", line=dict(color=SOLAR, width=2.5, shape="spline"),
            fill="tozeroy", fillcolor="rgba(240,162,2,0.18)",
            hovertemplate="%{x}:00<br>Solar %{y:.1f} kWh<extra></extra>",
        ))

    fig.add_trace(go.Bar(
        x=hours, y=hourly["grid_energy"], name="Bought from grid (kWh)",
        marker_color=GRID,
        hovertemplate="%{x}:00<br>Grid %{y:.1f} kWh<extra></extra>",
    ))

    if current_hour is not None:
        fig.add_vline(x=current_hour - 1, line=dict(color=INK, width=1.5, dash="dot"))

    fig.update_xaxes(title="Time of day", tickmode="array", tickvals=list(range(0, 24, 3)),
                     ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)], range=[-0.6, 23.6])
    fig.update_yaxes(title="kWh")
    fig.add_annotation(x=1, y=1.0, xref="paper", yref="paper", showarrow=False,
                       xanchor="right", yanchor="bottom",
                       text="shaded = peak tariff", font=dict(color=MUTED, size=12))
    return style(fig, height=300)


def soc_heatmap(ev, upto_hour=None):
    """Battery level of every bay across the day (blank = bay empty)."""
    df = ev.copy()
    if upto_hour is not None:
        df.loc[df["hour"] > upto_hour, "battery_after_pct"] = np.nan
    grid = df.pivot(index="ev", columns="hour", values="battery_after_pct")
    order = sorted(grid.index, key=lambda s: int(s.split()[-1]))
    grid = grid.loc[order]
    labels = [f"Bay {s.split()[-1]}" for s in grid.index]
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=grid.columns - 1, y=labels,
        colorscale=[[0, "#FDE7EA"], [0.5, "#9ED9CD"], [1, BATTERY]],
        zmin=0, zmax=100, xgap=2, ygap=2,
        colorbar=dict(title="%", thickness=10, outlinewidth=0),
        hovertemplate="%{y}, %{x}:00<br>Battery %{z:.0f}%<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Time of day", tickmode="array", tickvals=list(range(0, 24, 3)),
                     ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)])
    return style(fig, height=330, legend=False)


# ------------------------------------------------------------------
# Station bays
# ------------------------------------------------------------------
def bays_html(ev_now):
    cards = []
    for _, ev in ev_now.sort_values("ev", key=lambda s: s.str.split().str[-1].astype(int)).iterrows():
        bay = ev["ev"].split()[-1]
        if not bool(ev["present"]):
            cards.append(
                f'<div class="bay empty"><div class="bay-top"><b>Bay {bay}</b>'
                f'<span class="chip">Free</span></div>'
                f'<div class="cell"><div class="fill" style="height:0%"></div></div>'
                f'<div class="bay-foot">No vehicle</div></div>'
            )
            continue
        before = float(ev["battery_before_pct"])
        after = float(ev["battery_after_pct"])
        delta = after - before
        status = str(ev["status"])
        cls = status.lower()
        dep = ev["departure_in_hours"]
        dep_txt = f"leaves in {int(dep)} h" if pd.notna(dep) and dep > 0 else "leaving"
        bolt = '<span class="bolt">⚡</span>' if status == "Charging" else ""
        cards.append(
            f'<div class="bay {cls}"><div class="bay-top"><b>Bay {bay}</b>'
            f'<span class="chip">{bolt}{escape(status)}</span></div>'
            f'<div class="cell"><div class="fill" style="height:{after:.0f}%"></div>'
            f'<span class="pct">{after:.0f}%</span></div>'
            f'<div class="bay-foot"><span class="d">{delta:+.0f}%</span> · {dep_txt}</div></div>'
        )
    return f'<div class="bays">{"".join(cards)}</div>'


_CSS = f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Semi+Condensed:wght@500;600;700&display=swap");
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea,
[data-testid="stWidgetLabel"], [data-testid="stSidebarNav"] {{
  font-family: {FONT} !important;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}

.ev-header h1 {{
  font-family: "Barlow Semi Condensed", {FONT}; font-weight: 700;
  font-size: clamp(2.1rem, 4vw, 3.1rem); line-height: 1.05; margin: 0 0 .35rem;
}}
.ev-header p {{ color: {MUTED}; font-size: 1.12rem; max-width: 62ch; margin: 0 0 1.4rem; }}
.ev-section {{ margin: 2.2rem 0 .6rem; }}
.ev-section h2 {{ font-size: 1.45rem; font-weight: 600; margin: 0; padding: 0; }}
.ev-section p {{ color: {MUTED}; margin: .2rem 0 0; max-width: 75ch; }}

.ev-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 0; background: {SURFACE}; border: 1px solid {LINE}; border-radius: 10px; overflow: hidden; }}
.ev-stat {{ padding: 1rem 1.2rem; border-left: 4px solid var(--c); display: flex; flex-direction: column;
  border-right: 1px solid {LINE}; }}
.ev-stat .v {{ font-family: "Barlow Semi Condensed", {FONT}; font-size: 2rem; font-weight: 600; line-height: 1.1; }}
.ev-stats.compact .ev-stat .v {{ font-size: 1.55rem; }}
.ev-stat .l {{ color: {INK}; font-weight: 500; }}
.ev-stat .h {{ color: {MUTED}; font-size: .86rem; }}

.bays {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }}
@media (max-width: 900px) {{ .bays {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.bay {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px; padding: .75rem .8rem .7rem; }}
.bay-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: .55rem; }}
.bay-top b {{ font-weight: 600; }}
.chip {{ font-size: .8rem; padding: .1rem .5rem; border-radius: 999px; background: {PAPER}; color: {MUTED}; }}
.bay.charging .chip {{ background: #DDF3EE; color: #0B6F60; }}
.bay.discharging .chip {{ background: #E5E8FB; color: {GRID}; }}
.cell {{ position: relative; height: 112px; border: 2px solid {INK}; border-radius: 8px;
  background: {PAPER}; overflow: hidden; margin: 0 18%; }}
.cell::before {{ content: ""; position: absolute; top: -2px; left: 35%; width: 30%; height: 4px; background: {INK}; }}
.fill {{ position: absolute; bottom: 0; left: 0; right: 0; background: {BATTERY};
  transition: height .6s ease; }}
.bay.discharging .fill {{ background: {GRID}; }}
.bay.idle .fill {{ background: #7CC8BA; }}
.pct {{ position: absolute; inset: 0; display: grid; place-items: center;
  font-family: "Barlow Semi Condensed", {FONT}; font-weight: 700; font-size: 1.35rem; color: {INK};
  text-shadow: 0 0 6px rgba(255,255,255,.9); }}
.bay.empty {{ background: transparent; border-style: dashed; }}
.bay.empty .cell {{ border-color: {LINE}; border-style: dashed; }}
.bay.empty .cell::before {{ background: {LINE}; }}
.bay-foot {{ color: {MUTED}; font-size: .86rem; margin-top: .5rem; text-align: center; }}
.bay-foot .d {{ color: {INK}; font-weight: 600; }}
.bolt {{ display: inline-block; margin-right: 2px; animation: pulse 1.6s ease-in-out infinite; }}
@keyframes pulse {{ 50% {{ opacity: .35; }} }}
@media (prefers-reduced-motion: reduce) {{ .bolt {{ animation: none; }} .fill {{ transition: none; }} }}

.stButton > button[kind="primary"], .stFormSubmitButton > button {{
  background: {INK}; border-color: {INK}; font-weight: 600; }}
.stButton > button[kind="primary"]:hover {{ background: {GRID}; border-color: {GRID}; }}
[data-testid="stTabs"] button p {{ font-size: 1.02rem; font-weight: 500; }}
[data-testid="stExpander"] {{ background: {SURFACE}; border-radius: 10px; }}
.ev-note {{ color: {MUTED}; font-size: .92rem; }}
</style>
"""

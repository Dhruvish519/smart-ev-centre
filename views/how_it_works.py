import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui import theme as T

T.header(
    "How it works",
    "The charging station is a Markov decision process: every hour the agent "
    "sees the station, picks an action, and is scored on cost and on how full "
    "the departing cars are.",
)

T.section("The station")
st.markdown(
    """
- **10 charging bays.** Cars arrive at random with a 20–50% charge and stay for 4–9 hours.
- **Batteries** hold 30 kWh and can take up to 10 kWh per hour.
- **Rooftop solar** supplies free energy during the day. Anything more is bought from the grid.
- **Two tariffs:** 0.05 per kWh off-peak (night) and 0.10 per kWh at peak (7:00–20:00).
- **One episode is one day:** 24 hourly decisions.
"""
)

T.section("What the agent sees, does and is scored on")
a, b, c = st.columns(3, gap="large")
with a:
    st.markdown("#### State · 28 numbers")
    st.markdown(
        "- Solar radiation now and for the next 3 hours\n"
        "- Electricity price now and for the next 3 hours\n"
        "- Battery level of each bay (10)\n"
        "- Hours until each car leaves (10)"
    )
with b:
    st.markdown("#### Action")
    st.markdown(
        "**PPO and SAC:** a charging rate between −1 and 1 for every bay "
        "(negative means sending energy back from that car).\n\n"
        "**Priority DQN:** one of 21 choices. Either do nothing, or pick a "
        "*ranking rule* and *how many* of the top-ranked cars to charge."
    )
with c:
    st.markdown("#### Reward")
    st.markdown("Minus the cost of grid electricity this hour, minus a penalty "
                "for every car that leaves without a full battery.\n\n"
                "A car leaving at 70% costs 0.36. A car leaving at 40% costs 1.44, "
                "so the penalty grows quickly as cars leave emptier.")

st.latex(r"r_t = -\Big(\underbrace{E^{grid}_t \, p_t}_{\text{grid cost}}"
         r" + \sum_{i \in \text{leaving at } t} \underbrace{\big(2(1-\text{SoC}_i)\big)^2}_{\text{departure penalty}}\Big)")

T.section("The three algorithms",
          "All three come from Stable-Baselines3 and get the same budget, seeds and "
          "evaluation. They differ in how they learn and in what action they control.")
st.markdown(
    """
| | **PPO** | **SAC** | **Priority DQN** |
|---|---|---|---|
| Family | Policy gradient, actor-critic | Maximum-entropy actor-critic | Value-based (Q-learning) |
| On- or off-policy | On-policy: learns only from its latest experience | Off-policy: reuses a replay buffer of past experience | Off-policy: replay buffer |
| Action | Continuous rate for each of the 10 bays | Continuous rate for each of the 10 bays | 1 of 21 discrete priority actions |
| Exploration | Gaussian noise on actions | Entropy bonus, tuned automatically | ε-greedy (100% random → 5%) |
| Key idea | Clipped updates keep each policy change small and stable | Maximise reward *and* randomness, so it keeps trying alternatives | Learn the value of each action; pick the best |
| Why it fits here | Standard baseline for continuous control | Sample-efficient; widely used for EV-charging scheduling | Our action design shrinks a huge action space to 21 interpretable choices |
"""
)

st.markdown(
    "**A note on SAC.** With standard settings SAC got stuck around −40: it learned to "
    "drain energy from some parked cars to charge others, which makes the grid bill "
    "that hour almost zero. The penalty only arrives hours later, when the drained cars "
    "leave, and SAC's one-step updates linked it back to the cause far too slowly. "
    "Using **8-step returns** (each update looks 8 hours ahead) fixed it. The Results "
    "page shows both learning curves."
)

T.section("The Priority DQN action design",
          "Controlling 10 bays with a discrete action space would need 2¹⁰ = 1,024 "
          "on/off combinations, or more. Instead the agent learns which rule to trust in which situation.")
st.markdown(
    """
| Rule | Charges first the cars with… |
|---|---|
| Least laxity | largest battery deficit ÷ time left |
| Largest deficit | the emptiest battery |
| Earliest departure | the least time left |
| Soft laxity | deficit ÷ √time left |
| Urgency-weighted | deficit × (1 + 1 / time left) |

Each rule is combined with a fraction (25 / 50 / 75 / 100% of the cars present). Together with "do nothing", that gives 21 actions.
"""
)

T.section("Training protocol")
st.markdown(
    """
| | |
|---|---|
| Library | Stable-Baselines3 (PyTorch), MLP policies |
| Algorithms | PPO, SAC, Priority DQN |
| Budget | 100,000 steps per agent (≈ 4,200 simulated days) |
| Seeds | 3 independent training runs per agent |
| Validation | 20 fixed days (seeds 500–519), scored every 5,000 steps. The best checkpoint is kept. |
| Test | 30 different days (seeds 1000–1029), used only for the final comparison |
| Baselines | Charge everything at full power, the Chargym rule-based controller, a cheap-hours rule, no charging |
"""
)

T.section("Limitations")
st.markdown(
    """
- One price schedule and one weather file: the agents have never seen other seasons or tariffs.
- The grid has unlimited capacity, so the station never has to share a limited connection between cars.
- Priority DQN only ever charges, so sending energy from cars back to the station (vehicle-to-grid) is only available to PPO and SAC.
"""
)

st.caption("Simulator: Chargym (Karatzinis et al., 2022, MIT licence). "
           "Agents, action design, evaluation and app: this project.")

# ⚡ Smart EV Charging Scheduler

Reinforcement-learning agents that decide **which electric vehicles to charge, and how hard, every hour of the day** at a 10-bay charging station with solar power and time-of-use electricity prices.

The goal is to **minimise the cost of grid electricity** while making sure every car **leaves with a full battery**. Two RL agents are trained and compared against hand-written rules on the same 30 simulated days, and an interactive Streamlit dashboard lets you watch the trained agents make decisions hour by hour.

[![CI](https://github.com/Dhruvish519/smart-ev-centre/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhruvish519/smart-ev-centre/actions/workflows/ci.yml)

---

## The problem

| | |
|---|---|
| **Station** | 10 charging bays, 30 kWh EV batteries, up to ~10 kWh per car per hour, rooftop PV |
| **Episode** | One day = 24 hourly decisions; EVs arrive and leave at random (20–50 % initial charge) |
| **State (28 values)** | Solar radiation and price now + next 3 h forecast; each bay's battery level and hours until departure |
| **Reward** | −(grid energy × price) − Σ (2 × (1 − SoC at departure))² for every car that leaves |

The trade-off: charging early is safe but may use expensive grid power; waiting for cheap hours or sunshine is cheaper but risks a car leaving under-charged.

## Approach

| Agent | Action space | Idea |
|---|---|---|
| **PPO** (continuous) | 10 values in [−1, 1] — charging/discharging rate per bay | Policy-gradient agent learns a fine-grained rate for each car |
| **Priority DQN** (discrete) — *own design* | 21 actions = idle, or one of 5 **priority rules** × 4 **fractions** (25/50/75/100 % of cars) | Instead of controlling 10 bays directly (impossibly large discrete space), the agent picks *which ranking rule* to apply (e.g. "battery deficit ÷ time left") and *how many* top-ranked cars to charge |

Both use Stable-Baselines3 with an MLP policy. PPO was trained for 10 k steps, Priority DQN for 20 k steps.

## Results (30 unseen scenarios, seeds 1000–1029)

| Policy | Mean reward ↑ | Grid energy | Departure penalty |
|---|---:|---:|---:|
| **PPO** | **−29.16 ± 3.97** | 283.8 | 8.75 |
| Priority DQN | −30.22 ± 3.36 | **194.5** | 16.99 |
| Cheap-hours rule | −30.66 ± 3.54 | 442.4 | 0.00 |
| Charge-all at max power | −31.22 ± 3.82 | 429.1 | 0.00 |
| Chargym rule-based controller | −31.31 ± 3.57 | 426.0 | 0.00 |
| No charging | −46.88 ± 4.82 | 0.0 | 46.88 |

- **PPO is the best policy overall**, beating every baseline. Its advantage over Priority DQN is statistically significant (paired t-test p = 0.013, Wilcoxon p = 0.016; better on 23 / 30 days).
- **Priority DQN uses 31 % less grid energy than PPO** (and ~55 % less than the rules), but leaves more cars under-charged — a different point on the cost-vs-satisfaction trade-off.
- Rule-based policies always fully charge cars but buy much more grid electricity.

Reproduce: `python training/compare_models.py` and `python training/evaluate_baselines.py`.

## Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

- **Home** — run PPO or Priority DQN for a 24-hour day (choose a seed so both models face the same day), battery curves, hour-by-hour playback, 30-scenario comparison and baselines
- **Animated Simulation** — auto-playing animation of the agent's decisions
- **Charging Station View** — per-bay view of the station at any hour

## Project structure

```
├── dashboard.py                  Streamlit app (home page)
├── pages/                        Extra Streamlit pages
├── Chargym_Charging_Station/
│   ├── envs/
│   │   ├── Charging_Station_Enviroment.py   Simulator (from Chargym)
│   │   ├── gymnasium_wrapper.py             Gymnasium API wrapper (PPO)
│   │   └── dqn_wrapper.py                   Priority-based discrete actions (DQN)
│   ├── utils/                    Energy, arrivals and reward calculations
│   └── Files/                    Weather data
├── training/
│   ├── train_ppo.py / train_dqn.py          Training
│   ├── compare_models.py                    30-scenario RL evaluation
│   ├── evaluate_baselines.py                Rule-based baselines
│   └── live_simulation.py                   Single-day run used by the dashboard
├── models/                       Trained models (final + checkpoints)
├── reports/                      Result CSVs and figures
├── tests/                        Smoke tests run in CI
└── Solvers/                      Original Chargym example solvers (reference only)
```

## Retrain

```bash
python training/train_ppo.py
python training/train_dqn.py
python training/compare_models.py
python training/plot_comparison.py
```

## Limitations and next steps

- Short training budgets (10 k / 20 k steps ≈ 400–800 simulated days) and unequal between the two agents; longer, equal-budget training with several random seeds would make the comparison stronger.
- Priority DQN never discharges, so vehicle-to-grid (V2G) is only available to PPO.
- One fixed price profile and one weather file; results may not transfer to other tariffs or seasons.
- The grid has no capacity limit (an assumption of the simulator).

## Credits

The charging-station simulator is based on **Chargym** by the ConvCAO research group (CERTH), MIT License — Karatzinis et al., *"Chargym: An EV Charging Station Model for Controller Benchmarking"*, AIAI 2022 ([paper](docs/Chargym_preprint_paper.pdf), [original repo](https://github.com/georkara/Chargym-Charging-Station)).

This project adds: the Gymnasium wrapper, the priority-based DQN action design, the PPO/DQN training and evaluation pipeline, baseline benchmarking, and the interactive dashboard. Cross-platform fixes were also made to the simulator.

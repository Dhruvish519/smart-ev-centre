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

Both use Stable-Baselines3 with MLP policies.

### Training protocol

| | |
|---|---|
| Budget | 100,000 steps per agent (≈ 4,200 simulated days), equal for both |
| Seeds | 3 independent training runs per agent |
| Validation | 20 fixed days (seeds 500–519), scored every 5,000 steps; the best checkpoint is kept |
| Test | 30 different days (seeds 1000–1029), used only for the final numbers below |

## Results (30 held-out test days)

| Policy | Mean reward ↑ | Grid energy (kWh/day) | Departure penalty |
|---|---:|---:|---:|
| **Priority DQN** (ours) | **−25.67 ± 2.82** | 361 | **3.94** |
| PPO | −27.07 ± 3.17 | 280 | 7.92 |
| Cheap-hours rule | −30.66 ± 3.54 | 442 | 0.00 |
| Charge-all at max power | −31.22 ± 3.82 | 429 | 0.00 |
| Chargym rule-based controller | −31.31 ± 3.57 | 426 | 0.00 |
| No charging | −46.88 ± 4.82 | 0 | 46.88 |

RL rows are averaged over the 3 training seeds.

- **Priority DQN is the best policy.** It beats PPO on 26 of 30 test days (paired t-test p < 10⁻⁶) and the best hand-written rule by 16%.
- It learned to **buy 79% of its grid electricity off-peak**, when the tariff is half price (PPO: 59%).
- **Equal, longer training changed the conclusion.** With the original budgets (PPO 10k steps, DQN 20k), PPO looked better (−29.16 vs −30.22). The learning curves show why: DQN only starts improving after ~25k steps, while PPO peaks early (~15k) and then drifts down, so best-checkpoint selection matters.

Reproduce:

```bash
python training/train_v2.py --algo ppo --seed 0   # repeat for seeds 1, 2 and --algo dqn
python training/evaluate_v2.py                     # test-set results + deployed models
python training/evaluate_baselines.py
```

## Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

- **Simulate a day:** run either agent on any day, compare it with the other agent on the same day, and watch the 10 charging bays fill hour by hour
- **Results:** leaderboard against baselines, cost-versus-penalty trade-off, day-by-day head-to-head, learning curves
- **How it works:** the MDP, reward, action design and training protocol

## Project structure

```
├── dashboard.py                  Streamlit entry point
├── views/                        App pages (simulate, results, how it works)
├── ui/theme.py                   Shared colours, charts and station visual
├── Chargym_Charging_Station/
│   ├── envs/
│   │   ├── Charging_Station_Enviroment.py   Simulator (from Chargym)
│   │   ├── gymnasium_wrapper.py             Gymnasium API wrapper (PPO)
│   │   └── dqn_wrapper.py                   Priority-based discrete actions (DQN)
│   ├── utils/                    Energy, arrivals and reward calculations
│   └── Files/                    Weather data
├── training/
│   ├── train_v2.py / evaluate_v2.py         Main training + test evaluation
│   ├── train_ppo.py / train_dqn.py          Original (v1) training
│   ├── compare_models.py                    30-scenario RL evaluation
│   ├── evaluate_baselines.py                Rule-based baselines
│   └── live_simulation.py                   Single-day run used by the dashboard
├── models/v2/                    Retrained models (3 seeds each; *_final = deployed)
├── models/ppo, dqn_priority/     Original v1 models
├── reports/                      Result CSVs and figures
├── tests/                        Smoke tests run in CI
└── Solvers/                      Original Chargym example solvers (reference only)
```

## Limitations and next steps

- One price schedule and one weather file. Training on randomised tariffs and seasons would test whether the agents generalise.
- The grid has no capacity limit. A connection cap would make the scheduling problem harder and more realistic.
- Priority DQN never discharges, so vehicle-to-grid (V2G) is only available to PPO.
- Next: reward shaping or a constrained formulation (for example, "every car must reach at least 90%"), plus SAC or TD3 for the continuous action space.

## Credits

The charging-station simulator is based on **Chargym** by the ConvCAO research group (CERTH), MIT License — Karatzinis et al., *"Chargym: An EV Charging Station Model for Controller Benchmarking"*, AIAI 2022 ([paper](docs/Chargym_preprint_paper.pdf), [original repo](https://github.com/georkara/Chargym-Charging-Station)).

This project adds: the Gymnasium wrapper, the priority-based DQN action design, the multi-seed training and validation/test evaluation protocol, baseline benchmarking, and the interactive dashboard. Cross-platform fixes were also made to the simulator.


import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, SAC

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Chargym_Charging_Station.envs.gymnasium_wrapper import (
    ChargingGymnasiumWrapper,
)
from Chargym_Charging_Station.envs.dqn_wrapper import (
    ChargingDQNWrapper,
)

# Retrained models (training/train_v2.py + evaluate_v2.py) are used when
# present; otherwise fall back to the original models.
_V2 = ROOT / "models" / "v2"
MODEL_PATHS = {
    "Priority DQN": (
        _V2 / "dqn_final.zip" if (_V2 / "dqn_final.zip").exists()
        else ROOT / "models" / "dqn_priority" / "charging_dqn_priority_final.zip"
    ),
    "PPO": (
        _V2 / "ppo_final.zip" if (_V2 / "ppo_final.zip").exists()
        else ROOT / "models" / "ppo" / "charging_ppo_final.zip"
    ),
    "SAC": _V2 / "sac_final.zip",
}
# Only offer agents whose trained model is actually present.
MODEL_PATHS = {name: path for name, path in MODEL_PATHS.items() if path.exists()}

# Continuous-action agents act directly on per-bay charging rates.
CONTINUOUS = {"PPO": PPO, "SAC": SAC}


def run_live_simulation(model_name: str, seed=None):
    """Run a trained model and record hourly and EV-level results.

    If ``seed`` is given, the EV arrival/departure scenario is reproducible,
    so different models can be compared on exactly the same day.
    """

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported model: {model_name}")

    model_path = MODEL_PATHS[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    if model_name in CONTINUOUS:
        env = ChargingGymnasiumWrapper()
        model = CONTINUOUS[model_name].load(str(model_path), device="cpu")
    else:
        env = ChargingDQNWrapper()
        model = DQN.load(str(model_path), device="cpu")

    hourly_rows = []
    ev_rows = []
    total_reward = 0.0

    try:
        # The legacy environment draws EV scenarios from NumPy's global RNG.
        if seed is not None:
            np.random.seed(int(seed))
        obs, _ = env.reset(seed=seed)
        legacy = env.legacy_env

        for hour in range(24):
            # Record the environment state before the action.
            present = (
                legacy.Invalues["present_cars"][:, hour]
                .astype(bool)
                .copy()
            )

            battery_before = (
                legacy.BOC[:, hour].astype(float).copy()
            )

            departure = np.zeros(10, dtype=float)

            for car in range(10):
                if present[car]:
                    departures = legacy.Invalues["DepartureT"][car]

                    future_departures = [
                        float(value)
                        for value in departures
                        if float(value) > hour
                    ]

                    if future_departures:
                        departure[car] = (
                            min(future_departures) - hour
                        )

            # Get an action from the actual trained policy.
            policy_action, _ = model.predict(
                obs,
                deterministic=True,
            )

            if model_name in CONTINUOUS:
                action_vector = np.asarray(
                    policy_action,
                    dtype=float,
                ).reshape(-1)
            else:
                action_vector = None

            obs, reward, terminated, truncated, _ = env.step(
                policy_action
            )

            total_reward += float(reward)

            # The DQN wrapper converts its discrete action into
            # a per-EV action vector.
            if model_name == "Priority DQN":
                action_vector = (
                    env.last_action_vector.astype(float).copy()
                )

            battery_after = (
                legacy.BOC[:, hour + 1].astype(float).copy()
            )

            hourly_rows.append({
                "hour": hour + 1,
                "price": float(legacy.Energy["Price"][0, hour]),
                "solar_kwh": float(legacy.Energy["Renewable"][0, hour]),
                "evs_present": int(present.sum()),
                "reward": float(reward),
                "grid_energy": float(legacy.Grid_Evol[-1]),
                "departure_penalty": float(
                    legacy.Penalty_Evol[-1]
                ),
                "step_cost": float(legacy.Cost_History[-1]),
                "action_mean": float(np.mean(action_vector)),
                "action_min": float(np.min(action_vector)),
                "action_max": float(np.max(action_vector)),
            })

            # Record all 10 EVs, distinguishing absent EVs
            # from vehicles that are actually present.
            for car in range(10):
                is_present = bool(present[car])

                if not is_present:
                    status = "Absent"
                    battery_before_pct = np.nan
                    battery_after_pct = np.nan
                    ev_action = 0.0

                else:
                    battery_before_pct = float(
                        np.clip(
                            battery_before[car], 0.0, 1.0
                        ) * 100
                    )

                    battery_after_pct = float(
                        np.clip(
                            battery_after[car], 0.0, 1.0
                        ) * 100
                    )

                    ev_action = float(action_vector[car])

                    if ev_action > 0.05:
                        status = "Charging"
                    elif ev_action < -0.05:
                        status = "Discharging"
                    else:
                        status = "Idle"

                ev_rows.append({
                    "hour": hour + 1,
                    "ev": f"EV {car + 1}",
                    "present": is_present,
                    "battery_before_pct": battery_before_pct,
                    "battery_after_pct": battery_after_pct,
                    "departure_in_hours": (
                        float(departure[car])
                        if is_present
                        else np.nan
                    ),
                    "action": ev_action,
                    "status": status,
                })

            if terminated or truncated:
                break

        output_dir = ROOT / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        hourly_df = pd.DataFrame(hourly_rows)
        ev_df = pd.DataFrame(ev_rows)

        hourly_df.to_csv(
            output_dir / "live_hourly_results.csv",
            index=False,
        )

        ev_df.to_csv(
            output_dir / "live_ev_results.csv",
            index=False,
        )

        summary = {
            "model": model_name,
            "hours_completed": len(hourly_df),
            "total_reward": total_reward,
            "total_grid_energy": float(
                hourly_df["grid_energy"].sum()
            ),
            "total_departure_penalty": float(
                hourly_df["departure_penalty"].sum()
            ),
            "total_step_cost": float(
                hourly_df["step_cost"].sum()
            ),
        }

        return summary, hourly_df, ev_df

    finally:
        env.close()

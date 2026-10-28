
import sys
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN

# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Chargym_Charging_Station.envs.gymnasium_wrapper import (
    ChargingGymnasiumWrapper,
)
from Chargym_Charging_Station.envs.dqn_wrapper import (
    ChargingDQNWrapper,
)

PPO_PATH = ROOT / "models" / "ppo" / "charging_ppo_final.zip"

DQN_PATH = (
    ROOT
    / "models"
    / "dqn_priority"
    / "charging_dqn_priority_final.zip"
)

OUTPUT_PATH = ROOT / "reports" / "model_comparison.csv"

NUM_SCENARIOS = 30
START_SEED = 1000
EPISODE_LENGTH = 24


# ============================================================
# 2. CREATE ENVIRONMENT
# ============================================================

def create_environment(model_name):
    if model_name == "PPO":
        return ChargingGymnasiumWrapper()

    if model_name == "Priority DQN":
        return ChargingDQNWrapper()

    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# 3. CONVERT NESTED DATA TO A STABLE REPRESENTATION
# ============================================================

def make_json_safe(value):
    """Convert nested lists, dictionaries and NumPy values to JSON."""

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return repr(value)


# ============================================================
# 4. CREATE INITIAL SCENARIO SIGNATURE
# ============================================================

def get_scenario_signature(legacy_env):
    """
    Fingerprint initial EV conditions.

    Uses the actual Invalues dictionary in ChargingEnv.
    """

    initial_values = legacy_env.Invalues

    keys_to_check = [
        "present_cars",
        "BOC",
        "ArrivalT",
        "DepartureT",
        "evolution_of_cars",
    ]

    digest = hashlib.sha256()

    for key in keys_to_check:
        if key not in initial_values:
            raise KeyError(
                f"Missing '{key}' in Invalues. "
                f"Available keys: {list(initial_values.keys())}"
            )

        serialized = json.dumps(
            make_json_safe(initial_values[key]),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        digest.update(key.encode("utf-8"))
        digest.update(serialized.encode("utf-8"))

    return digest.hexdigest()


# ============================================================
# 5. RUN ONE EPISODE
# ============================================================

def run_episode(model, model_name, seed):
    env = create_environment(model_name)

    try:
        # The legacy environment uses NumPy's global RNG.
        # Seed immediately before reset to reproduce EV conditions.
        np.random.seed(seed)

        obs, _ = env.reset(seed=seed)

        legacy = env.legacy_env

        # Save initial conditions before the model takes any actions.
        scenario_signature = get_scenario_signature(legacy)

        total_reward = 0.0
        steps_completed = 0

        for _ in range(EPISODE_LENGTH):
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += float(reward)
            steps_completed += 1

            if terminated or truncated:
                break

        # Collect episode metrics.
        grid_energy = float(
            np.nansum(
                np.asarray(legacy.Grid_Evol, dtype=float)
            )
        )

        departure_penalty = float(
            np.nansum(
                np.asarray(legacy.Penalty_Evol, dtype=float)
            )
        )

        return {
            "scenario_seed": seed,
            "model": model_name,
            "total_reward": total_reward,
            "grid_energy": grid_energy,
            "departure_penalty": departure_penalty,
            "steps_completed": steps_completed,
            "scenario_signature": scenario_signature,
        }

    finally:
        env.close()


# ============================================================
# 6. MAIN COMPARISON
# ============================================================

def main():
    print("=" * 65)
    print("PPO VS PRIORITY DQN — 30-SCENARIO EVALUATION")
    print("=" * 65)

    if not PPO_PATH.exists():
        raise FileNotFoundError(
            f"PPO model not found:\n{PPO_PATH}"
        )

    if not DQN_PATH.exists():
        raise FileNotFoundError(
            f"Priority DQN model not found:\n{DQN_PATH}"
        )

    print("\nLoading trained models...")

    ppo_model = PPO.load(str(PPO_PATH))
    dqn_model = DQN.load(str(DQN_PATH))

    models = [
        ("PPO", ppo_model),
        ("Priority DQN", dqn_model),
    ]

    results = []
    scenario_signatures = {}

    # --------------------------------------------------------
    # Evaluate both models on every seed.
    # --------------------------------------------------------

    for index in range(NUM_SCENARIOS):
        seed = START_SEED + index

        print(
            f"\nScenario {index + 1}/{NUM_SCENARIOS} "
            f"(seed={seed})"
        )

        for model_name, model in models:
            result = run_episode(
                model=model,
                model_name=model_name,
                seed=seed,
            )

            signature = result["scenario_signature"]

            if seed not in scenario_signatures:
                scenario_signatures[seed] = signature

            elif scenario_signatures[seed] != signature:
                raise RuntimeError(
                    "\nSCENARIO MISMATCH DETECTED\n"
                    f"Seed: {seed}\n"
                    f"Model: {model_name}\n"
                    "Initial scenario data differs between models. "
                    "Results have not been accepted as a fair comparison."
                )

            results.append({
                "scenario_seed": result["scenario_seed"],
                "model": result["model"],
                "total_reward": result["total_reward"],
                "grid_energy": result["grid_energy"],
                "departure_penalty": result["departure_penalty"],
                "steps_completed": result["steps_completed"],
            })

            print(
                f"{model_name:15} | "
                f"Reward: {result['total_reward']:9.3f} | "
                f"Grid energy: {result['grid_energy']:9.3f} | "
                f"Departure penalty: "
                f"{result['departure_penalty']:9.3f}"
            )

    # ========================================================
    # 7. SAVE DETAILED RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # 8. SUMMARY STATISTICS
    # ========================================================

    summary = (
        results_df
        .groupby("model")
        .agg(
            mean_reward=("total_reward", "mean"),
            std_reward=("total_reward", "std"),
            mean_grid_energy=("grid_energy", "mean"),
            std_grid_energy=("grid_energy", "std"),
            mean_departure_penalty=("departure_penalty", "mean"),
            std_departure_penalty=("departure_penalty", "std"),
        )
        .reset_index()
    )

    print("\n" + "=" * 65)
    print("SUMMARY ACROSS SCENARIOS")
    print("=" * 65)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\n" + "=" * 65)
    print("SCENARIO VERIFICATION: PASSED")
    print(
        f"Matching initial scenario fingerprints: "
        f"{len(scenario_signatures)}/{NUM_SCENARIOS}"
    )
    print("=" * 65)

    print(f"\nDetailed results saved to:\n{OUTPUT_PATH}")

    print("\nInterpretation:")
    print("- Mean reward closer to zero is better.")
    print("- Lower grid energy means less energy drawn from the grid.")
    print("- Lower departure penalty indicates better battery "
          "satisfaction at departure under this environment's reward.")

    print(
        "\nNote: Matching fingerprints verifies the selected initial "
        "scenario data. It does not by itself prove that every "
        "environment variable or evaluation condition is identical."
    )


if __name__ == "__main__":
    main()

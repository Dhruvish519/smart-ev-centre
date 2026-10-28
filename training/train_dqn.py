
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from Chargym_Charging_Station.envs.dqn_wrapper import (
    ChargingDQNWrapper,
)


# Save the new experiment separately from the original DQN.
MODEL_DIR = PROJECT_ROOT / "models" / "dqn_priority"
LOG_DIR = PROJECT_ROOT / "logs" / "dqn_priority"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    env = Monitor(
        ChargingDQNWrapper(),
        filename=str(LOG_DIR / "training_monitor.csv"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5_000,
        save_path=str(MODEL_DIR),
        name_prefix="charging_dqn_priority",
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=20_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        seed=42,
        device="cpu",
        tensorboard_log=str(LOG_DIR),
    )

    print("Starting priority-based DQN training...")

    try:
        model.learn(
            total_timesteps=20_000,
            callback=checkpoint_callback,
            progress_bar=False,
        )

        final_path = MODEL_DIR / "charging_dqn_priority_final"
        model.save(str(final_path))

        print(
            f"Training complete. Model saved to: {final_path}.zip"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()

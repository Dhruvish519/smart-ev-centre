import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from Chargym_Charging_Station.envs.gymnasium_wrapper import (
    ChargingGymnasiumWrapper,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "ppo"
LOG_DIR = PROJECT_ROOT / "logs" / "ppo"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    env = Monitor(ChargingGymnasiumWrapper(), str(LOG_DIR))

    checkpoint_callback = CheckpointCallback(
        save_freq=5_000,
        save_path=str(MODEL_DIR),
        name_prefix="charging_ppo",
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=42,
        device="cpu",
        tensorboard_log=str(LOG_DIR),
    )

    print("Starting PPO training...")
    model.learn(
        total_timesteps=10_000,
        callback=checkpoint_callback,
        progress_bar=False,
    )

    final_path = MODEL_DIR / "charging_ppo_final"
    model.save(str(final_path))

    print(f"Training complete. Model saved to: {final_path}.zip")
    env.close()


if __name__ == "__main__":
    main()


import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gym
import Chargym_Charging_Station

def main():
    env = gym.make("ChargingEnv-v0")
    print("Chargym environment created successfully.")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    env.close()

if __name__ == "__main__":
    main()

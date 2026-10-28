from gymnasium.envs.registration import register

register(
    id="ChargingEnv-v0",
    entry_point="Chargym_Charging_Station.envs.Charging_Station_Enviroment:ChargingEnv",
    max_episode_steps=200,
)

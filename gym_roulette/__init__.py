from gym.envs.registration import register

register(
    id='roulette-v1',
    entry_point='gym_roulette.envs:RouletteEnv',
)
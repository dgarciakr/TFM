from gym.envs.registration import register


register(
    id='Mapping-v0',
    entry_point='gym_hpc.envs:MappingEnvSec',
)

register(
    id='Mapping-v1',
    entry_point='gym_hpc.envs:MappingEnvSwitch',
)

register(
    id='Colls-v0',
    entry_point='gym_hpc.envs:CollsEnv',
)

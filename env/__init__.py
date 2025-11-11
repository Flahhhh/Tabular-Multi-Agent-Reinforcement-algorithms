from env.configs import MatrixEnvCfg,GridV1EnvCfg,GridV2EnvCfg,GridV3EnvCfg
from env.envs import MatrixEnv,GridV1,GridV2,GridV3

envs = {
    "GridV1": GridV1(GridV1EnvCfg()),
    "GridV2": GridV2(GridV2EnvCfg()),
    "GridV3": GridV3(GridV3EnvCfg()),
    "Matrix": MatrixEnv(MatrixEnvCfg()),
}

def get_env(name):
    env = envs.get(name)
    if not env:
        raise NameError("Invalid environment")

    return env
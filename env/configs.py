from dataclasses import dataclass

@dataclass
class GridV1EnvCfg:
    name:str = "GridV1"
    num_agents:int = 2
    state_space:int = 5
    action_space:int = 2

    init_positions:tuple[int] = (1,3)
    target_pos:int = 2

    target_reward_1:float = 5.
    catch_reward_1:float = -10.
    catch_reward_2:float = 5.

    max_steps:int = 8

@dataclass
class GridV2EnvCfg:
    name:str = "GridV2"
    num_agents:int = 2
    state_space:int = 8
    action_space:int = 3

    init_positions:tuple[int] = (1, 2)
    barriers:tuple[int] = (5, 6)

    collision_reward:float = -1.

    max_steps:int = 2

@dataclass
class GridV3EnvCfg:
    name:str = "GridV3"
    num_agents:int = 2
    state_space:int = 9
    action_space:int = 4

    init_positions:tuple[int] = (0, 2)
    target_pos:int = 7

    collision_reward:float = -1.
    target_reward:float = 100.

    max_steps:int = 4

@dataclass
class MatrixEnvCfg:
    name:str = "Matrix"
    num_agents:int = 2
    state_space:int = 1
    action_space:int = 2
from dataclasses import dataclass

@dataclass
class NashQCfg:
    name:str = "NashQ"
    env_name:str = "GridV2"

    log_dir:str = None
    model_dir:str = None

    lr: float = 0.2
    discount:float = 0.99
    start_eps:float = 0.9
    min_eps:float = 0.05

    num_epoch:int = 1000
    num_agents:int = 2
    num_states:int = 8
    num_actions:int = 3

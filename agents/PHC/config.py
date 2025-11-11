from dataclasses import dataclass

@dataclass
class PHCCfg:
    name: str = "PHC"
    env_name:str = "Matrix"

    log_dir: str = None
    model_dir:str = None

    lr:float = 0.2
    policy_lr:float = 0.2
    discount:float = 0.99
    start_eps:float = 0.5
    min_eps:float = 0.01

    num_epoch:int = 1000
    num_agents:int = 2
    num_states:int = 1
    num_actions:int = 2
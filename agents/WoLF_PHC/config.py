from dataclasses import dataclass

@dataclass
class WoLFPHCCfg:
    name: str = "WoLFPHC"
    env_name:str = "Matrix"

    log_dir: str = None
    model_dir:str = None

    lr:float = 0.1
    policy_lr_w:float = 0.2
    policy_lr_l:float = 0.4
    discount:float = 0.99
    start_eps:float = 0.5
    min_eps:float = 0.01

    num_epoch:int = 1000
    num_agents:int = 2
    num_states:int = 1
    num_actions:int = 2
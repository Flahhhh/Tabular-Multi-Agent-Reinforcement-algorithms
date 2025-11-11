from dataclasses import asdict

import numpy as np

class BaseEnv:
    finished: bool
    steps: int
    max_steps: int

    cfg = None

    def load_config(self, cfg):
        self.cfg = cfg

        for attr, value in vars(cfg).items():
            setattr(self, attr, value)

    def reset(self) -> list[int]:
        raise NotImplementedError()

    def get_legals(self) -> list[int]:
        raise NotImplementedError()

    def get_state(self) -> list[int]:
        raise NotImplementedError()

    def step(self, actions: list[int]) -> tuple[list[int], list[float | np.float16], bool, list[list[int]]]:
        raise NotImplementedError()

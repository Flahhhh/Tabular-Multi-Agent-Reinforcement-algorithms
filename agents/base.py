import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
from tqdm import tqdm

from agents.utils import save_json, save_np


class BaseAgent:
    name: str
    env_name: str

    log_dir: str
    model_dir: str

    finished: bool
    reward: np.ndarray[float]
    next_state: list[int]
    action: list[int]
    state: list[int]
    legals: list[list[int]]

    num_epoch: int
    Q: np.ndarray[tuple[int, int, int] | tuple[int, int, int, int]]

    cfg = None

    def load_config(self, cfg):
        self.cfg = cfg

        for attr, value in vars(cfg).items():
            setattr(self, attr, value)

    def get_action(self) -> list[int]:
        raise NotImplementedError()

    def update(self) -> np.ndarray:
        raise NotImplementedError()

    def test(self, env):
        raise NotImplementedError()

    def train(self, env):
        logs = {"rewards": [], "loss": []}

        dir_name = f"{self.name}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}"
        log_dir = os.path.join("logs",dir_name) if not self.log_dir else os.path.join(self.log_dir,dir_name)
        model_dir = os.path.join(log_dir,"Models") if not self.model_dir else self.model_dir
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        for _ in tqdm(range(self.num_epoch)):
            ep_reward,ep_loss = np.array([0,0],dtype=np.float32),np.array([0,0],dtype=np.float32)

            self.state,self.legals = env.reset()
            self.finished = False
            while not self.finished:
                self.action = self.get_action()

                self.next_state,self.reward,self.finished,self.legals = env.step(self.action)
                loss = np.abs(self.update())

                self.state = self.next_state
                ep_loss+=loss
                ep_reward+=self.reward

            logs["rewards"].append(ep_reward.tolist())
            logs["loss"].append((ep_loss/env.steps).tolist())

        if log_dir:
            save_json(logs, os.path.join(log_dir, "logs.json"))
            save_json(asdict(self.cfg), os.path.join(log_dir, "agent_config.json"))
            save_json(asdict(env.cfg), os.path.join(log_dir, "env_config.json"))

        if model_dir:
            save_np(self.Q, os.path.join(model_dir, "model.npy"))



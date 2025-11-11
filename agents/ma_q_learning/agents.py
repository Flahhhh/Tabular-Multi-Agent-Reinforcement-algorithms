import random
import numpy as np

from agents.base import BaseAgent
from agents.ma_q_learning.config import MAQCfg


class MAQ(BaseAgent):
    name: str
    log_dir: str
    model_dir: str

    lr: float
    discount: float
    start_eps: float
    min_eps: float
    num_epoch: int

    num_agents: int
    num_states: int
    num_actions: int

    def __init__(self, cfg:MAQCfg):
        self.legals = None
        
        self.load_config(cfg)
        self.Q = np.zeros((self.num_agents, self.num_states, self.num_actions))

        self.eps = self.start_eps
        self.eps_decay_ratio = (self.start_eps-self.min_eps)/self.num_epoch

    def get_action(self) -> list[int]:
        actions = []

        for agent_idx in range(self.num_agents):
            legals = self.legals[agent_idx]

            if np.random.rand()<self.eps:
                action = random.choice(legals)
            else:
                legals_to_actions = dict(zip(list(range(len(legals))), legals))
                legal_action = self.Q[agent_idx, self.state[agent_idx], legals].argmax()

                action = legals_to_actions[int(legal_action)]

            actions.append(action)

        return actions

    def update(self) -> np.ndarray:
        total_loss = np.array([0,0],dtype=np.float32)
        for agent_idx in range(self.num_agents):
            prediction = self.Q[agent_idx, self.state[agent_idx], self.action[agent_idx]]
            if not self.finished:
                loss = self.reward[agent_idx] + self.discount*self.Q[agent_idx, self.next_state[agent_idx]].max() - prediction
            else:
                loss = self.reward[agent_idx] - prediction

            total_loss[agent_idx] = loss
            self.Q[agent_idx, self.state[agent_idx], self.action[agent_idx]] += self.lr*loss

        self.update_eps()

        return total_loss

    def update_eps(self):
        self.eps = max(self.min_eps, self.eps-self.eps_decay_ratio)

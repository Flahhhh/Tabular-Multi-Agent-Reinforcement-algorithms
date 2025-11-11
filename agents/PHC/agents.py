import numpy as np

from agents.PHC.config import PHCCfg
from agents.base import BaseAgent
from const import EPS

class PHC(BaseAgent):
    name: str
    log_dir: str
    model_dir: str

    lr: float
    policy_lr: float
    discount: float
    start_eps: float
    min_eps: float
    num_epoch: int

    num_agents: int
    num_states: int
    num_actions: int

    def __init__(self, cfg: PHCCfg):
        self.load_config(cfg)
        self.Q = np.zeros((self.num_agents, self.num_states, self.num_actions))
        self.pi = np.full((self.num_agents, self.num_states, self.num_actions), 1/self.num_actions)

        self.eps = self.start_eps
        self.eps_decay_ratio = (self.start_eps - self.min_eps) / self.num_epoch

    def get_action(self) -> list[int]:
        actions = []

        for agent_idx in range(self.num_agents):
            actions.append(np.random.choice(self.num_actions, p=self.pi[agent_idx, self.state[agent_idx], :]))

        return actions

    def update_q(self):
        total_loss = np.array([0,0],dtype=np.float32)

        for agent_idx in range(self.num_agents):
            target = self.reward[agent_idx] + 0 if self.finished else self.discount * self.Q[agent_idx, self.next_state[agent_idx]].max()
            loss = target - self.Q[agent_idx, self.state[agent_idx], self.action[agent_idx]]
            self.Q[agent_idx, self.state[agent_idx], self.action[agent_idx]] += self.lr * loss

            total_loss[agent_idx] = loss

        return total_loss

    def pi_norm(self):
        norm = self.pi[:, self.state, :].sum(2, keepdims=True) + EPS

        self.pi[:, self.state, :] /= norm

    def add_min_eps(self):
        min_prob = self.eps / self.num_actions
        self.pi[:, self.state] = np.maximum(self.pi[:, self.state], min_prob)

    def update_pi(self):
        for agent_idx in range(self.num_agents):
            best_action = self.Q[agent_idx, self.state[agent_idx]].argmax()

            for action in range(self.num_actions):
                if action==best_action:
                    delta = self.policy_lr
                else:
                    #1 action err
                    #for small policy values
                    delta = -min(float(self.pi[agent_idx, self.state[agent_idx], action]), self.policy_lr/(self.num_actions-1))

                self.pi[agent_idx, self.state[agent_idx], action] += delta


        self.add_min_eps() #PHC eps strategy
        self.pi_norm() #normalize policy after update

    def update(self) -> np.ndarray:
        q_loss = self.update_q()
        self.update_pi()
        self.update_eps()

        return q_loss

    def update_eps(self):
        self.eps = max(self.min_eps, self.eps-self.eps_decay_ratio)



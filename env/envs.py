import env.configs
from env.base import BaseEnv
import numpy as np

class GridV1(BaseEnv):
    agent_positions: list[int]
    init_positions: list[int]
    target_pos: int
    target_reward_1: float
    catch_reward_1: float
    catch_reward_2: float

    action_space: int
    state_space: int
    num_agents: int

    def __init__(self, cfg: env.configs.GridV1EnvCfg):
        self.load_config(cfg)

        self.T = {
            0: {
                0: 4,
                1: 1,
            },
            1: {
                0: 0,
                1: 2,
            },
            2: {
                0: 1,
                1: 3,
            },
            3: {
                0: 2,
                1: 4,
            },
            4: {
                0: 3,
                1: 0,
            },
        }
        self.reset()

    def reset(self):
        self.agent_positions = list(self.init_positions)
        self.steps = 0
        self.finished = False

        return self.agent_positions, self.get_legals()

    def get_state(self):
        return self.agent_positions

    def get_legals(self):
        return [list(self.T[self.agent_positions[agent_idx]].keys()) for agent_idx in range(self.num_agents)]

    def _apply_action(self, agent_idx: int, action: int):
        new_pos = self.T[self.agent_positions[agent_idx]].get(action)
        reward = np.array([0, 0], dtype=np.float16)
        if new_pos is None:
            raise ValueError(f"Invalid agent_{agent_idx} action")

        self.agent_positions[agent_idx] = new_pos
        if agent_idx == 0:
            if new_pos == self.target_pos:
                reward[0] += 5
        elif agent_idx == 1:
            if new_pos == self.agent_positions[0]:
                self.finished = True
                reward += np.array([-10., 5.], dtype=np.float16)

        return new_pos, reward

    def step(self, actions: list[int]):
        self.steps += 1

        next_state_1, reward = self._apply_action(0, actions[0])
        next_state_2, reward_ = self._apply_action(1, actions[1])

        reward += reward_
        truncated = self.steps >= self.max_steps

        return self.get_state(), reward, truncated or self.finished, self.get_legals()

class GridV2(BaseEnv):
    barriers: list[int]
    agent_positions: list[int]
    init_positions: list[int]
    collision_reward: float

    action_space: int
    state_space: int
    num_agents: int

    def __init__(self, cfg: env.configs.GridV2EnvCfg):
        self.load_config(cfg)

        self.T = {
            0: {
                1: 1,
                2: 4,
            },
            1: {
                0: 0,
                1: 2,
                2: 5,
            },
            2: {
                0: 1,
                1: 3,
                2: 6,
            },
            3: {
                0: 2,
                2: 7,
            },
            4: {
                1: 5,
            },
            5: {
                0: 4,
                1: 6,
            },
            6: {
                0: 5,
                1: 7,
            },
            7: {
                0: 6,
            },
        }

    def reset(self):
        self.agent_positions = list(self.init_positions)
        self.steps = 0
        self.finished = False

        return self.agent_positions, self.get_legals()

    def _apply_action(self, agent_idx: int, action: int):
        new_pos = self.T[self.agent_positions[agent_idx]].get(action)
        reward = 0.

        if new_pos is None:
            raise ValueError(f"Invalid agent_{agent_idx} action")
        if new_pos in self.barriers:
            self.finished = True
            reward += self.collision_reward

        self.agent_positions[agent_idx] = new_pos
        return new_pos, reward

    def get_state(self):
        return self.agent_positions

    def get_legals(self):
        return [list(self.T[self.agent_positions[agent_idx]].keys()) for agent_idx in range(self.num_agents)]

    def step(self, actions: list[int]):
        self.steps += 1

        next_state_1, reward_1 = self._apply_action(0, actions[0])
        next_state_2, reward_2 = self._apply_action(1, actions[1])

        reward = np.array([reward_1,reward_2], dtype=np.float16)
        if not self.finished and next_state_1 == next_state_2:
            self.finished = True
            reward += self.collision_reward

        truncated = self.steps >= self.max_steps

        return self.get_state(), reward, truncated or self.finished, self.get_legals()

class GridV3(BaseEnv):
    agent_positions: list[int]
    init_positions: list[int]
    target_pos: int
    collision_reward: float
    target_reward: float

    action_space: int
    state_space: int
    num_agents: int

    def __init__(self, cfg: env.configs.GridV3EnvCfg):
        self.load_config(cfg)

        self.T = {
            0: {
                1: 1,
                2: 3,
            },
            1: {
                0: 0,
                1: 2,
                2: 4,
            },
            2: {
                0: 1,
                2: 5,
            },
            3: {
                1: 4,
                2: 6,
                3: 0,
            },
            4: {
                0: 3,
                1: 5,
                2: 7,
                3: 1,
            },
            5: {
                0: 4,
                2: 8,
                3: 2,
            },
            6: {
                1: 7,
                3: 3,
            },
            7: {
                0: 6,
                1: 8,
                3: 4,
            },
            8: {
                0: 7,
                3: 5,
            }
        }

    def reset(self):
        self.agent_positions = list(self.init_positions)
        self.steps = 0
        self.finished = False

        return self.agent_positions, self.get_legals()

    def _apply_action(self, agent_idx: int, action: int):
        new_pos = self.T[self.agent_positions[agent_idx]].get(action)
        reward = 0.

        if new_pos is None:
            raise ValueError(f"Invalid agent_{agent_idx} action")

        if new_pos == self.target_pos:
            reward += self.target_reward
            self.finished = True

        self.agent_positions[agent_idx] = new_pos

        return new_pos, reward

    def get_state(self):
        return self.agent_positions

    def get_legals(self):
        return [list(self.T[self.agent_positions[agent_idx]].keys()) for agent_idx in range(self.num_agents)]

    def step(self, actions: list[int]):
        self.steps+=1

        next_state_1, reward_1 = self._apply_action(0, actions[0])
        next_state_2, reward_2 = self._apply_action(1, actions[1])

        reward = np.array([reward_1,reward_2],dtype=np.float16)
        if next_state_1==next_state_2 and not self.finished:
            reward += self.collision_reward

        truncated = self.steps >= self.max_steps

        return self.get_state(), reward, truncated or self.finished, self.get_legals()

class MatrixEnv(BaseEnv):
    action_space: int
    state_space: int
    num_agents: int
    def __init__(self, cfg:env.configs.MatrixEnvCfg):
        self.load_config(cfg)

        self.R = np.array([[[10,0],[-10,1]],[[10,-10],[0,1]]], dtype=np.float16)

    def get_state(self):
        return [0, 0]

    def reset(self):
        self.steps = 1

        return self.get_state(),self.get_legals()

    def get_legals(self):
        return [[0,1], [0,1]]

    def step(self, actions: list[int]):
        reward_1 = self.R[0][actions[0],actions[1]]
        reward_2 = self.R[1][actions[0],actions[1]]

        reward = np.array([reward_1,reward_2],dtype=np.float16)
        finished = True

        return self.get_state(),reward,finished,self.get_legals()
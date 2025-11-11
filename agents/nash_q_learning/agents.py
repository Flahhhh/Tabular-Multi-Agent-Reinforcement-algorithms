import numpy as np
import nashpy

from agents.nash_q_learning.config import NashQCfg
from agents.base import BaseAgent

#ONLY FOR 2 AGENTS
class NashQ(BaseAgent):
    log_dir: str
    model_dir: str

    lr: float
    discount: float
    start_eps: float
    min_eps: float

    num_agents: int
    num_states: int
    num_actions: int

    def __init__(self, cfg: NashQCfg):
        self.legals = None

        self.load_config(cfg)
        self.Q = np.zeros((self.num_agents, self.num_states, self.num_actions, self.num_actions))
        self.C = np.zeros((self.num_agents, self.num_states, self.num_actions, self.num_actions))

        self.eps = self.start_eps
        self.eps_decay_ratio = (self.start_eps - self.min_eps) / self.num_epoch

    def get_action(self) -> list[int]:
        actions = []

        # Используем эпсилон-жадную стратегию
        if np.random.rand() < self.eps:
            action1 = np.random.choice(self.legals[0])
            action2 = np.random.choice(self.legals[1])
        else:
            # В режиме эксплуатации выбираем действия, которые являются частью равновесия Нэша
            # (Для простоты выберем одно из чистых или смешанных действий из найденного равновесия)

            # В идеале, если равновесие смешанное, агент должен выбирать действия стохастически
            # в соответствии с распределением вероятностей pi1_star и pi2_star.
            # Для простоты пока используем детерминированный выбор из первого равновесия:

            payoff_matrix_1 = self.Q[0][self.state[0]][np.ix_(self.legals[0], self.legals[1])]
            payoff_matrix_2 = self.Q[1][self.state[1]][np.ix_(self.legals[0], self.legals[1])]
            game = nashpy.Game(payoff_matrix_1, payoff_matrix_2)
            equilibria = list(game.support_enumeration())

            if equilibria:
                pi1_star, pi2_star = equilibria[0]
                # Выбираем действие на основе распределения вероятностей
                legal_idx_1 = np.random.choice(len(self.legals[0]), p=pi1_star)
                legal_idx_2 = np.random.choice(len(self.legals[1]), p=pi2_star)
                action1 = self.legals[0][legal_idx_1]
                action2 = self.legals[1][legal_idx_2]
            else:
                # Запасной вариант, если равновесие не найдено
                action1 = np.random.choice(self.legals[0])
                action2 = np.random.choice(self.legals[1])

        actions = [action1, action2]

        return actions

    def _init_pi(self):
        self.pi_1 = np.zeros((self.num_actions*self.num_actions, self.num_actions))
        self.pi_1_op = np.zeros((self.num_actions*self.num_actions, self.num_actions))
        self.pi_2 = np.zeros((self.num_actions*self.num_actions, self.num_actions))
        self.pi_2_op = np.zeros((self.num_actions*self.num_actions, self.num_actions))

    def _solve_nash_equilibrium(self):
        """
        Вычисляет значение равновесия Нэша (V1*, V2*) для заданного состояния.
        """
        # Создаем матрицу выплат, ограниченную легальными действиями
        # Q[s, a1, a2]
        payoff_matrix_1 = self.Q[0][self.state[0]][np.ix_(self.legals[0], self.legals[1])]
        payoff_matrix_2 = self.Q[1][self.state[1]][np.ix_(self.legals[0], self.legals[1])]

        # Создаем игру в nashpy
        game = nashpy.Game(payoff_matrix_1, payoff_matrix_2)

        # Находим одно из равновесий Нэша (например, с помощью поддержки перечисления)
        # В играх с общей суммой часто ищут глобальный оптимум.
        # Support Enumeration находит все, мы берем первое
        equilibria = list(game.support_enumeration())

        if not equilibria:
            # Если равновесие не найдено (что редкость для стандартных игр),
            # можно использовать запасной вариант, например, max-max для cooperative games.
            # Здесь мы просто вернем 0 или бросим ошибку
            return None, None

        # Берем первое найденное равновесие
        pi1_star, pi2_star = equilibria[0]

        # Вычисляем ожидаемое значение (Value) этого равновесия
        # V_i*(s) = sum_{a1, a2} pi1*(a1) * pi2*(a2) * Q_i(s, a1, a2)
        v1_star = np.dot(pi1_star, np.dot(payoff_matrix_1, pi2_star))
        v2_star = np.dot(pi2_star, np.dot(payoff_matrix_2.T, pi1_star))  # Транспонируем для матричного умножения

        return v1_star, v2_star


    def update(self):
        # --- Основная логика обновления Nash Q-learning ---

        # 1. Вычислить значение равновесия Нэша V*(s') для следующего состояния
        if self.finished:
            v1_star, v2_star = 0, 0
        else:
            # Здесь предполагается, что легальные действия в s' доступны (должны передаваться в update или храниться)
            # Для простоты предположим, что все действия всегда легальны в этом примере

            v1_star, v2_star = self._solve_nash_equilibrium()

            if v1_star is None:
                # Обработка ошибки
                v1_star, v2_star = 0, 0

        # 2. Вычислить целевое значение (Bellman target)
        target1 = self.reward[0] + self.discount * v1_star
        target2 = self.reward[1] + self.discount * v2_star

        # 3. Обновить Q-значения с помощью learning rate
        loss_1 = target1 - self.Q[0][self.state[0], self.action[0], self.action[1]]
        loss_2 = target2 - self.Q[1][self.state[1], self.action[0], self.action[1]]

        self.Q[0][self.state[0], self.action[0], self.action[1]] += self.lr * loss_1
        self.Q[1][self.state[1], self.action[0], self.action[1]] += self.lr * loss_2

        self.update_eps()


        return np.array([loss_1, loss_2])

    #def update(self):
    #    q_loss = self.update_q()
    #    self.update_eps()
    #
    #    return q_loss

    def update_eps(self):
        self.eps = max(self.min_eps, self.eps-self.eps_decay_ratio)

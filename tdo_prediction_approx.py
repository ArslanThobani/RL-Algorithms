# -*- coding: utf-8 -*-
"""
Created on 7/25/2020

@author: Arslan Thobani
"""
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid
from policy_evaluation import print_values, print_policy
from td0_prediction import ALPHA, play_game, GAMMA


class Model:
    def __init__(self):
        self.theta = np.random.randn(4) / 2

    def s2x(self, s):
        return np.array([s[0] - 1, s[1] - 1.5, s[0] * s[1] - 3, 1])

    def predict(self, s):
        x = self.s2x(s)
        return self.theta.dot(x)

    def grad(self, s):
        return self.s2x(s)


if __name__ == '__main__':
    # use the standard grid again (0 for every step) so that we can compare
    # to iterative policy evaluation
    grid = standard_grid()

    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    # state -> action
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }

    model = Model()
    deltas = []

    k = 1.0
    # repeat until convergence
    for it in range(30000):
        if it % 10 == 0:
            k = k + 0.01
        alpha = ALPHA / k
        biggest_change = 0

        # generating an episode using pi policy
        states_and_rewards = play_game(grid, policy)
        # the first (s, r) tuple is the state we start in and 0
        # (since we don't get a reward) for simply starting the game
        # the last (s, r) tuple is the terminal state and the final reward
        # the value for the terminal state is by definition 0, so we don't
        # care about updating it.

        for t in range(len(states_and_rewards) - 1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t+1]

            old_theta = model.theta.copy()
            if grid.is_terminal(s2):
                target = r
            else:
                target = r + GAMMA*model.predict(s2)
            model.theta += alpha*(target - model.predict(s))*model.grad(s)
            biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    # obtain predicted values
    V = {}
    for s in grid.all_states():
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            # terminal state or state which is inaccessible
            V[s] = 0

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
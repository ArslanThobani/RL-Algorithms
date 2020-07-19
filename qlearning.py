# -*- coding: utf-8 -*-
"""
Created on 7/19/2020

@author: Arslan Thobani
"""

import matplotlib.pyplot as plt
import numpy as np

from grid_world import negative_grid
from policy_evaluation import print_values, print_policy
from td0_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    # put this into a function since we are using it so often
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

if __name__ == '__main__':

    grid = negative_grid()

    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0

    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0

    t = 1.0
    deltas = []
    for it in range(10000):
        if it % 100 == 0:
            t += 10e-3
        if it % 2000 == 0:
            print('it:', it)

        s = (2, 0)
        grid.set_state(s)

        a = random_action(a, eps=0.5 / t)
        biggest_change = 0
        while not grid.game_over():
            a = random_action(a, eps=0.5 / t)
            r = grid.move(a)
            s1 = grid.current_state()

            a1, max_q1a1 = max_dict(Q[s1])
            a1 = random_action(a1, eps=0.5 / t)

            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005
            old_qsa = Q[s][a]

            Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * max_q1a1 - Q[s][a])

            biggest_change = max(biggest_change, np.abs(Q[s][a] - old_qsa))

            s = s1
            a = a1

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    # what's the proportion of time we spend updating each part of Q?
    print("update counts:")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)

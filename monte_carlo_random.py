# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:29:03 2020

@author: Arslan Thobani
"""
from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_world import standard_grid, negative_grid
from policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4 # threshold for convergence
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U','D','L','R')

def random_action(a):
    
    if np.random.random() < 0.5:
        return a
    else:
        tmp = list(ALL_POSSIBLE_ACTIONS)
        tmp.remove(a)
        return np.random.choice(tmp)


def play_game(grid: standard_grid, policy):
    #Playing one episode
    #Exploring states method
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])
    
    s = grid.current_state()
    states_and_rewards = [(s, 0)]
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))
    
    first = True
    G = 0
    states_and_returns = []
    for s, r in reversed(states_and_rewards):
        if first:
            first = False
        else:
            states_and_returns.append((s, G))
        G = r + GAMMA*G
    
    states_and_returns.reverse()
    return states_and_returns

if __name__ == '__main__':
    # use the standard grid again (0 for every step) so that we can compare
    # to iterative policy evaluation
    grid = standard_grid()
      
    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)
      
    # state -> action
    # found by policy_iteration_random on standard_grid
    # MC method won't get exactly this, but should be close
    # values:
    # ---------------------------
    #  0.43|  0.56|  0.72|  0.00|
    # ---------------------------
    #  0.33|  0.00|  0.21|  0.00|
    # ---------------------------
    #  0.25|  0.18|  0.11| -0.17|
    # policy:
    # ---------------------------
    #   R  |   R  |   R  |      |
    # ---------------------------
    #   U  |      |   U  |      |
    # ---------------------------
    #   U  |   L  |   U  |   L  |
    policy = {
      (2, 0): 'U',
      (1, 0): 'U',
      (0, 0): 'R',
      (0, 1): 'R',
      (0, 2): 'R',
      (1, 2): 'U',
      (2, 1): 'L',
      (2, 2): 'U',
      (2, 3): 'L',
    }
    
    
    V = {}
    returns = {} # dictionary of state -> list of returns we've received
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0
            
    for t in range(100):
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns:
            if s not in seen_states:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                seen_states.add(s)
                
                
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
      





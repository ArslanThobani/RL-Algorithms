# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:09:59 2020

@author: Arslan Thobani
"""

from __future__ import print_function, division
from builtins import range

import matplotlib.pyplot as plt
import numpy as np
from grid_world import standard_grid, negative_grid
from policy_evaluation import print_values, print_policy

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U','D','L','R')

def play_game(grid: standard_grid, policy):
    #Playing one episode
    #Exploring states method
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])
    
    s = grid.current_state()
    a = np.random.choice(ALL_POSSIBLE_ACTIONS)
    
    states_actions_rewards = [(s, a, 0)]
    # seen_states = set()
    # seen_actions.add(grid.current_state())
    # num_steps = 0
    while True:
        old_s = grid.current_state()
        r = grid.move(a)
        # num_steps += 1
        s = grid.current_state()
        if old_s == s:
            states_actions_rewards.append((s, None, -100))
            break
        elif grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = policy[s]
            states_actions_rewards.append((s, a, r))
    
    
        
    first = True
    G = 0
    states_actions_returns = []
    for s, a, r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA*G
    
    states_actions_returns.reverse()
    return states_actions_returns

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
    # use the standard grid again (0 for every step) so that we can compare
    # to iterative policy evaluation
    grid = negative_grid(step_cost=-0.1)
      
    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)
    
    #Initialize a random policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    
    #Initialize Q(s, a) and returns
    Q = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0 # needs to be initialized so that we can perform argmax
                returns[(s, a)] = []
        else:
            #terminal state or not reachable state
            pass
        
    deltas = []
    for t in range(2000):
        # if t % 100 == 0:
        #     print(t)
        print(t)
        
        # generate an episode using pi
        biggest_change = 0
        states_actions_returns = play_game(grid, policy)
        seen_state_action_pairs = set()
        for s, a, G in states_actions_returns:
            
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)
        
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]
            
    plt.plot(deltas)
    plt.show()

    print("final policy:")
    print_policy(policy, grid)
    
    V = {}
    for s, Qs in Q.items():
      V[s] = max_dict(Q[s])[1]
    
    print("final values:")
    print_values(V, grid)
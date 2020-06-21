# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:30:33 2020

@author: Arslan Thobani
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_VALUES = ('U', 'D', 'L', 'R')

if __name__ == '__main__':
    
    grid = negative_grid(step_cost=-1.0)
    
    print("rewards")
    print_values(grid.rewards, grid)
    
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_VALUES)
        
    print("Initial Policy")
    print_policy(policy, grid)
    
    V = {}
    states = grid.all_states()
    for s in states:
        # V[s] = 0
        if s in grid.actions: # check if non terminal state
            V[s] = np.random.random()
        else:
            V[s] = 0 # Terminal state V is 0 
            
    print("Initial V")
    print_values(V, grid)
    
    while True:
        #Policy evaluation
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]
                new_v = 0                
                if s in policy:
                    for a in ALL_POSSIBLE_VALUES:
                        if a == policy[s]:
                            p = 0.5
                        else:
                            p = 0.5/3
                        
                        grid.set_state(s)
                        reward = grid.move(a)
                        new_v += p*(reward + GAMMA*V[grid.current_state()])
                    V[s] = new_v
                    biggest_change = max(biggest_change, np.abs(V[s] - old_v))
            if biggest_change < SMALL_ENOUGH:
                break
        
        #Policy Improvement
        is_policy_converged = True

        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')
                for a in ALL_POSSIBLE_VALUES:
                    new_v = 0
                    for a2 in ALL_POSSIBLE_VALUES:
                        if a == a2:
                            p = 0.5
                        else:
                            p = 0.5/3
                        grid.set_state(s)
                        reward = grid.move(a2)
                        new_v += p*(reward + GAMMA*V[grid.current_state()])
                    if new_v > best_value:
                        best_value = new_v
                        new_a = a
                policy[s] = new_a
                if new_a != old_a:
                    is_policy_converged = False
        if is_policy_converged:
            break

    print("final V")
    print_values(V, grid)
    print("final policy")
    print_policy(policy, grid)
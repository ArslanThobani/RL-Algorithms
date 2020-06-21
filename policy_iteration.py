# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:53:01 2020

@author: Arslan Thobani
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_VALUES = ('U', 'D', 'L', 'R')

if __name__ == '__main__':
    
    grid = negative_grid(step_cost=-0.1)
    
    print("rewards")
    print_values(grid.rewards, grid)
    
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_VALUES)
        
    print("Initial Policy")
    print_policy(policy, grid)
    
    V = {}
    for s in grid.all_states():
        V[s] = 0
        # if s in grid.actions.keys(): # check if non terminal state
        #     V[s] = np.random.random()
        # else:
        #     V[s] = 0 # Terminal state V is 0 
            
    print("Initial V")
    print_values(V, grid)
    
    while True:
        is_policy_converged = True
        
        #Policy evaluation
        while True:
            biggest_change = 0
            for s in grid.all_states():
                old_v = V[s]
                
                if s in policy:
                    a = policy[s]
                    grid.set_state(s)
                    reward = grid.move(a)
                    V[s] = reward + GAMMA*V[grid.current_state()]
                    biggest_change = max(biggest_change, np.abs(V[s] - old_v))
                 
            if biggest_change < SMALL_ENOUGH:
                break
        
        #Policy Improvement
        
        for s in grid.actions.keys():
            old_a = policy[s]
            new_a = None
            
            best_value = float('-inf')
            for a in grid.actions[s]:
                grid.set_state(s)
                reward = grid.move(a)
                new_v = reward + GAMMA*V[grid.current_state()]
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
                
            
            
            
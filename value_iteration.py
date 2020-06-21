# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:35:17 2020

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
    
    # Repeat until convergence
    while True:
        biggest_change = 0
        for s in grid.all_states():
            old_v = V[s]
            new_v = float('-inf')
            if s in policy:
                for a in ALL_POSSIBLE_VALUES:
                    grid.set_state(s)
                    reward = grid.move(a)
                    v = reward + GAMMA*V[grid.current_state()]
                    if v > new_v:
                        new_v = v
                
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(V[s] - old_v))
                
        if biggest_change < SMALL_ENOUGH:
            break
    
    # Find a policy that leads to optimal value function
    for s in policy.keys():
        best_a = None
        best_v = float('-inf')
        
        for a in ALL_POSSIBLE_VALUES:
            grid.set_state(s)
            r = grid.move(a)
            v = r + GAMMA*V[grid.current_state()]
            if v > best_v:
                best_v = v
                best_a = a
        policy[s] = best_a
        
    
    print("final V")
    print_values(V, grid)
    print("final policy")
    print_policy(policy, grid)
                
            
            
            